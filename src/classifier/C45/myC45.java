package classifier.C45;

import classifier.ID3.myID3;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by nathanjamesruntuwene on 9/20/17.
 */
public class myC45 extends AbstractClassifier {
    private DTLNode root;
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();   // returns the object from weka.classifiers.Classifier

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        root = new DTLNode();
        Instances instancesCopy = new Instances(instances);
        fillMissingValue(instancesCopy);
        root.buildTree(instancesCopy);
    }

    private void fillMissingValue(Instances instances) {
        @SuppressWarnings("unchecked")
        HashMap<Double, Integer>[] counter = new HashMap[instances.numAttributes()];
        Double[] popularAttribute = new Double[instances.numAttributes()];
        Integer[] maxCounter = new Integer[instances.numAttributes()];
        Enumeration<Instance> instanceEnumeration = instances.enumerateInstances();
        while (instanceEnumeration.hasMoreElements()) {
            Instance instance = instanceEnumeration.nextElement();
            Enumeration<Attribute> attributeEnumeration = instance.enumerateAttributes();
            while (attributeEnumeration.hasMoreElements()) {
                Attribute attribute = attributeEnumeration.nextElement();
                if (instance.isMissing(attribute))
                    continue;
                counter[attribute.index()].put(instance.value(attribute), counter[attribute.index()].getOrDefault(instance.value(attribute), 0) + 1);
                if (counter[attribute.index()].get(instance.value(attribute)) > maxCounter[attribute.index()]) {
                    maxCounter[attribute.index()] = counter[attribute.index()].get(instance.value(attribute));
                    popularAttribute[attribute.index()] = instance.value(attribute);
                }
            }
        }
        instanceEnumeration = instances.enumerateInstances();
        while (instanceEnumeration.hasMoreElements()) {
            Instance instance = instanceEnumeration.nextElement();
            Enumeration<Attribute> attributeEnumeration = instance.enumerateAttributes();
            while (attributeEnumeration.hasMoreElements()) {
                Attribute attribute = attributeEnumeration.nextElement();
                if (instance.isMissing(attribute)) {
                    instance.setValue(attribute, popularAttribute[attribute.index()]);
                }
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return root.classify(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }
}

class DTLNode {
    private boolean isLeaf = false;
    private Double classifiedClass;
    private Attribute attributeToClassify;
    private HashMap<Double, DTLNode> children;
    private DTLNode popularChild;
    private Double threshold;
    private DTLNode parent = null;

    DTLNode() {
        children = new HashMap<>();
    }

    public DTLNode getParent() {
        return parent;
    }

    // hackish method to swap, see https://stackoverflow.com/a/16826296
    private double returnFirst(double x, double y) {
        return x;
    }

    void buildTree(Instances instances) {
        if (instances.isEmpty()) {
            throw new Error("EMPTY INSTANCES");
        }
        if (instances.numDistinctValues(instances.classAttribute()) == 1) {
            this.isLeaf = true;
            classifiedClass = instances.firstInstance().classValue();
            return;
        }
        if (instances.numAttributes() <= 1) {
            makeLeaf(instances);
            return;
        }
        if (calcInformationGainMax(instances) == 0) {
            makeLeaf(instances);
            return;
        }
        makeChildren(instances);
    }

    private void makeChildren(Instances instances) {
        HashMap<Double, Instances> childInstances = new HashMap<>();
        HashMap<Double, Integer> counter = new HashMap<>();
        Enumeration<Instance> instanceEnumeration = instances.enumerateInstances();
        Integer maxCount = 0;
        Double favValue = null;
        while (instanceEnumeration.hasMoreElements()) {
            Instance instance = instanceEnumeration.nextElement();
            if (enableThreshold(this.attributeToClassify)) {
                if (instance.value(this.attributeToClassify) <= this.threshold) {
                    childInstances.putIfAbsent(0.0, new Instances(instances, 0));
                    childInstances.get(0.0).add(instance);
                } else {
                    childInstances.putIfAbsent(1.0, new Instances(instances, 0));
                    childInstances.get(1.0).add(instance);
                }
                continue;
            }
            childInstances.putIfAbsent(instance.value(this.attributeToClassify), new Instances(instances, 0));
            childInstances.get(instance.value(this.attributeToClassify)).add(instance);
            counter.put(instance.value(this.attributeToClassify), counter.getOrDefault(instance.value(this.attributeToClassify), 0) + 1);
            if (counter.get(instance.value(this.attributeToClassify)) > maxCount) {
                maxCount = counter.get(instance.value(this.attributeToClassify));
                favValue = instance.value(this.attributeToClassify);
            }
        }
        Double finalFavValue = favValue;
        childInstances.forEach((val, ci) -> {
            DTLNode node = new DTLNode();
            ci.deleteAttributeAt(this.attributeToClassify.index());
            node.buildTree(ci);
            node.parent = this;
            if (Objects.equals(val, finalFavValue)) {
                this.popularChild = node;
            }
            this.children.put(val, node);
        });
    }

    // Calculate the maximum information gain
    private double calcInformationGainMax(Instances instances) {
        double informationGainMax = 0;
        Enumeration<Attribute> attributeEnumeration = instances.enumerateAttributes();
        while (attributeEnumeration.hasMoreElements()){
            Attribute attribute = attributeEnumeration.nextElement();
            if (attribute == instances.classAttribute())
                continue;
            double informationGain;
            Double thresholdMax = null;
            if (enableThreshold(attribute)) {
                double[] attributeValues = instances.attributeToDoubleArray(attribute.index());

                Arrays.sort(attributeValues);
                double[] thresholdCandidate = new double[attributeValues.length - 1];
                for (int i = 0; i < attributeValues.length - 1; i++) {
                    thresholdCandidate[i] = (attributeValues[i + 1] + attributeValues[i]) / 2;
                }
                if (thresholdCandidate.length > 10) {
                    // Fisher–Yates shuffle
                    for (int i = 0; i < thresholdCandidate.length - 1; i++) {
                        int j = ThreadLocalRandom.current().nextInt(i, thresholdCandidate.length);
                        thresholdCandidate[i] = returnFirst(thresholdCandidate[j], thresholdCandidate[j] = thresholdCandidate[i]);
                    }
                    thresholdCandidate = Arrays.copyOf(thresholdCandidate, 10);
                }
                thresholdMax = thresholdCandidate[0];
                double maxGain = 0;
                for (double candidate : thresholdCandidate) {
                    double gain = calcThresholdGain(candidate, attribute, instances);
                    if (gain > maxGain) {
                        maxGain = gain;
                        thresholdMax = candidate;
                    }
                }
                informationGain = calcInformationGain(attribute, instances, thresholdMax);
            } else {
                informationGain = calcInformationGain(attribute, instances);
            }
            if (informationGain > informationGainMax) {
                informationGainMax = informationGain;
                this.attributeToClassify = attribute;
                this.threshold = thresholdMax;
            }
        }
        return informationGainMax;
    }

    // Make leaf with classified class as the most frequent class in instances
    private void makeLeaf(Instances instances) {
        double[] instancesClassValues = instances.attributeToDoubleArray(instances.classIndex());
        HashMap<Double, Integer> counter = new HashMap<>();
        Integer maxCount = 0;
        Double maxCountValue = null;
        for (double val : instancesClassValues) {
            Integer count = counter.getOrDefault(val, 0) + 1;
            counter.put(val, count);
            if (maxCount < count) {
                maxCount = count;
                maxCountValue = val;
            }
        }
        this.isLeaf = true;
        this.classifiedClass = maxCountValue;
    }

    private boolean enableThreshold(Attribute attribute) {
        return attribute.isNumeric();
    }

    private double calcThresholdGain(double candidate, Attribute attribute, Instances instances) {
        // TODO(ParadiseCatz): Is this correct?
        return calcInformationGain(attribute, instances, candidate);
    }

    private double calcInformationGain(Attribute attribute, Instances instances) {
        try {
            return myID3.calculateGain(myID3.calculateEntropy(instances), instances, attribute.index(), "");
        } catch (Exception e) {
            throw new Error(e);
        }
    }

    private double calcInformationGain(Attribute attribute, Instances instances, double threshold) {
        double informationGain;
        double[] attributeValues = instances.attributeToDoubleArray(attribute.index());
        for (int i = 0; i < instances.numInstances(); i++) {
            if (instances.instance(i).value(attribute) <= threshold) {
                instances.instance(i).setValue(attribute, 0.0);
            } else {
                instances.instance(i).setValue(attribute, 1.0);
            }
        }
        try {
            informationGain = myID3.calculateGain(myID3.calculateEntropy(instances), instances, attribute.index(), "");
        } catch (Exception e) {
            throw new Error(e);
        }
        for (int i = 0; i < instances.numInstances(); i++) {
            instances.instance(i).setValue(attribute, attributeValues[i]);
        }
        return informationGain;
    }

    Double classify(Instance instance) {
        if (this.isLeaf) {
            return this.classifiedClass;
        }
        if (enableThreshold(this.attributeToClassify)) {
            if (instance.value(this.attributeToClassify) <= this.threshold) {
                return this.children.get(0.0).classify(instance);
            } else {
                return this.children.get(1.0).classify(instance);
            }
        }
        if (instance.isMissing(this.attributeToClassify)) {
            return this.popularChild.classify(instance);
        }
        return this.children.get(instance.value(this.attributeToClassify)).classify(instance);
    }
}