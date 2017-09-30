package classifier.C45;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.HashMap;

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

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        root = new DTLNode();
        Instances instancesCopy = new Instances(instances);
        root.buildTree(instancesCopy);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
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

    DTLNode() {
        children = new HashMap<>();
    }

    public void buildTree(Instances instances) {
        if (instances.isEmpty()) {
            throw new Error("EMPTY INSTANCES");
        }
        if (instances.numDistinctValues(instances.classAttribute()) == 1) {
            this.isLeaf = true;
            classifiedClass = instances.firstInstance().classValue();
            return;
        }
        // TODO(ParadiseCatz): check attribute number / base case
        double informationGainMax = 0;
        Enumeration<Attribute> attributeEnumeration = instances.enumerateAttributes();
        while (attributeEnumeration.hasMoreElements()){
            Attribute attribute = attributeEnumeration.nextElement();
            if (attribute == instances.classAttribute())
                continue;
            double informationGain = calcInformationGain(attribute, instances);
            if (informationGain > informationGainMax) {
                informationGainMax = informationGain;
                this.attributeToClassify = attribute;
            }
        }
        if (informationGainMax == 0) {
            double[] instancesClassValue = instances.attributeToDoubleArray(instances.classIndex());
            HashMap<Double, Integer> counter = new HashMap<>();
            Integer maxCount = 0;
            Double maxCountValue = null;
            for (double val : instancesClassValue) {
                Integer count = counter.getOrDefault(val, 0) + 1;
                counter.put(val, count);
                if (maxCount < count) {
                    maxCount = count;
                    maxCountValue = val;
                }
            }
            this.isLeaf = true;
            this.classifiedClass = maxCountValue;
            return;
        }
        HashMap<Double, Instances> childInstances = new HashMap<>();
        Enumeration<Instance> instanceEnumeration = instances.enumerateInstances();
        while (instanceEnumeration.hasMoreElements()) {
            Instance instance = instanceEnumeration.nextElement();
            childInstances.put(instance.value(this.attributeToClassify), Instances.mergeInstances(childInstances.get(instance.value(this.attributeToClassify)), (Instances) instance));
        }
        childInstances.forEach((val, ci) -> {
            DTLNode node = new DTLNode();
            node.buildTree(ci);
            this.children.put(val, node);
        });
    }

    private double calcInformationGain(Attribute attribute, Instances instances) {
        // TODO(ParadiseCatz): implement this
        return 0;
    }
}