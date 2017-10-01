import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Random;

import classifier.C45.myC45;
import classifier.ID3.myID3;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Created by nathanjamesruntuwene on 9/30/17.
 */
public class Main {

    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        String loadLocation = Utils.getOption("load_location", args);
        String saveLocation = Utils.getOption("save_location", args);
        String trainingLocation = Utils.getOption("training_location", args);
        String testLocation = Utils.getOption("test_location", args);

        String removedAttributes = Utils.getOption("remove", args);
        boolean useResample = Utils.getFlag("resample", args);
        boolean useCrossValidation = Utils.getFlag("use_cv", args);
        String splitPercentage = Utils.getOption("split",args);

        AbstractClassifier classifier;

        if (loadLocation.length() > 0) {     // LOAD MODEL
            classifier = (AbstractClassifier) SerializationHelper.read(loadLocation);
            System.out.println("Classifier loaded");

            if (testLocation.length() > 0) {
                DataSource testSource = new DataSource(testLocation);
                Instances testData = testSource.getDataSet();

                int countCorrect = 0;

                for (int i = 0; i < testData.size(); i++) {
                    Instance instance = testData.get(i);
                    if (classifier.classifyInstance(instance) == instance.classValue()) {
                        countCorrect++;
                    }
                }

                double accuracy = ((double) (countCorrect * 100) / testData.size());

                System.out.println("Test Data Accuracy: " + accuracy + " %");
            }

        } else {                            // BUILD MODEL
            String classifierName = Utils.getOption("classifier", args);

            DataSource trainingSource = new DataSource(trainingLocation);
            Instances trainingDataFull = trainingSource.getDataSet();

            // Remove attributes
            Remove remove = new Remove();
            remove.setAttributeIndices(removedAttributes);
            remove.setInputFormat(trainingDataFull);
            Instances removedTrainingData = Filter.useFilter(trainingDataFull, remove);

            // Apply resample
            Instances resampledTrainingData;
            if (useResample) {
                Resample resample = new Resample();
                System.out.println("Input resample size: ");
                double resampleSize = Double.parseDouble(br.readLine());
                System.out.println("Input bias value: ");
                double biasValue = Double.parseDouble(br.readLine());

                resample.setSampleSizePercent(resampleSize);
                resample.setBiasToUniformClass(biasValue);

                resample.setInputFormat(removedTrainingData);
                resampledTrainingData = Filter.useFilter(removedTrainingData, resample);
            } else {
                resampledTrainingData = removedTrainingData;
            }

            // Assign class index
            if (resampledTrainingData.classIndex() == -1)
                resampledTrainingData.setClassIndex(resampledTrainingData.numAttributes() - 1);

            // Assign classifiers
            Instances trainingData;
            if (classifierName.equals("myID3")) {
                classifier = new myID3();

                // Discretize attributes
                Discretize discretize = new Discretize();
                discretize.setInputFormat(resampledTrainingData);
                trainingData = Filter.useFilter(resampledTrainingData, discretize);

                System.out.println(trainingData.toString());
            } else if (classifierName.equals("myC45")) {
                classifier = new myC45();
                trainingData = resampledTrainingData;
            } else if (classifierName.equals("ID3")) {
                classifier = new myID3();

                // Discretize attributes
                Discretize discretize = new Discretize();
                discretize.setInputFormat(resampledTrainingData);
                trainingData = Filter.useFilter(resampledTrainingData, discretize);

                System.out.println(trainingData.toString());
            } else if (classifierName.equals("J48")) {
                classifier = new J48();
                trainingData = resampledTrainingData;
            } else {
                System.out.println("Classifier needs to be either 'myID3', 'myC45', 'ID3', or 'J48");
                return;
            }

            if (useCrossValidation) {                   // Use Cross Validation
                System.out.println("Using 10-fold cross validation");

                Evaluation eval = new Evaluation(trainingData);
                eval.crossValidateModel(classifier,trainingData,10,new Random());
                System.out.println(eval.toSummaryString());
            } else if (splitPercentage.length() > 0) {  // Use Split percentage
                System.out.println("Using split percentage");

                trainingData.randomize(new Random());

                int threshold = (int)Math.round((double)trainingData.numInstances() * Double.parseDouble(splitPercentage) / 100.0D);
                int numTestingInstances = trainingData.numInstances() - threshold;
                Instances training = new Instances(trainingData, 0, threshold);
                Instances testing = new Instances(trainingData, threshold, numTestingInstances);

                classifier.buildClassifier(training);

                DataSource testSource = new DataSource(testLocation);
                Instances testData = testSource.getDataSet();

                int countCorrect = 0;

                for (int i = 0; i < testData.size(); i++) {
                    Instance instance = testData.get(i);
                    if (classifier.classifyInstance(instance) == instance.classValue()) {
                        countCorrect++;
                    }
                }

                double accuracy = ((double) (countCorrect * 100) / testData.size());

                System.out.println("Test Data Accuracy: " + accuracy + " %");

            } else {     // Use training-test
                System.out.println("Using training-test");
                classifier.buildClassifier(trainingData);

                int countCorrect = 0;

                for (int i = 0; i < trainingData.size(); i++) {
                    Instance instance = trainingData.get(i);
                    if (classifier.classifyInstance(instance) == instance.classValue()) {
                        countCorrect++;
                    }
                }

                double accuracy = ((double) (countCorrect * 100) / trainingData.size());

                System.out.println("Test Data Accuracy: " + accuracy + " %");
            }
        }

        // SAVE MODEL
        if (saveLocation.length() > 0)
            SerializationHelper.write(saveLocation, classifier);
    }
}
