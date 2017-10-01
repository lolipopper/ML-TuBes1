import java.io.BufferedReader;
import java.io.InputStreamReader;

import classifier.C45.myC45;
import classifier.ID3.myID3;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
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

        String removedAttributes = Utils.getOption("R", args);
        boolean useResample = Utils.getFlag("resample", args);

        AbstractClassifier classifier;

        if (loadLocation.length() > 0) {     // LOAD MODEL
            classifier = (AbstractClassifier) SerializationHelper.read(loadLocation);

        } else {                            // BUILD MODEL
            String classifierName = Utils.getOption("classifier_type", args);

            DataSource trainingSource = new DataSource(trainingLocation);
            Instances trainingDataFull = trainingSource.getDataSet();

            Remove remove = new Remove();
            remove.setAttributeIndices(removedAttributes);
            remove.setInputFormat(trainingDataFull);
            Instances trainingData = Filter.useFilter(trainingDataFull, remove);

            if (trainingData.classIndex() == -1)
                trainingData.setClassIndex(trainingData.numAttributes() - 1);

            if (classifierName.equals("ID3")) {
                classifier = new myID3();
            } else if (classifierName.equals("C45")) {
                classifier = new myC45();
            } else {
                System.out.println("Classifier needs to be either \'ID3\' or \'C45\'");
                return;
            }

            classifier.buildClassifier(trainingData);
        }

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

        if (saveLocation.length() > 0)
            SerializationHelper.write(saveLocation, classifier);
    }
}
