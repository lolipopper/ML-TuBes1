import classifier.C45.myC45;
import classifier.ID3.myID3;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.supervised.attribute.Discretize;

/**
 * Created by nathanjamesruntuwene on 9/30/17.
 */
public class MainWithEvaluation {

    public static void main(String[] args) throws Exception {
        try {
            if(args.length == 0) {
                throw new Exception("The first argument must be the class name of a classifier");
            }

            String classifierName = args[0];
            args[0] = "";

            String trainingLocation = Utils.getOption("t", args);
            String testLocation = Utils.getOption("T", args);

            Instances trainingData;
            Instances testData;

            if (trainingLocation.length() > 0) {
                ConverterUtils.DataSource trainingSource = new ConverterUtils.DataSource(trainingLocation);
                trainingData = trainingSource.getDataSet();
            } else {
                throw new Exception("please provide training data location with -t");
            }

            if (testLocation.length() > 0) {
                ConverterUtils.DataSource testSource = new ConverterUtils.DataSource(testLocation);
                testData = testSource.getDataSet();
            } else {
                throw new Exception("please provide test data location with -T");
            }

            FilteredClassifier fc = new FilteredClassifier();
            Classifier classifier;

            if (classifierName.equals("ID3")) {
                classifier = new myID3();
                Discretize discretizeFilter = new Discretize();
                discretizeFilter.setInputFormat(trainingData);
            } else if (classifierName.equals("C45")) {
                classifier = new myC45();
            } else {
                throw new Exception("Classifier name must be 'ID3' or 'C45'");
            }

            fc.setClassifier(classifier);

            Evaluation evaluation = new Evaluation(trainingData);
            System.out.println(Evaluation.evaluateModel(fc, args));
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println(e.getMessage());
        }
    }
}
