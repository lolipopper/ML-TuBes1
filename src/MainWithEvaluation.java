import classifier.C45.myC45;
import classifier.ID3.myID3;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;

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

            Classifier classifier;

            if (classifierName.equals("ID3")) {
                classifier = new myID3();
            } else if (classifierName.equals("C45")) {
                classifier = new myC45();
            } else {
                throw new Exception("Classifier name must be 'ID3' or 'C45'");
            }

            System.out.println(Evaluation.evaluateModel(classifier, args));
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println(e.getMessage());
        }
    }
}
