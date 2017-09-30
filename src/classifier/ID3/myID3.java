package classifier.ID3;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nathanjamesruntuwene on 9/20/17.
 */
public class myID3 extends AbstractClassifier {
    @Override
    public void buildClassifier(Instances instances) throws Exception {

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
