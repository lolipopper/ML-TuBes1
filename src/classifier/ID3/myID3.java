import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import weka.classifiers.AbstractClassifier;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.SubsetByExpression;

/**
 * Created by nathanjamesruntuwene on 9/20/17.
 */
public class myID3 extends AbstractClassifier {    
    private Instances data;
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        data = new Instances(instances);
        ArrayList<Integer> processedIndex = new ArrayList<>();
        String addCondition = "";
        recursiveIterate(instances, addCondition, processedIndex);
//        int attributeIndex = decideAttributeFactor(instances, addCondition, processedIndex);
//        processedIndex.add(attributeIndex);
        
    }
    
    public void recursiveIterate(Instances instances, String decisionCondition, ArrayList<Integer> processedIndexes) throws Exception
    {
        if(processedIndexes.size()<4){
            double entropyS = calculateEntropy(instances);
            if (entropyS>0){
                Instances newInstances;
//                System.out.println(decisionCondition);
                int attributeIndex = decideAttributeFactor(entropyS, instances, decisionCondition, processedIndexes);
                ArrayList<Integer> copyList = new ArrayList<>(processedIndexes);
                copyList.add(attributeIndex);
                for (int i=0; i<instances.attribute(attributeIndex).numValues(); i++){
//                    System.out.println(instances.attribute(attributeIndex).value(i));
                    String condition = addStringCondition(decisionCondition,attributeIndex,instances.attribute(attributeIndex).value(i));
//                    System.out.println(condition);
                    newInstances = filterInstances(instances,condition);        
//                    printInstances(newInstances);

                    recursiveIterate(newInstances, insertAnd(condition), copyList);
                }
            }else{
                System.out.println(decisionCondition);
                System.out.println("This is leaf for = " + instances.instance(0).value(instances.classIndex()));
            }
        }else{
            System.out.println("Ada yang sampai sini ga?");
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }
    
    
    
    public int decideAttributeFactor(double entropyS, Instances instances, String addCondition, ArrayList<Integer> processedIndexes) throws Exception{
        double maxIG = calculateGain(entropyS, instances, 0, addCondition);
        int id = 0;
        for(int i=1; i<instances.numAttributes()-1; i++){
            if (!processedIndexes.contains(i)){
                double gain;
                if ((gain = calculateGain(entropyS, instances, i, addCondition)) > maxIG){
                    maxIG = gain;
                    id = i;
                }                
            }
        }
        System.out.println(id);
        System.out.println("");
        return id;
    }

    public double calculateEntropy(Instances instances){
        AttributeStats stats = instances.attributeStats(instances.classIndex());
        int[] countResults = (stats.nominalCounts);
        int totalCount = stats.totalCount;
        double ret= 0.0;
        for(int i=0; i < countResults.length; i++){
            if(countResults[i]>0){
                double distribution = ((double)countResults[i]/totalCount);
                ret -= distribution*(Math.log(distribution)/Math.log(2));
            }
        }
        return ret;
    }

    public double calculateGain(double entropy, Instances instances, int attributeIndex, String addCondition) throws Exception{
        double ret = entropy;
        System.out.println(instances.attribute(attributeIndex));
        for(int i=0; i<instances.attribute(attributeIndex).numValues(); i++){
//            System.out.println(instances.attribute(attributeIndex).value(i));
            String condition = addCondition + "(ATT"+(attributeIndex+1)+" is '" + instances.attribute(attributeIndex).value(i) +"')";
            Instances filteredInstances = filterInstances(instances,condition);
//            System.out.println(filteredInstances.numInstances());
            ret -= calculateEntropy(filteredInstances)*((double)filteredInstances.numInstances()/instances.numInstances());            
        }
        System.out.println("Information Gain = "+ret);
//        System.out.println("");
        return ret;
    }
    
    public void printInstances(Instances instances){
        for(int j=0; j < instances.numInstances(); j++){
            Instance curInstance = instances.instance(j);
            System.out.println(curInstance);
        }        
        System.out.println("");
    }
    
    public Instances filterInstances(Instances instances, String condition) throws Exception{
        SubsetByExpression filter = new SubsetByExpression();
        String[] options = new String[2];
        options[0] = "-E";
        options[1] = condition;
        filter.setOptions(options);
        filter.setInputFormat(instances);
        Instances filteredInstances = Filter.useFilter(instances, filter);
        return filteredInstances;
    }
    
    public String addStringCondition(String addOption, int attributeIndex, String value){
        return addOption+"(ATT"+(attributeIndex+1)+" is '"+value+"')";
    }
    
    public String insertAnd(String addOption){
        return addOption+" and ";
    }
    
    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String filename = "D:\\Programming\\_Project\\WekaTest\\test\\weather.nominal.arff";
        DataSource source = new DataSource(filename);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes()-1);
        }
        myID3 classifier = new myID3();
        classifier.buildClassifier(data);
    }

}
