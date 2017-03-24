/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaapplications;

import java.io.BufferedReader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.trees.Id3;
import weka.core.Attribute;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.lazy.IBk;
import weka.filters.unsupervised.attribute.StringToWordVector;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;


/**
 *
 * @author Siamul Karim Khan
 */
public class WekaApplications {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        System.out.println("=============================================================================================================");
        System.out.println("Calling for Iris Dataset");
        System.out.println("=============================================================================================================");
        BufferedReader reader = new BufferedReader(new FileReader("wekaLibrary/weka-3-9-0/data/iris.arff"));
        Instances data = new Instances(reader);
        NumericToNominal nm = new NumericToNominal();
        nm.setInputFormat(data);
        data = Filter.useFilter(data, nm);
        data.setClassIndex(data.numAttributes() - 1);
        Id3 tree = new Id3();
        String[] ops = {"-U"};
        tree.setOptions(ops);
        Random rnd = new Random(System.currentTimeMillis());
        data.randomize(rnd);
        Evaluation eval = new Evaluation(data);
        Instances train = data.trainCV(5, 0);
        Instances test = data.testCV(5, 0);
        tree.buildClassifier(train);
        eval.evaluateModel(tree, test);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        //System.out.println(data);
//        System.out.println("=============================================================================================================");
//        System.out.println("Statistics for ID3 Decision Tree: ");
//        System.out.println("=============================================================================================================");
//        decisionTreeCode();
//        Scanner r = new Scanner(new File("wordListforRemoval.txt"));
//        ArrayList<String> wordList = new ArrayList<>();
//        while(r.hasNextLine())
//        {
//            String s = r.nextLine();
//            if(s.length() > 2) wordList.add(s);
//        }
//        String[] files = {"3d_printer", "Coffee", "Windows_Phone"};
//        Instances dataSet = loadXMLFile("Data/Training/", files, wordList, false);
//        Instances dataSetFiltered = getBagOfWords(dataSet);
//        int[] nnArr = {5,10,20,30};
//        for(int i = 0; i<4; i++)
//        {
//             System.out.println("=============================================================================================================");
//             System.out.println("Statistics for k-Nearest Neighbor with k = " + nnArr[i]);
//             System.out.println("=============================================================================================================");
//             KNNCode(dataSetFiltered, files, nnArr[i]);
//        }
//        System.out.println("=============================================================================================================");        
//        System.out.println("Statistics for Naive Bayes: ");
//        System.out.println("=============================================================================================================");
//        NaiveBayesCode(dataSet, files);
//        System.out.println("=============================================================================================================");
       
    }  
    static void NaiveBayesCode(Instances dataSet, String[] files) throws Exception
    {
        Random rnd = new Random(System.currentTimeMillis());
        dataSet.randomize(rnd);
        Instances trainSet = dataSet.trainCV(5, 0);
        Instances testSet = dataSet.testCV(5, 0);
        NaiveBayesMultinomialText model = new NaiveBayesMultinomialText();
        model.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(dataSet);
        eval.evaluateModel(model, testSet);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
    }
    static void KNNCode(Instances dataSet, String[] files, int noOfNeighbors) throws Exception
    {
        Random rnd = new Random(System.currentTimeMillis());
        dataSet.randomize(rnd);
        Instances trainSet = dataSet.trainCV(5, 0);
        Instances testSet = dataSet.testCV(5, 0);
        IBk model = new IBk(noOfNeighbors);
        model.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(dataSet);
        eval.evaluateModel(model, testSet);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
    }
    static Instances loadXMLFile(String directory, String[] files, ArrayList<String> remWordList, boolean isNormalize) throws Exception
    {
        String[][] allBodies = new String[files.length][];
        int totalData = 0;
        for(int i = 0; i<files.length; i++)
        {
            String fname = directory + files[i] + ".xml";
            File inputFile = new File(fname);
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(inputFile);
            //doc.getDocumentElement().normalize();
            NodeList nList = doc.getElementsByTagName("row");
            allBodies[i] = new String[nList.getLength()];
            for (int j = 0; j< nList.getLength(); j++)
            {
                Node nNode = nList.item(j);
                Element eElem = (Element) nNode;
                allBodies[i][j] = eElem.getAttribute("Body").replaceAll("<[^>]+>", "").replaceAll("(\\r|\\n|\\r\\n)+", "").replaceAll("\\p{Punct}+", " ").replaceAll("\\b[\\w']{1,2}\\b", "").toLowerCase();
                for(int k = 0; k < remWordList.size(); k++)
                {
                    allBodies[i][j] = allBodies[i][j].replaceAll(" " + remWordList.get(k) + " ", " ").replaceAll("\\s{2,}", " ");
                }
                totalData++;
            }
        }
        Attribute body = new Attribute("Body", (FastVector) null);
        FastVector classVal = new FastVector(files.length);
        for(int i = 0; i<files.length; i++)
        {
            classVal.addElement(files[i]);
        }
        Attribute classAttr = new Attribute("DocClass", classVal);
        FastVector feature = new FastVector(2);
        feature.addElement(body);
        feature.addElement(classAttr);
        Instances dataSet = new Instances("Documents", feature, totalData);
        dataSet.setClassIndex(1);
        for(int i = 0; i<allBodies.length; i++)
        {
            for (String allBody : allBodies[i]) {
                if (allBody.length() > 2) {
                    Instance inst = new DenseInstance(2);
                    inst.setValue((Attribute) feature.elementAt(0), allBody);
                    inst.setValue((Attribute) feature.elementAt(1), files[i]);
                    dataSet.add(inst);
                }
            }
        }
        return dataSet;
    }
    static Instances getBagOfWords(Instances dataSet) throws Exception
    {
        StringToWordVector m_Filter = new StringToWordVector();
        String[] options;
        options = new String[4];
        options[0] = "-C";
        options[1] = "-I";
        options[2] = "-N";
        options[3] = "1";
        m_Filter.setOptions(options);
        m_Filter.setInputFormat(dataSet);
        return Filter.useFilter(dataSet, m_Filter);
    }
    static void decisionTreeCode() throws Exception
    {
        Instances data = loadCSVFile("assignment1_data_set.csv");
        Id3 tree = new Id3();
        String[] ops = {"-U"};
        tree.setOptions(ops);
        Random rnd = new Random(System.currentTimeMillis());
        data.randomize(rnd);
        Evaluation eval = new Evaluation(data);
        Instances train = data.trainCV(5, 0);
        Instances test = data.testCV(5, 0);
        tree.buildClassifier(train);
        eval.evaluateModel(tree, test);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
    }
    static Instances loadCSVFile(String filename) throws Exception
    {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filename));
        Instances numData = loader.getDataSet();
        NumericToNominal nm = new NumericToNominal();
        nm.setInputFormat(numData);
        Instances data = Filter.useFilter(numData, nm);
        data.setClassIndex(data.numAttributes()-1);
        return data;
    }
}
