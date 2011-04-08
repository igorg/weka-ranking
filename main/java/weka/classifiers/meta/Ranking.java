/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Ranking.java
 *    Copyright (C) 2002 University of Waikato
 *    Authors: Bianca Zadrozny, ..., ...
 *    Implementation: Bianca Zadrozny, Igor Giusti.
 */

package weka.classifiers.meta;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import static weka.core.Utils.*;

/**
 *
 * @author Igor
 */
public class Ranking extends RandomizableIteratedSingleClassifierEnhancer {

	private static final long serialVersionUID = 1L;

    protected String m_Mode;
    protected int m_PairsPerInstance;
    protected Instances headerMatcher;
    
    @Override
    public String[] getOptions() {
        String[] superOptions = super.getOptions();
        String[] options = new String[superOptions.length + 4];
        
        int current = 0;
        options[current++] = "-m";
        options[current++] = m_Mode;
        options[current++] = "-p";
        options[current++] = "" + m_PairsPerInstance;
        System.arraycopy(superOptions, 0, options, current,
                         superOptions.length);
        
        return options;
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        String mode = getOption('m', options);
        if (mode.length() == 0 || (!mode.equals("quicksort") && !mode.equals("tournament")))
            mode = "tournament";
        setMode(mode);

        String pairsPerInstanceString = getOption('p', options);
        if (pairsPerInstanceString.length() != 0) {
    		int pairsPerInstance = Integer.parseInt(pairsPerInstanceString);
            if (pairsPerInstance <= 0)
                setPairsPerInstance(Integer.MAX_VALUE);
            else
            	setPairsPerInstance(pairsPerInstance);
        } else
        	setPairsPerInstance(Integer.MAX_VALUE);
        
        super.setOptions(options);
    }
    
    @SuppressWarnings("unchecked")
	@Override
    public Enumeration listOptions() {
        Vector allOptions = new Vector(4);
        allOptions.add(new Option(
                "\tThe number of pairs per instance to be used in\n"
               +"\tclassifier training. If not informed, the whole\n" +
               	"\ttraining set will be used.", "p", 1,
                "-p <num>"));
        allOptions.add(new Option(
                "\tThe mode which will be used to rank the instances.\n"
               +"\tCan be \"quicksort\" or \"tournament\".\n"
               +"\t(default: tournament)", "m", 1,
                "-m <tournament, quicksort>"));

        Enumeration enu = super.listOptions();
        while (enu.hasMoreElements()) {
            allOptions.add(enu.nextElement());
        }

        return allOptions.elements();
    }


    public void setPairsPerInstance(int s) {
        m_PairsPerInstance = s;
    }

    public void setMode(String m) {
        m_Mode = m;
    }


    @Override
    public void buildClassifier(Instances data) throws Exception{

        if (!data.classAttribute().isNominal()) {
            throw new Exception("Class attribute must be nominal!");
        }

        if (data.numClasses() != 2) {
            throw new Exception("Number of classes must be 2!");
        }

        Instances instancesByClass[] = instancesByClass(data);
        Random random = new Random(m_Seed);

        System.out.println("Number of classifiers: " + m_NumIterations);

        m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);

        for (int i = 0; i < m_Classifiers.length; i++) {
            System.out.println("\nBuilding classifier " + i);
            
            Instances train = null;
            if (m_PairsPerInstance < data.numInstances() - 1) {
                train = makePairs(instancesByClass);
            } else {
            	train = crossProduct(instancesByClass);
            }
            prepareToBuild(train, data);

            if (m_Classifier instanceof Randomizable) {
                ((Randomizable) m_Classifiers[i]).setSeed(random.nextInt());
            }

            m_Classifiers[i].buildClassifier(train);
        }

        headerMatcher = new Instances(data, 0);
    }


    @Override
    public double[] distributionForInstance(Instance instance) 
        throws Exception{

        double [] sums = new double [instance.numClasses()], newProbs;

        for (int i = 0; i < m_NumIterations; i++) {
            newProbs = m_Classifiers[i].distributionForInstance(instance);
            for (int j = 0; j < newProbs.length; j++)
                sums[j] += newProbs[j];
        }

        if (eq(sum(sums), 0)) {
            return sums;
        } else {
            normalize(sums);
            return sums;
        }
    }


    @SuppressWarnings("unchecked")
	protected static String makeOptionString(Classifier classifier) {

        StringBuffer optionsText = new StringBuffer("");
        // General options
        optionsText.append("\n\nGeneral options:\n\n");
        optionsText.append("-t\n");
        optionsText.append("\tSets the path of training file.\n");
        optionsText.append("-T\n");
        optionsText.append("\tSets the path of test file. If not informed,\n" +
        				   "\ta cross-validation will be performed.\n");
        optionsText.append("-c <num, first, last>\n");
        optionsText.append("\tSets the index of class attribute\n\t(default: last).\n");
        optionsText.append("-r\n");
        optionsText.append("\tSets the path of file to write the rank and\n" +
        				   "\tits statistics.\n");
        optionsText.append("-x <num>\n");
        optionsText.append("\tNumber of cross-validation folds.\n\t(default: 10)\n");
        optionsText.append("-s <num>\n");
        optionsText.append("\tRandom number for training and testing purposes.\n\t(default: 1)\n");

        // Get scheme-specific options
        if (classifier instanceof OptionHandler) {
            optionsText.append("\nOptions specific to " + classifier.getClass().getName() + ":\n\n");
            Enumeration enu = ((OptionHandler)classifier).listOptions();

            while (enu.hasMoreElements()) {
                Option option = (Option) enu.nextElement();
                optionsText.append(option.synopsis() + '\n');
                optionsText.append(option.description() + "\n");
            }
        }

        return optionsText.toString();
    }


	protected Instances[] instancesByClass(Instances instances) throws Exception {
    	Instances[] byClass = new Instances[2];
    	byClass[0] = new Instances(instances, 0);
    	byClass[1] = new Instances(instances, 0);
    	
    	for (int i = 0; i < instances.numInstances(); i++) {
    		Instance instance = (Instance) instances.instance(i).copy();
    		if (instance.classValue() == 0.0) {
    			byClass[0].add(instance);
    		} else if (instance.classValue() == 1.0) {
    			byClass[1].add(instance);
    		} else {
    			throw new Exception("Unsupported class.");
    		}
    	}
    	
    	return byClass;
    }


    protected double auc(Instances rank) {
    	double summation = 0.0, zeros = 0.0, ones = 0.0;
    	
    	for (int i=0; i < rank.numInstances(); i++) {
    		Instance inst = rank.instance(i);
    		
    		if (inst.classValue() == 0.0) {
    			summation += rank.numInstances() - i;
    			zeros++;
    		} else if (inst.classValue() == 1.0) {
    			ones++;
    		}
    	}
    	
    	return (summation - zeros*(zeros + 1)/2)/(zeros * ones);
    }    

    protected Instances makePairs(Instances[] instances) {
        FastVector newAttributes = new FastVector();
        for (int i = 0; i < instances[0].numAttributes(); i++) {
            newAttributes.addElement((Attribute) instances[0].attribute(i).copy());
        }
        for (int i = 0; i < instances[1].numAttributes(); i++) {
            newAttributes.addElement((Attribute) instances[1].attribute(i).copy());
        }

        Instances result = new Instances("to_rank", newAttributes, 0);
    	
        Random random = new Random(m_Seed);

        int zeroIterations = instances[0].numInstances();
        if (instances[0].numInstances() > m_PairsPerInstance)
        	zeroIterations = m_PairsPerInstance;
        
        int oneIterations = instances[1].numInstances();
        if (instances[1].numInstances() > m_PairsPerInstance)
        	oneIterations = m_PairsPerInstance;
        
    	for (int i = 0; i < instances[0].numInstances(); i++) {
    		Instance zeroInstance = instances[0].instance(i);
    		
    		if (oneIterations == m_PairsPerInstance)
    			instances[1].randomize(random);
            
    		for (int j = 0; j < oneIterations; j++) {
    			Instance oneInstance = instances[1].instance(j);
        		result.add(zeroInstance.mergeInstance(oneInstance));
        	}
    	}
        
    	for (int i = 0; i < instances[1].numInstances(); i++) {
    		Instance oneInstance = instances[1].instance(i);
    		
    		if (zeroIterations == m_PairsPerInstance)
    			instances[0].randomize(random);
    		
    		for (int j = 0; j < zeroIterations; j++) {
    			Instance zeroInstance = instances[0].instance(j);
        		result.add(oneInstance.mergeInstance(zeroInstance));
        	}
    	}

        return result;
    }


    protected Instances crossProduct(Instances[] instances) {
        FastVector newAttributes = new FastVector();
        for (int i = 0; i < instances[0].numAttributes(); i++) {
            newAttributes.addElement((Attribute) instances[0].attribute(i).copy());
        }
        for (int i = 0; i < instances[1].numAttributes(); i++) {
            newAttributes.addElement((Attribute) instances[1].attribute(i).copy());
        }

        Instances crossProduct = new Instances("to_rank", newAttributes, 0);

        for (int i = 0; i < instances[0].numInstances(); i++) {
            Instance zeroInst = (Instance) instances[0].instance(i).copy();
            
            for (int k = 0; k < instances[1].numInstances(); k++) {
                Instance oneInst = (Instance) instances[1].instance(k).copy();

                //Joining "aInst" and "bInst"
                Instance joined = zeroInst.mergeInstance(oneInst);
                crossProduct.add(joined);

                //Joining "bInst" and "aInst"
                joined = oneInst.mergeInstance(zeroInst);
                crossProduct.add(joined);
            }
        }

        return crossProduct;
    }


    protected void prepareToBuild(Instances crossedDataset, Instances ancestor) {
        int firstOldClassIndex = ancestor.classIndex();
        int secondOldClassIndex = firstOldClassIndex + ancestor.numAttributes();

        // Inserindo um novo atributo que vai passar a ser a classe das instancias
        // em crossedProduct, esse atributo é nominal e pode ter os valores 0 ou 1.
        FastVector classAttValues = new FastVector(2);
        classAttValues.addElement("0");
        classAttValues.addElement("1");

        Attribute classAttribute = new Attribute("class", classAttValues);
        crossedDataset.insertAttributeAt(classAttribute,
                                         crossedDataset.numAttributes());
        crossedDataset.setClassIndex(crossedDataset.numAttributes() - 1);

        // Iterating over all instances defining their classes
        for (int i = 0; i < crossedDataset.numInstances(); i++) {
            Instance instance = crossedDataset.instance(i);
            double y1 = instance.value(firstOldClassIndex);
            if (Double.isNaN(y1)) y1 = Double.MAX_VALUE;
            double y2 = instance.value(secondOldClassIndex);
            if (Double.isNaN(y2)) y2 = Double.MAX_VALUE;
            instance.setClassValue(oneOperator(y1 < y2));
        }

        //Deleting original class attributes
        crossedDataset.deleteAttributeAt(secondOldClassIndex);
        crossedDataset.deleteAttributeAt(firstOldClassIndex);
    }

    
    protected int oneOperator(boolean b) {
        int result = 0;
        if (b)
            result = 1;
        return result;
    }


    private Instances createTemplateInstances(Instances insts) {
        FastVector newAttrs = new FastVector();

        for (int i = 0; i < insts.numAttributes(); i++) {
            newAttrs.addElement((Attribute) insts.attribute(i).copy());
        }

        //Deleting class attribute and replicating existing attributes
        newAttrs.removeElementAt(insts.classIndex());
        newAttrs.appendElements(newAttrs);

        //Adding new class attribute
        FastVector classValues = new FastVector(2);
        classValues.addElement("0");
        classValues.addElement("1");

        newAttrs.addElement(new Attribute("Class", classValues));

        Instances template = new Instances("template", newAttrs, 0); 
        template.setClassIndex(template.numAttributes() - 1);

        return template;
    }


    private Instance mergeOnTemplate(Instance first, Instance second, Instances template) {
        //Copy of instances, so the original won't be modified
        Instance firstCopy = (Instance) first.copy();
        Instance secondCopy = (Instance) second.copy();

        firstCopy.setDataset(null);
        secondCopy.setDataset(null);

        firstCopy.deleteAttributeAt(first.classIndex());
        secondCopy.deleteAttributeAt(second.classIndex());

        //Merging and adjusting instance to rank
        Instance merged = firstCopy.mergeInstance(secondCopy);

        merged.insertAttributeAt(template.classIndex());
        merged.setDataset(template);
        merged.setClassValue(oneOperator(first.classValue() < second.classValue()));

        return merged;
    }


    private Instance compareInstances(Instance i, Instance j, Instances template) throws Exception {
    	int count = 0;

        Instance ijInst = mergeOnTemplate(i, j, template);
        double[] ijDist = this.distributionForInstance(ijInst);
        int ijPred = (int) maxIndex(ijDist);
        
        if (template.classAttribute().value(ijPred).equals("1")) count++;
        else count--;

        Instance jiInst = mergeOnTemplate(j, i, template);
        double[] jiDist = this.distributionForInstance(jiInst);
        int jiPred = (int) maxIndex(jiDist);
        
        if (template.classAttribute().value(jiPred).equals("1")) count--;
        else count++;

        Instance winner = null;
        if (count > 0) winner = i;
        else if (count < 0) winner = j;

        return winner;
    }

    
    private void tournament(Instances data, Instances template) throws Exception {
    	int[] wins = new int[data.numInstances()];

        // Comparing instances
        for (int i = 0; i < data.numInstances(); i++) {
            Instance iInst = data.instance(i);

            for (int j = i + 1; j < data.numInstances(); j++) {
                Instance jInst = data.instance(j);

                Instance greater = compareInstances(iInst, jInst, template);
                if (iInst.equals(greater)) {
                	wins[i] += 2;
                } else if (jInst.equals(greater)) {
                	wins[j] += 2;
                } else {
                	wins[i]++;
                	wins[j]++;
                }
            }            
        }
        
        Instances aux = new Instances(data);
        data.delete();
        for (int i = 0; i < wins.length; i++) {
        	int position = maxIndex(wins);
        	wins[position] = Integer.MIN_VALUE;
        	Instance inst = aux.instance(position);
        	data.add(inst);
        }
        
    }


    private void quickSort(Instances instances, int start, int end, Instances template) throws Exception {
        if (start < end) {
            int pivot = partition(instances, start, end, template);
            quickSort(instances, start, pivot-1, template);
            quickSort(instances, pivot+1, end, template);
        }
    }


    protected int partition(Instances instances, int begin, int end, Instances template) throws Exception {
        Instance pivot = instances.instance(end);
        
        int i = begin;
        
        Instance instance = instances.instance(i);
        Instance greater = compareInstances(pivot, instance, template); 
        while ((greater == null || greater.equals(instance)) && i < end) {
        	instance = instances.instance(++i);
        	greater = compareInstances(pivot, instance, template);
        }
        
        int j = i + 1;
        while (j < end) {
        	instance = instances.instance(j);
        	greater = compareInstances(pivot, instance, template);
        	
        	if (greater == null || greater.equals(instance)){
        		instances.swap(i++, j);
        	}
        	
        	j++;
        }
        
        instances.swap(i, end);

        return i;
    }

    
    public void rank(Instances instances) throws Exception {
        if (headerMatcher == null) {
            throw new Exception("Ranker not build yet!");
        }

        if (!instances.equalHeaders(headerMatcher)) {
            throw new Exception("Ranking and training instances must have " +
                                "compatible attribute and class info.");
        }

        //New instances format to work with
        Instances rankInsts = createTemplateInstances(instances);

        //QuickSort or Tournament between instances
        System.out.println("\nRanking Instances...");
        if (m_Mode.equals("tournament")) {
            tournament(instances, rankInsts);
        } else if (m_Mode.equals("quicksort")){
            quickSort(instances, 0, instances.numInstances()-1, rankInsts);
        }

        System.out.println("\nDone!\n");
    }

    
    protected double[] nominalAttValues(Attribute att) {
        double[] values = new double[att.numValues()];

        for (int i = 0; i < att.numValues(); i++) {
            values[i] = (double) i;
        }

        return values;
    }
    
    public static void removeMissingClassInstances(Instances instances) {
    	int index = 0;
    	
    	while (index < instances.numInstances()){
    		if (instances.instance(index).classIsMissing())
    			instances.delete(index);
    		else
    			index++;
    	}
    }

    
    public static void main(String[] args) throws Exception{

        int classIndex = -1, cvFolds = 10, seed = 1;
        long trainTimeStart = 0, testTimeStart = 0;
        long[] trainTimes = new long[1],
        	   testTimes = new long[1];
        
        String dataFileName, testFileName, rankFileName, classIndexString,
        cvFoldsString, seedString = null;
        BufferedReader dataReader, testReader = null;
        BufferedWriter rankWriter = null;
        Instances data, test = null;
        Vector<Instances> results = new Vector<Instances>();

        Ranking ranker = new Ranking();

        //Get basic options
        try {
            classIndexString = getOption('c', args);
            dataFileName = getOption('t', args);
            testFileName = getOption('T', args);
            rankFileName = getOption('r', args);
            cvFoldsString = getOption('x', args);
            seedString = getOption('s', args);

            if (classIndexString.length() != 0) {
                if (classIndexString.equals("first"))
                    classIndex = 1;
                else if (classIndexString.equals("last"))
                    classIndex = -1;
                else
                    classIndex = Integer.parseInt(classIndexString);
            }
            
            if (seedString.length() != 0)
            	seed = Integer.parseInt(seedString);

            //Reading training instances and instances to rank.
            if (dataFileName.length() == 0) {
                throw new Exception("No training file given!");
            } else if (rankFileName.length() == 0) {
                throw new Exception("No rank file given!");
            } else {
                try {
                    //Training instances
                    File dataFile = new File(dataFileName);
                    dataReader = new BufferedReader(new FileReader(dataFile));
                    data = new Instances(dataReader);
                    
                    if (testFileName.length() != 0) {
	                	try {
	    	                //Instances to rank
	    	                File testFile = new File(testFileName);
	    	                testReader = new BufferedReader(new FileReader(testFile));
	    	                test = new Instances(testReader);
	    	                
	    	                if (data.equalHeaders(test)) {
	    		                if (classIndex == -1)
	    		                    test.setClassIndex(test.numAttributes() - 1);
	    		                else
	    		                	test.setClassIndex(classIndex - 1);
	    	                } else {
	    	                    throw new Exception("Train and Test instances must have" +
	    	                                        "the same format.");
	    	                }
	                	} catch(IOException e) {
	                        throw new Exception("File can't be open: " + 
	                                e.getMessage());
	                	}
                    }
                    
                    if (classIndex == -1) {
                        data.setClassIndex(data.numAttributes() - 1);
                    } else {
                        data.setClassIndex(classIndex - 1);
                    }
                } catch(IOException e) {
                    throw new Exception("File can't be open: " + 
                                        e.getMessage());
                }
            }

            // Removing instances with missing classes in training dataset
            removeMissingClassInstances(data);            
            ranker.setOptions(args);

            // Ranking with test file
            if (testFileName.length() != 0) {
            	removeMissingClassInstances(test);

                trainTimeStart = System.currentTimeMillis();
                ranker.buildClassifier(data);
                trainTimes[0] = System.currentTimeMillis() - trainTimeStart;

                testTimeStart = System.currentTimeMillis();
                ranker.rank(test);
                testTimes[0] = System.currentTimeMillis() - testTimeStart;
                
                results.add(test);

            // Ranking with cross-validation
            } else {
            	if (cvFoldsString.length() != 0)
            		cvFolds = Integer.parseInt(cvFoldsString);
            	
            	trainTimes = new long[cvFolds];
            	testTimes = new long[cvFolds];
            	
            	Random random = new Random(seed);
            	
            	Instances instances = new Instances(data);
            	instances.randomize(random);
            	instances.stratify(cvFolds);
            	
            	for (int fold = 0; fold < cvFolds; fold++){
            		System.out.println("--- Fold #" + fold + " ---");
            		Instances train = instances.trainCV(cvFolds, fold);
            		test = instances.testCV(cvFolds, fold);

                    trainTimeStart = System.currentTimeMillis();
                    ranker.buildClassifier(train);
                    trainTimes[fold] = System.currentTimeMillis() - trainTimeStart;

                    testTimeStart = System.currentTimeMillis();
                    ranker.rank(test);
                    testTimes[fold] = System.currentTimeMillis() - testTimeStart;
                    
                    results.add(test);
            	}
            }

            //Writing the output
            try {
                File rankFile = new File(rankFileName);
                rankWriter = new BufferedWriter(new FileWriter(rankFile));

                int counter = 0;
                long totalTrainTime = 0, totalTestTime = 0;
            	double sum = 0;
            	double[] auc = new double[results.size()];
                Enumeration<Instances> ranks = results.elements();
                
                while (ranks.hasMoreElements()) {
                	data = ranks.nextElement();
                	
                	rankWriter.write("--- Ranking #" + counter++ + " ---");
	                rankWriter.newLine();
	                
                	auc[counter-1] = ranker.auc(data);
                	sum += auc[counter-1];
                	
	                //Writing the auc
	                rankWriter.write("Area under the curve: " + auc[counter-1]);
	                rankWriter.newLine();
	
	                //Writing the training, test and total times
	                totalTrainTime += trainTimes[counter-1];
	                rankWriter.write("Training time: " + trainTimes[counter-1] + "ms");
	                rankWriter.newLine();
	
	                totalTestTime += testTimes[counter-1];
	                rankWriter.write("Test time: " + testTimes[counter-1] + "ms");
	                rankWriter.newLine();
	
	                rankWriter.write("Total time: " + (trainTimes[counter-1] + testTimes[counter-1]) + "ms");
	                rankWriter.newLine();
	                rankWriter.newLine();
                }

                double mean = 0, variance = 0, stdDeviation = 0;

                rankWriter.write("--- Stats ---");
                rankWriter.newLine();

                //Mean
                mean = sum / results.size();
                rankWriter.write("Mean: " + mean);
                rankWriter.newLine();

                //Variance
                for (int i = 0; i < results.size(); i++)
                	variance += Math.pow(auc[i] - mean, 2);
                variance /= results.size();
                rankWriter.write("Variance: " + variance);
                rankWriter.newLine();

                //Standard Deviation
                stdDeviation = Math.sqrt(variance);
                rankWriter.write("Standard deviation: " + stdDeviation);
                rankWriter.newLine();
                rankWriter.newLine();
                
                //Printing total times
                rankWriter.write("Total training time: " + totalTrainTime + "ms");
                rankWriter.newLine();
                
                rankWriter.write("Total test time: " + totalTestTime + "ms");
                rankWriter.newLine();
                
                rankWriter.write("Total time: " + (totalTrainTime + totalTestTime) + "ms");
                
                rankWriter.close();
            } catch(IOException e) {
                throw new IOException("File can't be created or open: " + 
                                    e.getMessage() + "; " + e.getCause().toString());
            }

        } catch(Exception e) {
            System.err.println("Weka exception: " + e.getMessage() +
                                makeOptionString(ranker));
        }

    }

}