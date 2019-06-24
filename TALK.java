package Classification;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instances;

public class TALK {
	public static String address = "src/data/arff/experiment/uniformbinarydata/arcene.arff";;
	public static final double percentage = 0.1;
	public static int numBought = 50;
	public static final int kValue = 5;
	
	public static final int TAUGHT  = 0;
	public static final int GUESSED = 1;
	public static final int DELAYED = 2;
	
	private Instances data;
	private double TCost;
	private double[][] MCost;
	
	private int[] statuVector;
	
	// Real labels
	private int[] labelVector;
	
	
	// Predict labels
	private int[] predictVector;
	private double[][] distanceMatrix;
	private double[] expectedVector;
	private int bestLabel;
	
	private int numTaught = 0;
	private int numPredict = 0;
	private int numCurrentTaught = 0;
	private int numCurrentPredict = 0;
	private boolean terminalSign = false;
	
	
	
	public void getOverlapDistance() {
		distanceMatrix = new double[data.numInstances()][data.numInstances()];
		
		for (int i = 0; i < data.numInstances(); i ++) {
			for (int j = 0; j < data.numInstances(); j ++) {
				double distance = 0;
				for (int k = 0; k < data.numAttributes() - 1; k ++) {
					double valueA = data.instance(i).value(k);
					double valueB = data.instance(j).value(k);
					if ((int)valueA != (int)valueB) {
						distance ++;
					}// Of if
				}// Of for k
				// assignment
				distanceMatrix[i][j] = distance;
			}// Of for j
		}// Of for i
	}// Of getOverlapDistance
	
	
	
	public void getEuclideanDistance() {
		distanceMatrix = new double[data.numInstances()][data.numInstances()];
		
		for (int i = 0; i < data.numInstances(); i ++) {
			for (int j = 0; j < data.numInstances(); j ++) {
				double distanceSquare = 0;
				for (int k = 0; k < data.numAttributes() - 1; k ++) {
					double valueA = data.instance(i).value(k);
					double valueB = data.instance(j).value(k);
					distanceSquare += Math.pow((valueA - valueB), 2);
				}// Of for k
				distanceMatrix[i][j] = Math.sqrt(distanceSquare);
			}// Of for j
		}// Of for i
	}// Of getEuclideanDistance
	
	
	
	public void getManhattanDistance() {
		distanceMatrix = new double[data.numInstances()][data.numInstances()];
		
		for (int i = 0; i < data.numInstances(); i ++) {
			for (int j = 0; j < data.numInstances(); j ++) {
				double distance = 0;
				for (int k = 0; k < data.numAttributes() - 1; k ++) {
					double valueA = data.instance(i).value(k);
					double valueB = data.instance(j).value(k);
					distance += Math.abs(valueA - valueB);
				}// Of for k
				distanceMatrix[i][j] = distance;
			}// Of for j
		}// Of for i
	}// Of getManhattanDistance
	
	
	
	public void getTerminalSign () {
		terminalSign = true;
		for (int i = 0; i < statuVector.length; i ++) {
			if (statuVector[i] == DELAYED) {
				terminalSign = false;
				break;
			}// Of if
		}// Of for i
	}// Of getTerminalSign
	
	
	
	public int[] getNeibors (int index) {
		int[] neighborVector = new int[kValue + 1];
		double[] distanceVector = new double[kValue + 1];
	
		int determinedNeighors = 0;
		for (int i = 0; i < data.numInstances(); i ++) {
			if (statuVector[i] == DELAYED) {
				continue;
			}//Of if
			
			double tempCurrentDistance = distanceMatrix[index][i];
			int j;
			for (j = determinedNeighors - 1; j >= 0; j --) {
				if (distanceVector[j] > tempCurrentDistance) {
					distanceVector[j + 1] = distanceVector[j];
					distanceVector[j + 1] = distanceVector[j];
				} else {
					break;
				}//Of if
			}//Of for j
			
			distanceVector[j + 1] = tempCurrentDistance;
			neighborVector[j + 1] = i;
			
			if (determinedNeighors < kValue) {
				determinedNeighors ++;
			}//Of if
		}// Of for i
		
		int[] resultNeighbors =  new int[kValue];
		for (int i = 0; i < kValue; i ++) {
			resultNeighbors[i] = neighborVector[i];
		}// Of for i

		return resultNeighbors;
	}// Of getNeighbors
	
	
	
	public double getExpectedValue (int[] neighborVector) {
		double tempCost = 0;
		double minimalCost = Double.MAX_VALUE;
		int tempClassLabel;
		
		for (int i = 0; i < data.numClasses(); i ++) {
			tempCost = 0;
			for (int j = 0; j < neighborVector.length; j ++) {
				tempClassLabel = predictVector[neighborVector[j]];
				tempCost += MCost[tempClassLabel][i];
			}//Of for j
			
			if (minimalCost > tempCost) {
				minimalCost = tempCost;
				bestLabel = i;
			}//Of if
		}//Of for i
		
		return minimalCost / kValue;
	}// Of getExpectedValue
	
	
	
	public void getPredictedInstances () {
		numCurrentPredict = 0;
		for (int i = 0; i < statuVector.length; i ++) {
			if (statuVector[i] == TAUGHT) {
				continue;
			} else if (statuVector[i] == GUESSED) {
				continue;
			}// Of if
			
			int[] neighborVector = getNeibors(i);
			double expectedValue = getExpectedValue(neighborVector);
			expectedVector[i] = expectedValue;
			
			if (expectedValue < TCost) {
				predictVector[i] = bestLabel;
				statuVector[i] = GUESSED;
				
				numCurrentPredict ++;
				numPredict ++;
			}//Of if
		}// Of for i
	}// Of getPredictedInstances
	
	
	
	public void getBoughtInstances() {
		int[] tempIndices = new int[numBought + 1];
		double[] tempCosts = new double[numBought + 1];
		numCurrentTaught = 0;
				
		for (int i = 0; i < data.numInstances(); i ++) {
			if (statuVector[i] != DELAYED) {
				continue;
			}//Of if
				
			double tempExceptedMisclassificationCosts = expectedVector[i];
			int j;
			for (j = numCurrentTaught - 1; j >= 0; j--) {
				if (tempCosts[j] < tempExceptedMisclassificationCosts) {
					tempCosts[j + 1] = tempCosts[j];
					tempIndices[j + 1] = tempIndices[j];
				} else {
					break;
				}// Of if 
			}// Of for j
					
			tempCosts[j + 1] = tempExceptedMisclassificationCosts;
			tempIndices[j + 1] = i;
					
			if (numCurrentTaught < numBought) {
				numCurrentTaught ++;
			}// Of if 
		}//Of for i
		
		for (int i = 0; i < numCurrentTaught; i ++) {
			statuVector[tempIndices[i]] = TAUGHT;
			predictVector[tempIndices[i]] = labelVector[tempIndices[i]];
			numTaught ++;
		}//Of for i
	}// Of getBoughtInstances
	
	
	
	public int[][] getIterativeLearning () {
		int[][] tempRecordMatrix = new int[data.numInstances() / numBought + 1][4];
		int times = 0;
		
		while (!terminalSign) {
			getPredictedInstances();
			tempRecordMatrix[times][0] = numCurrentPredict;
			tempRecordMatrix[times][1] = numPredict;
			
			getBoughtInstances();
			tempRecordMatrix[times][2] = numCurrentTaught;
			tempRecordMatrix[times][3] = numTaught;
			
			getTerminalSign();
			times ++;
		}// Of while
		
		int[][] recordMatrix = new int[times][];
		for (int i = 0; i < times; i ++) {
			recordMatrix[i] = tempRecordMatrix[i];
		}// Of for i
		
		return recordMatrix;
	}// Of getIterativeLearning
	
	
	
	public double getAccuracy () {
		int count = 0;
		for (int i = 0; i < labelVector.length; i ++) {
			if (labelVector[i] == predictVector[i]) {
				count ++;
			}// Of if
		}// Of for i
		double accuracy = (count + 0.0) / labelVector.length;
		
		return accuracy;
	}// Of getAccuracy
	
	
	public double getClassificationCost () {
		double totalCost = 0;
		for (int i = 0; i < labelVector.length; i ++) {
			totalCost += MCost[predictVector[i]][labelVector[i]];
		}// Of for i
		for (int i = 0; i < statuVector.length; i ++) {
			if (statuVector[i] == TAUGHT) {
				totalCost += TCost;
			}// Of if
		}// Of for i
		double aveCost = totalCost / labelVector.length;
		
		return aveCost;
	}// Of getClassificationCost
	
	
	
	public void experiment () {
		//getOverlapDistance();
		getEuclideanDistance();
		int[][] recordMatrix = getIterativeLearning();
//		System.out.println(Arrays.deepToString(recordMatrix));
		
		double accuracy = getAccuracy();
		double aveCost = getClassificationCost();
//		System.out.println("Accuracy " + accuracy);
//		System.out.println("Average cost " + aveCost);
	}// Of experiment
	
	public void setInitialLabels () {
		boolean[] tempArray = null;
		try {
			tempArray = SimpleTool.generateBooleanArrayForDivision(data.numInstances(), 2.0/data.numInstances());
			//tempArray = SimpleTool.generateBooleanArrayForDivision(data.numInstances(), percentage);
		} catch (Exception e) {
			e.printStackTrace();
		}// Of try
		for (int i = 0; i < tempArray.length; i ++) {
			if (tempArray[i]) {
				statuVector[i] = TAUGHT;
				predictVector[i] = labelVector[i];
				numTaught ++;
			}// Of if
		}// Of for i
	}// Of setInitialLabels
	
	
	public void setInitialLabels1() {
		boolean[] tempArray = new boolean[data.numInstances()];
		Random random = new Random();
		int r1 = random.nextInt(data.numInstances());
		tempArray[r1] = true;
		int r2 = 0;
		while ((r2 = random.nextInt(data.numInstances())) == r1 || labelVector[r2] == labelVector[r1]) {}
		tempArray[r2] = true;	
		for (int i = 0; i < tempArray.length; i ++) {
			if (tempArray[i]) {
				statuVector[i] = TAUGHT;
				predictVector[i] = labelVector[i];
				numTaught ++;
			}// Of if
		}// Of for i
	}// Of setInitialLabels
	
	public void setStatuVector () {
		statuVector = new int[data.numInstances()];
		for (int i = 0; i < data.numInstances(); i ++) {
			statuVector[i] = DELAYED;
		}// Of for i
	}// Of setStatuVector
	
	
	
	public void setActualVector () {
		expectedVector = new double[data.numInstances()];
		predictVector = new int[data.numInstances()];
		labelVector = new int[data.numInstances()];
		for (int i = 0; i < data.numInstances(); i ++) {
			labelVector[i] = (int)data.instance(i).classValue();
		}// Of for i
	}// Of setActualVector
	
	
	
	public void setData () {
		try {
			FileReader fileReader = new FileReader(address);
			data = new Instances(fileReader);
			fileReader.close();
			
			data.setClassIndex(data.numAttributes() - 1);
		} catch (Exception e) {
			e.printStackTrace();
		}// Of try
	}// Of setData
	
	
	
	public void setCost () {
		TCost = 1;
		MCost = new double[][]{{0, 2}, {4, 0}};
		//MCost = new double[data.numClasses()][data.numClasses()];
		
//		for (int i = 0; i < MCost.length; i ++) {
//			for (int j = 0; j < MCost[i].length; j ++) {
//				if (i == j) {
//					MCost[i][j] = 0;
//				} else {
//					MCost[i][j] = 50;
//				}// Of if
//			}// Of for j
//		}// Of for i
	}// Of for setCost
	
	
	
	public static void main (String[] args) {
		String[] strings = {
				"ALLAML",
				"arcene",
				"banana",
				"credit6000_126",
				"german",
				"heart",
				"ionosphere_real",
				"jain",
				"madelon",
				"sonar",
				"spambase",
				"thyroid"
		};
		int[] steps = new int[]{1, 5, 30, 50, 5, 3, 3, 3, 40, 4, 25, 3};
		int[] numIteratives = new int[]{20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20};
		
		for (int i = 0; i < strings.length; i++) {
			
			System.out.println(strings[i]);
			
			for (int j = 0; j < numIteratives[i]; j++) {
				TALK learner = new TALK();
				learner.address = "src/data/arff/experiment/uniformbinarydata/"+strings[i]+".arff";
				learner.numBought = 1;
				learner.setData();
				learner.setCost();
				learner.setStatuVector();
				learner.setActualVector();
				learner.setInitialLabels1();
				learner.getEuclideanDistance();
				learner.getIterativeLearning();
				System.out.print(learner.getClassificationCost() + "\t");
				// Output accuracy
				int[] predictedLabels = learner.predictVector;
				
				int[] realLabels = learner.labelVector;
				
				// double acc = accuracy(predictedLabels, realLabels);
				// System.out.println(acc);
			}// Of for j
			System.out.println();
		}// Of for i
		
	}// Of for main
}// Of class TALK
