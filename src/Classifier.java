import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Classifier {
    boolean laplacianCorrection = false;
    int N = 0;
    Map<String, Integer> feature1Indices = new HashMap<String, Integer>();
    Map<String, Integer> feature2Indices = new HashMap<String, Integer>();
    Map<String, Integer> feature3Indices = new HashMap<String, Integer>();
    Map<String, Integer> feature4Indices = new HashMap<String, Integer>();
    Map<String, Integer> feature5Indices = new HashMap<String, Integer>();
    Map<String, Integer> feature6Indices = new HashMap<String, Integer>();
    Map<String, Integer> majorityVoteMap = new HashMap<String, Integer>();


    public ArrayList<Record> learningDataset = new ArrayList<>();
    public ArrayList<Record> testDataset = new ArrayList<>();
    ArrayList<String> uniqueClasses = new ArrayList<>();
    Map<String, Integer> classesProbabilities = new HashMap<>();

    public Classifier() {
  
        feature1Indices.put("vhigh", 1);
        feature1Indices.put("high", 2);
        feature1Indices.put("med", 3);
        feature1Indices.put("low", 4);

        feature2Indices.put("vhigh", 1);
        feature2Indices.put("high", 2);
        feature2Indices.put("med", 3);
        feature2Indices.put("low", 4);

        feature3Indices.put("2", 1);
        feature3Indices.put("3", 2);
        feature3Indices.put("4", 3);
        feature3Indices.put("5more", 4);

        feature4Indices.put("2", 1);
        feature4Indices.put("4", 2);
        feature4Indices.put("more", 3);

        //small, med, big
        feature5Indices.put("small", 1);
        feature5Indices.put("med", 2);
        feature5Indices.put("big", 3);

        feature6Indices.put("low", 1);
        feature6Indices.put("med", 2);
        feature6Indices.put("high", 3);
    }

    public static void run() throws IOException {
        Classifier cl = new Classifier();
        cl.start();
    }

    public void start() throws IOException {

        loadLearningData();
        loadTestData();

        classify();
        writeClassifiedTestData("Naive bayesian");
        //System.out.println("Finished Writing records back");
        System.out.println("\nBayesian Classifier accuracy = " + calculateClassifierAccuracy("Naive bayesian"));
        
        performK_NearestNeighbourClassification();
        writeClassifiedTestData("K Nearest Neighbour");
        //System.out.println("Finished Writing records back");
        System.out.println("\nK Nearest Neighbour Classifier accuracy = " + calculateClassifierAccuracy("K Nearest Neighbour"));


    }

    public void loadLearningData() throws IOException {
        String fileName = "Training dataset.txt";
        int numberOfCars = 1296;
        RandomAccessFile file = new RandomAccessFile(fileName, "r");
        file.seek(0);
        for (int i = 0; i < numberOfCars; i++) {
            String inputParts[] = file.readLine().split(",");
            String id = inputParts[0];
            String bPrice = inputParts[1];
            String mPrice = inputParts[2];
            String doors = inputParts[3];
            String capacity = inputParts[4];
            String lugg = inputParts[5];
            String safety = inputParts[6];
            String carAcc = inputParts[7];
            Record r = new Record(id, bPrice, mPrice, doors, capacity, lugg, safety);
            r.setCarAcceptability(carAcc);
            learningDataset.add(r);
            if (uniqueClasses.indexOf(carAcc) == -1) {
                uniqueClasses.add(carAcc);
                classesProbabilities.put(carAcc, 1);
            } else {
                classesProbabilities.put(carAcc, classesProbabilities.get(carAcc) + 1);
            }
        }
        N = learningDataset.size();
        file.close();
    }

    public void loadTestData() throws IOException {
        String fileName = "Test dataset.txt";
        int numberOfCars = 432;
        RandomAccessFile file = new RandomAccessFile(fileName, "r");
        file.seek(0);
        for (int i = 0; i < numberOfCars; i++) {
            String inputParts[] = file.readLine().split(",");
            String id = inputParts[0];
            String bPrice = inputParts[1];
            String mPrice = inputParts[2];
            String doors = inputParts[3];
            String capacity = inputParts[4];
            String lugg = inputParts[5];
            String safety = inputParts[6];
            Record r = new Record(id, bPrice, mPrice, doors, capacity, lugg, safety);
            testDataset.add(r);
        }
        file.close();
    }

    public void classify() {
        // Loop through all objects
        for (int i = 0; i < testDataset.size(); i++) {

            // Loop through all class labels to determine the best
            double bestProbability = -1;
            int indexOfBestClass = -1;
            for (int j = 0; j < uniqueClasses.size(); j++) {
                // System.out.println("Hi");
                String C = uniqueClasses.get(j);
                double p = calculateBayesianProbability(testDataset.get(i), C);
                if (bestProbability < p) {
                    indexOfBestClass = j;
                    bestProbability = p;
                }
            }

            // Set the best class as the label
            testDataset.get(i).setCarAcceptability(uniqueClasses.get(indexOfBestClass));
        }
    }

    private double calculateBayesianProbability(Record X, String C) {
        // bayes = P(Ci|X) = P(X|Ci) * P(C)
        // P(X|Ci) = PROD [P(xk,Ci)] where k loops through the attributes of X
        // = P(x1,Ci) * P(x2,Ci) * ..... P(x6,Ci)
        double probOfClass = classesProbabilities.get(C) * 1.0 / learningDataset.size();
        double prod = tupleGivenClass(X, C); // P(X|Ci)
        // System.out.println(prod + " " + probOfClass);
        double bayes = prod * probOfClass;
        return bayes;
    }

    private double tupleGivenClass(Record X, String C) {
        double p = getProbabilityOfAttributeAndClass(1, X.BuyingPrice, C);
        p *= getProbabilityOfAttributeAndClass(2, X.MaintenancePrice, C);
        p *= getProbabilityOfAttributeAndClass(3, X.NumberOfDoors, C);
        p *= getProbabilityOfAttributeAndClass(4, X.Capacity, C);
        p *= getProbabilityOfAttributeAndClass(5, X.SizeOfLuggageBoot, C);
        p *= getProbabilityOfAttributeAndClass(6, X.EstimatedSafety, C);
        return p;
    }

    private double getProbabilityOfAttributeAndClass(int attNumber, String value, String C) {
        int count = getCOUNT(attNumber, value, C);
        int ALPHA = 0;
        if (count == 0) {
            ALPHA = 1;
        }
        // The equation below is Laplace smoothing
        double p = (1.0 * count + ALPHA) / (classesProbabilities.get(C) + 6 * ALPHA);
        // System.out.println(p);
        return p;
    }

    private int getCOUNT(int attNumber, String value, String C) {
        int p = 0;
        if (attNumber == 1) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).BuyingPrice.compareTo(value) == 0
                        && learningDataset.get(i).CarAcceptability.compareTo(C) == 0) {
                    p++;
                }
            }
        } else if (attNumber == 2) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).MaintenancePrice.compareTo(value) == 0
                        && learningDataset.get(i).CarAcceptability.compareTo(C) == 0) {
                    p++;
                }
            }
        }
        if (attNumber == 3) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).NumberOfDoors.compareTo(value) == 0
                        && learningDataset.get(i).CarAcceptability.compareTo(C) == 0) {
                    p++;
                }
            }
        } else if (attNumber == 4) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).Capacity.compareTo(value) == 0
                        && learningDataset.get(i).CarAcceptability.compareTo(C) == 0) {
                    p++;
                }
            }
        } else if (attNumber == 5) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).SizeOfLuggageBoot.compareTo(value) == 0
                        && learningDataset.get(i).CarAcceptability.compareTo(C) == 0) {
                    p++;
                }
            }
        } else if (attNumber == 6) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).EstimatedSafety.compareTo(value) == 0
                        && learningDataset.get(i).CarAcceptability.compareTo(C) == 0) {
                    p++;
                }
            }
        }
        return p;
    }

    private void writeClassifiedTestData(String classifierName) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("Classified Test dataset with " + classifierName + ".txt"));
        for (int i = 0; i < testDataset.size(); i++) {
            String output = testDataset.get(i).toString();
            bw.write(output);
        }
        bw.close();
    }

    private double calculateClassifierAccuracy(String classifierName) throws IOException {
        int n = 432 ;
        double accuracy = 0 ;
        String fileName = "Test dataset with class label.txt";
        String fileName2 = "Classified Test dataset with " + classifierName + ".txt";
        
        RandomAccessFile file = new RandomAccessFile(fileName, "r");
        RandomAccessFile file2 = new RandomAccessFile(fileName2, "r");
        file.seek(0);
        for (int i = 0; i < n; i++) {
            String inputParts[] = file.readLine().split(",");
            String inputParts2[] = file2.readLine().split(",");
            String label1 = inputParts[7];
            String label2 = inputParts2[7];
            if(label1.compareTo(label2) == 0){
                accuracy++ ;
            }
        }
        file.close();
        file2.close();
        return accuracy * 1.0 / n ;
    }

    private void resetTestDatasetClassLabels(){
        for(int i = 0 ; i < testDataset.size() ; i++){
            testDataset.get(i).CarAcceptability = "" ;
        }
    }

    private void performK_NearestNeighbourClassification(){
        resetTestDatasetClassLabels();
        for(int i = 0 ; i < testDataset.size(); i++) {
            Record test = testDataset.get(i);
            String majorityVoteTo = makeMajorityVote(getAllDistances(test));
            testDataset.get(i).setCarAcceptability(majorityVoteTo);
        }
    }

    private String makeMajorityVote(ArrayList<Neighbour> neighbours){
        majorityVoteMap.clear();
        for(int i = 0 ; i < 5 ; i++){
            majorityVoteMap.put(neighbours.get(i).classLabel, 1) ;
        }
        Map.Entry<String, Integer> maxEntry = null;
        for (Map.Entry<String, Integer> entry : majorityVoteMap.entrySet())
        {
            if (maxEntry == null || entry.getValue() > maxEntry.getValue())
            {
                maxEntry = entry;
            }
        }
        majorityVoteMap.clear();
        String vote = maxEntry.getKey();
        return vote ;
    }

    private ArrayList<Neighbour> getAllDistances(Record r1){
        ArrayList<Neighbour> neighbours = new ArrayList<Neighbour>() ;
        for(int i = 0; i < N ; i++){
            double distance = getManhattanDistance(r1, learningDataset.get(i));
            neighbours.add(new Neighbour(distance, learningDataset.get(i).CarAcceptability));
        }
        
        neighbours.sort(new Neighbour());
        ArrayList<Neighbour> bestNeighbours = new ArrayList<Neighbour>() ;
        for(int i = 0 ; i < 5 ; i++)    bestNeighbours.add(neighbours.get(i));
        return bestNeighbours ;
    }

    private double getManhattanDistance(Record r1, Record r2) {
        double distance = 0 ;
        distance += Math.abs(feature1Indices.get(r1.BuyingPrice) - feature1Indices.get(r2.BuyingPrice)) * 0.00001;
        distance += Math.abs(feature2Indices.get(r1.MaintenancePrice) - feature2Indices.get(r2.MaintenancePrice))  * 0.001;
        distance += Math.abs(feature3Indices.get(r1.NumberOfDoors) - feature3Indices.get(r2.NumberOfDoors)) * 0.1 ;
        distance += Math.abs(feature4Indices.get(r1.Capacity) - feature4Indices.get(r2.Capacity)) * 10 ;
        distance += Math.abs(feature5Indices.get(r1.SizeOfLuggageBoot) - feature5Indices.get(r2.SizeOfLuggageBoot))  * 100;
        distance += Math.abs(feature6Indices.get(r1.EstimatedSafety) - feature6Indices.get(r2.EstimatedSafety)) * 10000;
        return distance ;
    }

}
