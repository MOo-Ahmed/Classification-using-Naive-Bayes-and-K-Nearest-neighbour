import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Classifier {
    boolean laplacianCorrection = false ;
    int N = 0 ;

    public ArrayList<Record> learningDataset = new ArrayList<>();
    public ArrayList<Record> testDataset = new ArrayList<>();
    ArrayList<String> uniqueClasses = new ArrayList<>();
    Map<String, Integer> classesProbabilities = new HashMap<>();

    public Classifier() {
    }

    public static void run() throws IOException {
        Classifier cl = new Classifier();
        cl.start();
    }

    public void start() throws IOException {
        
        loadLearningData();
        loadTestData();

        
        classify();

        writeClassifiedTestData();
        System.out.println("Finished Writing records back");
        

    }

    public void loadLearningData() throws IOException {
        String fileName = "Training dataset.txt";
        int numberOfCars = 1296;
        RandomAccessFile file = new RandomAccessFile(fileName, "r");
        file.seek(0);
        for (int i = 0; i < numberOfCars; i++) {
            String inputParts[] = file.readLine().split(",");
            String bPrice = inputParts[0];
            String mPrice = inputParts[1];
            String doors = inputParts[2];
            String capacity = inputParts[3];
            String lugg = inputParts[4];
            String safety = inputParts[5];
            String carAcc = inputParts[6];
            Record r = new Record(bPrice, mPrice, doors, capacity, lugg, safety);
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
            String bPrice = inputParts[0];
            String mPrice = inputParts[1];
            String doors = inputParts[2];
            String capacity = inputParts[3];
            String lugg = inputParts[4];
            String safety = inputParts[5];
            Record r = new Record(bPrice, mPrice, doors, capacity, lugg, safety);
            testDataset.add(r);
        }
        file.close();
    }

    public void classify() {
        // Loop through all objects
        for (int i = 0; i < testDataset.size(); i++) {
            Record X = testDataset.get(i);
            // Loop through all class labels to determine the best
            double bestProbability = -1;
            int indexOfBestClass = -1;
            for (int j = 0; j < uniqueClasses.size(); j++) {
                String C = uniqueClasses.get(j);
                double p = calculateBayesianProbability(X, C);
                if (bestProbability < p) {
                    indexOfBestClass = j;
                    bestProbability = p;
                }
            }
            // Set the best class as the label
            X.setCarAcceptability(uniqueClasses.get(indexOfBestClass));
        }
    }

    private double calculateBayesianProbability(Record X, String C) {
        // bayes = P(Ci|X) = P(X|Ci) * P(C)
        // P(X|Ci) = PROD [P(xk,Ci)] where k loops through the attributes of X
        // = P(x1,Ci) * P(x2,Ci) * ..... P(x6,Ci)
        double probOfClass = classesProbabilities.get(C) * 1.0 / learningDataset.size();
        double prod = tupleGivenClass(X, C); // P(X|Ci)
        //System.out.println(prod + " " + probOfClass);
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
        return p ;
    }

    private double getProbabilityOfAttributeAndClass(int attNumber, String value, String C) {
        int count = getCOUNT(attNumber, value, C);
        int ALPHA = 0 ;
        if(count == 0) {
            ALPHA = 1 ;
        }
        // The equation below is Laplace smoothing
        double p = (1.0 * count + ALPHA)/ (classesProbabilities.get(C) + 6 * ALPHA) ;
        //System.out.println(p);
        return p;
    }

    private int getCOUNT(int attNumber, String value, String C) {
        int p = 0;
        if (attNumber == 1) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).BuyingPrice == value && learningDataset.get(i).CarAcceptability == C) {
                    p++;
                }
            }
        } else if (attNumber == 2) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).MaintenancePrice == value && learningDataset.get(i).CarAcceptability == C) {
                    p++;
                }
            }
        }
        if (attNumber == 3) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).NumberOfDoors == value && learningDataset.get(i).CarAcceptability == C) {
                    p++;
                }
            }
        } else if (attNumber == 4) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).Capacity == value && learningDataset.get(i).CarAcceptability == C) {
                    p++;
                }
            }
        } else if (attNumber == 5) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).SizeOfLuggageBoot == value && learningDataset.get(i).CarAcceptability == C) {
                    p++;
                }
            }
        } else if (attNumber == 6) {
            for (int i = 0; i < learningDataset.size(); i++) {
                if (learningDataset.get(i).EstimatedSafety == value && learningDataset.get(i).CarAcceptability == C) {
                    p++;
                }
            }
        }
        return p;
    }

    private void writeClassifiedTestData() throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter("Classified Test dataset.txt"));
        for (int i = 0; i < testDataset.size(); i++) {
            String output = testDataset.get(i).toString();
            bw.write(output);
        }
        bw.close();
    }
}
