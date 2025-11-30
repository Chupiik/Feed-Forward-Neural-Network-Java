// 582919 Patrik Chupáč

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();

        System.out.println("Setting up...");
        Random random = new Random(0);

        NeuralNetwork network = new NeuralNetwork(0.0001,0.6,0, random, 784, 128, 64, 10);

        String trainVectorsPath = "data/fashion_mnist_train_vectors.csv";
        String trainLabelsPath = "data/fashion_mnist_train_labels.csv";
        System.out.println("Loading all training data from " + trainVectorsPath);
        List<MnistImage> allTrainingData = DataReader.loadData(trainVectorsPath, trainLabelsPath);

        //Collections.shuffle(allTrainingData, random);

        int validationSize = allTrainingData.size() / 10;
        List<MnistImage> validationData = allTrainingData.subList(0, validationSize);
        List<MnistImage> trainingData = allTrainingData.subList(validationSize, allTrainingData.size());

        System.out.println("Data split into:");
        System.out.println(" - Training set size: " + trainingData.size());
        System.out.println(" - Validation set size: " + validationData.size());


        int epochs = 15;
        double bestValidationAccuracy = 0.0;
        int epochsWithoutImprovement = 0;
        final int patience = 2;

        System.out.println("\nStarting training for up to " + epochs + " epochs...");

        for (int epoch = 0; epoch < epochs; epoch++) {

            //Collections.shuffle(trainingData, random);

            for (int i = 0; i < trainingData.size(); i++) {
                MnistImage image = trainingData.get(i);
                double[] expectedOutput = createOneHotVector(image.label(), 10);
                network.trainADAM(image.pixels(), expectedOutput);
            }

            int correctValidation = 0;
            for(MnistImage image : validationData) {
                if (network.predict(image.pixels()) == image.label()) {
                    correctValidation++;
                }
            }
            double validationAccuracy = (double) correctValidation / validationData.size();

            System.out.printf("Epoch %d complete. Validation Accuracy: %.4f\n",
                    epoch + 1, validationAccuracy);

            if (validationAccuracy > bestValidationAccuracy) {
                bestValidationAccuracy = validationAccuracy;
                epochsWithoutImprovement = 0;
                System.out.println("  -> New best validation accuracy!");
            } else {
                epochsWithoutImprovement++;
                if (epochsWithoutImprovement >= patience) {
                    System.out.printf("Stopping early. Validation accuracy has not improved for %d epochs.\n", patience);
                    break;
                }
            }
        }
        System.out.println("Training finished.");


        System.out.println("\n--- Final Evaluation Phase ---");

        System.out.println("\nGenerating train_predictions.csv...");
        savePredictions(network, allTrainingData, "train_predictions.csv");

        String testVectorsPath = "data/fashion_mnist_test_vectors.csv";
        String testLabelsPath = "data/fashion_mnist_test_labels.csv";
        System.out.println("Loading test data...");
        List<MnistImage> testData = DataReader.loadData(testVectorsPath, testLabelsPath);

        System.out.println("Generating test_predictions.csv...");
        savePredictions(network, testData, "test_predictions.csv");

        evaluateAndPrint(network, testData);

        long endTime = System.currentTimeMillis();
        double totalTimeSeconds = (endTime - startTime) / 1000.0;
        System.out.println("\nTotal execution time: " + totalTimeSeconds + " seconds.");

    }


    private static void savePredictions(NeuralNetwork network, List<MnistImage> data, String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (MnistImage image : data) {
                int prediction = network.predict(image.pixels());
                writer.write(Integer.toString(prediction));
                writer.newLine();
            }
        } catch (IOException e) {
            System.err.println("Error writing to file: " + filename);
            e.printStackTrace();
        }
    }

    private static void evaluateAndPrint(NeuralNetwork network, List<MnistImage> data) {
        int correctPredictions = 0;
        for (MnistImage image : data) {
            if (network.predict(image.pixels()) == image.label()) {
                correctPredictions++;
            }
        }
        double accuracy = (double) correctPredictions / data.size() * 100.0;
        System.out.printf("Final Test Accuracy: %.2f%%\n", accuracy);
    }


    public static double[] createOneHotVector(int label, int numClasses) {
        double[] oneHot = new double[numClasses];
        oneHot[label] = 1.0;
        return oneHot;
    }
}


//import java.util.Arrays;
//import java.util.List;
//import java.util.Random;
//
//public class Main {
//
//    public static void main(String[] args) {
//
//        Random random = new Random(42);
//
//
//        NeuralNetworkXOR network = new NeuralNetworkXOR(0.1, 0.5, random, 2, 4, 1);
//
//        double[][] inputs = {
//                {0, 0},
//                {0, 1},
//                {1, 0},
//                {1, 1}
//        };
//        double[][] expectedOutputs = {
//                {0},
//                {1},
//                {1},
//                {0}
//        };
//
//        int epochs = 5000;
//        System.out.println("Starting training for " + epochs + " epochs...");
//
//        for (int epoch = 0; epoch < epochs; epoch++) {
//            double totalEpochError = 0;
//            for (int i = 0; i < inputs.length; i++) {
//                totalEpochError += network.train(inputs[i], expectedOutputs[i]);
//            }
//
//            if ((epoch + 1) % 500 == 0) {
//                System.out.printf("Epoch %d complete. Average Error: %.8f\n", epoch + 1, totalEpochError / inputs.length);
//            }
//        }
//        System.out.println("Training finished.");
//
//        System.out.println("\n--- Final Predictions for XOR ---");
//        for (int i = 0; i < inputs.length; i++) {
//            List<double[]> activations = network.feedForward(inputs[i]);
//            double prediction = activations.get(activations.size() - 1)[0];
//            System.out.printf("Input: %s, Expected: %s, Prediction: %.4f\n",
//                    Arrays.toString(inputs[i]),
//                    Arrays.toString(expectedOutputs[i]),
//                    prediction);
//        }
//    }
//}