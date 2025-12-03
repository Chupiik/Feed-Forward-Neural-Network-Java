// 582919 Patrik Chupáč

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * The main entry point for the Neural Network application.
 * <p>
 * This class handles the setup of the network, loading of data,
 * execution of the training loop (using the Adam optimizer),
 * and generation of the final prediction files for submission.
 */
public class Main {

    /**
     * The main method that orchestrates the training and evaluation process.
     * <p>
     * It performs the following steps:
     * 1. Initializes the neural network and random number generator.
     * 2. Loads the Fashion-MNIST training data.
     * 3. Splits data into training and validation sets.
     * 4. Runs the training loop for a specified number of epochs, checking validation accuracy.
     * 5. Implements early stopping if validation accuracy does not improve.
     * 6. Generates prediction files ('train_predictions.csv' and 'test_predictions.csv').
     * 7. Prints the final execution time.
     *
     * @param args Command line arguments (not used).
     */
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


    /**
     * Generates predictions for a given dataset and writes them to a CSV file.
     * <p>
     * The output format corresponds to the assignment requirements:
     * one integer (class label) per line.
     *
     * @param network  The trained neural network.
     * @param data     The list of images to predict.
     * @param filename The name of the output file (e.g., "train_predictions.csv").
     */
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

    /**
     * Evaluates the network on a given dataset and prints the accuracy to the console.
     *
     * @param network The trained neural network.
     * @param data    The dataset to evaluate against.
     */
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


    /**
     * Creates a One-Hot encoded vector for a given class label.
     * <p>
     * For example, if label is 2 and numClasses is 10, returns [0, 0, 1, 0, ...].
     *
     * @param label      The integer class label (0-9).
     * @param numClasses The total number of classes (10 for Fashion-MNIST).
     * @return A double array where the index corresponding to the label is 1.0, others 0.0.
     */
    public static double[] createOneHotVector(int label, int numClasses) {
        double[] oneHot = new double[numClasses];
        oneHot[label] = 1.0;
        return oneHot;
    }
}