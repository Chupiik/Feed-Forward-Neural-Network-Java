import java.util.List;
import java.util.Random;

public class Main {

     static void main() {

        System.out.println("Setting up...");
        Random random = new Random(42);

        NeuralNetwork network = new NeuralNetwork(0.01, random, 784, 128, 64, 10);

        String trainVectorsPath = "data/fashion_mnist_train_vectors.csv";
        String trainLabelsPath = "data/fashion_mnist_train_labels.csv";
        System.out.println("Loading training data from " + trainVectorsPath);
        List<MnistImage> trainingData = DataReader.loadData(trainVectorsPath, trainLabelsPath);

        int epochs = 100;
        System.out.println("Starting training for " + epochs + " epochs...");

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalEpochError = 0;


            for (int i = 0; i < trainingData.size(); i++) {
                MnistImage image = trainingData.get(i);

                double[] expectedOutput = createOneHotVector(image.label(), 10);

                totalEpochError += network.train(image.pixels(), expectedOutput);

                if ((i + 1) % 10000 == 0) {
                    System.out.printf("  Epoch %d: Processed %d / %d images\n", epoch + 1, i + 1, trainingData.size());
                }
            }
            System.out.printf("Epoch %d complete. Average Error: %.6f\n", epoch + 1, totalEpochError / trainingData.size());
        }
        System.out.println("Training finished.");


    }

    public static double[] createOneHotVector(int label, int numClasses) {
        double[] oneHot = new double[numClasses];
        oneHot[label] = 1.0;
        return oneHot;
    }
}