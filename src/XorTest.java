//// In your XorTrainer.java file
//
//import java.util.Arrays;
//import java.util.List;
//import java.util.Random;
//
//public class XorTest {
//
//    public static void main(String[] args) {
//        Random random = new Random(23);
//        NeuralNetwork network = new NeuralNetwork(0.05, random, 2, 3, 1);
//        double[][] xorInputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
//        double[][] xorExpectedOutputs = {{0.0}, {1.0}, {1.0}, {0.0}};
//
//        System.out.println("Testing forward pass on an untrained network...");
//        System.out.println("----------------------------------------------");
//
//
//        for (double[] input : xorInputs) {
//            List<double[]> allActivations = network.feedForward(input);
//
//            double[] finalOutput = allActivations.get(allActivations.size() - 1);
//
//            System.out.println("Input: " + Arrays.toString(input) + " -> Output: " + Arrays.toString(finalOutput));
//        }
//
//        System.out.println("----------------------------------------------");
//        System.out.println("Test complete.");
//
//        int epochs = 10000;
//        System.out.println("Starting training for " + epochs + " epochs...");
//
//        for (int epoch = 0; epoch < epochs; epoch++) {
//            double totalEpochError = 0;
//
//            for (int i = 0; i < xorInputs.length; i++) {
//                double sampleError = network.train(xorInputs[i], xorExpectedOutputs[i]);
//                totalEpochError += sampleError;
//            }
//
//            if ((epoch + 1) % 1000 == 0) {
//                double averageError = totalEpochError / xorInputs.length;
//                System.out.printf("Epoch %d/%d, Average Error: %.6f\n", (epoch + 1), epochs, averageError);
//            }
//        }
//        System.out.println("Training finished.\n");
//
//        System.out.println("Testing forward pass on an trained network...");
//        System.out.println("----------------------------------------------");
//
//
//        for (double[] input : xorInputs) {
//            List<double[]> allActivations = network.feedForward(input);
//
//            double[] finalOutput = allActivations.get(allActivations.size() - 1);
//
//            System.out.println("Input: " + Arrays.toString(input) + " -> Output: " + Arrays.toString(finalOutput));
//        }
//
//        System.out.println("----------------------------------------------");
//        System.out.println("Test complete.");
//    }
//}