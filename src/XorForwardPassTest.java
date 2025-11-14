//import java.util.Arrays;
//import java.util.List;
//import java.util.Random;
//
//
//public class XorForwardPassTest {
//
//    public static void main(String[] args) {
//        Random random = new Random(23);
//
//        NeuralNetwork network = new NeuralNetwork(0.1, random, 2, 3, 1);
//
//        double[][] xorInputs = {
//                {0.0, 0.0},
//                {0.0, 1.0},
//                {1.0, 0.0},
//                {1.0, 1.0}
//        };
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
//    }
//}