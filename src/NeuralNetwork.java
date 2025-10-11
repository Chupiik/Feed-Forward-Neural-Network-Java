import java.util.Random;
import java.util.ArrayList;
import java.util.List;


public class NeuralNetwork {

    private final List<Layer> layers;
    private final double learningRate;


    public NeuralNetwork(double learningRate, Random random, int... sizes) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;

        for (int i = 0; i < sizes.length - 1; i++) {
            int inputSize = sizes[i];
            int outputSize = sizes[i + 1];
            this.layers.add(new Layer(inputSize, outputSize, random));
        }
    }


    public List<double[]> feedForward(double[] input) {
        List<double[]> allActivations = new ArrayList<>();
        allActivations.add(input);

        double[] currentActivations = input;

        for (Layer layer : this.layers) {
            double[] weightedSum = MathUtils.matrixVectorMultiply(layer.weights, currentActivations);
            double[] biasedSum = MathUtils.addVectors(weightedSum, layer.biases);

            double[] newActivations = new double[biasedSum.length];
            for (int i = 0; i < newActivations.length; i++) {
                newActivations[i] = MathUtils.sigmoid(biasedSum[i]);
            }

            currentActivations = newActivations;
            allActivations.add(currentActivations);
        }

        return allActivations;
    }

}