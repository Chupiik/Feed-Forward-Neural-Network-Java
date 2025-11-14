import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.LinkedList;

public class NeuralNetwork {

    private final List<Layer> layers;
    private final double learningRate;
    private final double momentum;

    public NeuralNetwork(double learningRate, double momentum, Random random, int... sizes) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.momentum = momentum;

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
        for (int i = 0; i < this.layers.size(); i++) {
            Layer layer = this.layers.get(i);
            double[] weightedSum = MathUtils.matrixVectorMultiply(layer.weights, currentActivations);
            double[] biasedSum = MathUtils.addVectors(weightedSum, layer.biases);
            if (i == this.layers.size() - 1) {
                currentActivations = MathUtils.softmax(biasedSum);
            } else {
                double[] newActivations = new double[biasedSum.length];
                for (int j = 0; j < newActivations.length; j++) {
                    newActivations[j] = MathUtils.relu(biasedSum[j]);
                }
                currentActivations = newActivations;
            }
            allActivations.add(currentActivations);
        }
        return allActivations;
    }

    public double train(double[] input, double[] expectedOutput) {
        List<double[]> allActivations = feedForward(input);
        double[] finalOutput = allActivations.get(allActivations.size() - 1);

        double sampleError = 0.0;
        for (int i = 0; i < expectedOutput.length; i++) {
            double diff = expectedOutput[i] - finalOutput[i];
            sampleError += diff * diff;
        }

        LinkedList<double[]> deltas = new LinkedList<>();
        double[] outputDelta = MathUtils.subtractVectors(finalOutput, expectedOutput);
        deltas.addFirst(outputDelta);
        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer frontLayer = layers.get(i + 1);
            double[][] transposedWeights = MathUtils.transposeMatrix(frontLayer.weights);
            double[] propagatedError = MathUtils.matrixVectorMultiply(transposedWeights, deltas.getFirst());
            double[] currentActivations = allActivations.get(i + 1);
            double[] hiddenDerivatives = new double[currentActivations.length];
            for (int j = 0; j < currentActivations.length; j++) {
                hiddenDerivatives[j] = MathUtils.reluDerivative(currentActivations[j]);
            }
            double[] currentDelta = MathUtils.elementMultVectors(propagatedError, hiddenDerivatives);
            deltas.addFirst(currentDelta);
        }

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            double[] previousActivations = allActivations.get(i);
            double[] currentLayerDelta = deltas.get(i);

            for (int j = 0; j < layer.biases.length; j++) {
                double velocity = (layer.biasVelocities[j] * momentum) - (learningRate * currentLayerDelta[j]);
                layer.biases[j] += velocity;
                layer.biasVelocities[j] = velocity;
            }

            for (int j = 0; j < layer.weights.length; j++) {
                for (int k = 0; k < layer.weights[j].length; k++) {
                    double velocity = (layer.weightVelocities[j][k] * momentum) - (learningRate * currentLayerDelta[j] * previousActivations[k]);
                    layer.weights[j][k] += velocity;
                    layer.weightVelocities[j][k] = velocity;
                }
            }
        }
        return sampleError;
    }
}