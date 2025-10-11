import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.LinkedList;


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
                if (layer == this.layers.getLast()) {
                    newActivations[i] = MathUtils.sigmoid(biasedSum[i]);
                } else {
                    newActivations[i] = MathUtils.relu(biasedSum[i]);
                }
            }

            currentActivations = newActivations;
            allActivations.add(currentActivations);
        }

        return allActivations;
    }


    public double train(double[] input, double[] expectedOutput) {
        List<double[]> allActivations = feedForward(input);

        LinkedList<double[]> deltas = new LinkedList<>();

        int lastLayerIndex = layers.size() - 1;
        double[] finalOutput = allActivations.get(lastLayerIndex + 1);
        double[] error = MathUtils.subtractVectors(expectedOutput, finalOutput);

        double sampleError = 0.0;
        for (int i = 0; i < expectedOutput.length; i++) {
            double diff = expectedOutput[i] - finalOutput[i];
            sampleError += diff * diff;
        }



        double[] outputDerivatives = new double[finalOutput.length];
        for (int i = 0; i < finalOutput.length; i++) {
            outputDerivatives[i] = MathUtils.sigmoidDerivative(finalOutput[i]);
        }

        double[] currentDelta = MathUtils.elementMultVectors(error, outputDerivatives);
        deltas.addFirst(currentDelta);

        for (int i = lastLayerIndex - 1; i >= 0; i--) {
            Layer frontLayer = layers.get(i + 1);
            double[][] transposedWeights = MathUtils.transposeMatrix(frontLayer.weights);
            double[] propagatedError = MathUtils.matrixVectorMultiply(transposedWeights, deltas.getFirst());

            double[] currentActivations = allActivations.get(i + 1);
            double[] hiddenDerivatives = new double[currentActivations.length];
            for (int j = 0; j < currentActivations.length; j++) {
                hiddenDerivatives[j] = MathUtils.reluDerivative(currentActivations[j]);
            }

            currentDelta = MathUtils.elementMultVectors(propagatedError, hiddenDerivatives);
            deltas.addFirst(currentDelta);
        }

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            double[] previousActivations = allActivations.get(i);
            double[] currentLayerDelta = deltas.get(i);

            for (int j = 0; j < layer.biases.length; j++) {
                layer.biases[j] += learningRate * currentLayerDelta[j];
            }


            for (int j = 0; j < layer.weights.length; j++) {
                for (int k = 0; k < layer.weights[j].length; k++) {
                    layer.weights[j][k] += learningRate * currentLayerDelta[j] * previousActivations[k];
                }
            }
        }

        return sampleError;
    }

}