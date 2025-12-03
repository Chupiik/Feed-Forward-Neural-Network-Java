import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * Represents a fully connected Feed-Forward Neural Network.
 * <p>
 * This class manages the network topology (layers), performs forward propagation,
 * and implements training algorithms including standard SGD with Momentum and the
 * Adam optimizer. It handles backpropagation and weight updates manually.
 */
public class NeuralNetwork {

    private final List<Layer> layers;
    private final double learningRate;


    private final double momentum;
    private final double lambda;

    //ADAM
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private int t = 0;

    private double beta1_t = 1.0;
    private double beta2_t = 1.0;


    /**
     * Constructs a new Neural Network with the specified hyperparameters and topology.
     *
     * @param learningRate The step size for weight updates (alpha).
     * @param momentum     The momentum factor for SGD (gamma).
     * @param lambda       The L2 regularization strength (weight decay).
     * @param random       The random number generator for weight initialization.
     * @param sizes        A variable argument list defining the number of neurons in each layer
     *                     (e.g., 784, 128, 10 means input 784, one hidden layer of 128, output 10).
     */
    public NeuralNetwork(double learningRate, double momentum, double lambda, Random random, int... sizes) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.lambda = lambda;



        for (int i = 0; i < sizes.length - 1; i++) {
            int inputSize = sizes[i];
            int outputSize = sizes[i + 1];
            this.layers.add(new Layer(inputSize, outputSize, random));
        }
    }

    /**
     * Performs the forward pass through the network.
     * <p>
     * Computes the activations for every layer.
     * Uses Leaky ReLU for hidden layers and Softmax for the output layer.
     *
     * @param input The input vector.
     * @return A list of activation vectors for every layer (including the input layer).
     */
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
                    newActivations[j] = MathUtils.leakyRelu(biasedSum[j]);
                }
                currentActivations = newActivations;
            }
            allActivations.add(currentActivations);
        }
        return allActivations;
    }

    /**
     * Trains the network on a single sample using SGD with Momentum.
     * <p>
     * Performs a forward pass, calculates gradients via backpropagation,
     * and updates weights and biases using the standard momentum update rule.
     *
     * @param input          The input vector.
     * @param expectedOutput The target (ground truth) vector.
     * @return The mean squared error (MSE) for this sample.
     */
    public double train(double[] input, double[] expectedOutput) {
        List<double[]> allActivations = feedForward(input);
        double[] finalOutput = allActivations.get(allActivations.size() - 1);

        double sampleError = 0.0;
        for (int i = 0; i < expectedOutput.length; i++) {
            sampleError += (expectedOutput[i] - finalOutput[i]) * (expectedOutput[i] - finalOutput[i]);
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
                hiddenDerivatives[j] = MathUtils.leakyReluDerivative(currentActivations[j]);
            }

            double[] currentDelta = MathUtils.elementMultVectors(propagatedError, hiddenDerivatives);
            deltas.addFirst(currentDelta);
        }

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            double[] previousActivations = allActivations.get(i);
            double[] currentLayerDelta = deltas.get(i);

            for (int j = 0; j < layer.biases.length; j++) {
                double gradient = currentLayerDelta[j];
                double velocity = (layer.biasVelocities[j] * momentum) - (learningRate * gradient);
                layer.biases[j] += velocity;
                layer.biasVelocities[j] = velocity;
            }

            for (int j = 0; j < layer.weights.length; j++) {
                for (int k = 0; k < layer.weights[j].length; k++) {
                    double gradient = currentLayerDelta[j] * previousActivations[k];
                    gradient += lambda * layer.weights[j][k];
                    double velocity = (layer.weightVelocities[j][k] * momentum) - (learningRate * gradient);
                    layer.weights[j][k] += velocity;
                    layer.weightVelocities[j][k] = velocity;
                }
            }
        }
        return sampleError;
    }


    /**
     * Trains the network on a single sample using the Adam optimizer.
     * <p>
     * Implements Adaptive Moment Estimation (Adam). It maintains moving averages
     * of the gradients (m) and squared gradients (v), applies bias correction,
     * and updates parameters accordingly.
     *
     * @param input          The input vector.
     * @param expectedOutput The target (ground truth) vector.
     * @return The mean squared error (MSE) for this sample.
     */
    public double trainADAM(double[] input, double[] expectedOutput) {
        List<double[]> allActivations = feedForward(input);
        double[] finalOutput = allActivations.get(allActivations.size() - 1);

        double sampleError = 0.0;
        for (int i = 0; i < expectedOutput.length; i++) {
            sampleError += (expectedOutput[i] - finalOutput[i]) * (expectedOutput[i] - finalOutput[i]);
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
                hiddenDerivatives[j] = MathUtils.leakyReluDerivative(currentActivations[j]);
            }

            double[] currentDelta = MathUtils.elementMultVectors(propagatedError, hiddenDerivatives);
            deltas.addFirst(currentDelta);
        }

        beta1_t *= beta1;
        beta2_t *= beta2;

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            double[] previousActivations = allActivations.get(i);
            double[] currentLayerDelta = deltas.get(i);

            for (int j = 0; j < layer.biases.length; j++) {
                double gradient = currentLayerDelta[j];

                layer.m_biases[j] = beta1 * layer.m_biases[j] + (1 - beta1) * gradient;
                layer.v_biases[j] = beta2 * layer.v_biases[j] + (1 - beta2) * (gradient * gradient);

                double m_hat = layer.m_biases[j] / (1 - beta1_t);
                double v_hat = layer.v_biases[j] / (1 - beta2_t);

                layer.biases[j] -= learningRate * m_hat / (Math.sqrt(v_hat) + epsilon);
            }

            for (int j = 0; j < layer.weights.length; j++) {
                for (int k = 0; k < layer.weights[j].length; k++) {
                    double gradient = currentLayerDelta[j] * previousActivations[k];
                    gradient += lambda * layer.weights[j][k];

                    layer.m_weights[j][k] = beta1 * layer.m_weights[j][k] + (1 - beta1) * gradient;
                    layer.v_weights[j][k] = beta2 * layer.v_weights[j][k] + (1 - beta2) * (gradient * gradient);

                    double m_hat = layer.m_weights[j][k] / (1 - beta1_t);
                    double v_hat = layer.v_weights[j][k] / (1 - beta2_t);

                    layer.weights[j][k] -= learningRate * m_hat / (Math.sqrt(v_hat) + epsilon);
                }
            }
        }
        return sampleError;
    }



    /**
     * Predicts the class label for a given input.
     * <p>
     * Runs the forward pass and finds the index of the neuron with the highest activation.
     *
     * @param input The input vector.
     * @return The predicted class label (index of the maximum output).
     */
    public int predict(double[] input) {
        List<double[]> allActivations = feedForward(input);
        double[] finalOutput = allActivations.get(allActivations.size() - 1);
        int maxIndex = 0;
        for (int i = 1; i < finalOutput.length; i++) {
            if (finalOutput[i] > finalOutput[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}