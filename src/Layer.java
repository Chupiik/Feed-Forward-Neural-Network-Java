import java.util.Random;

/**
 * Represents a single fully connected (dense) layer in the neural network.
 * <p>
 * This class acts as a container for the layer's trainable parameters (weights and biases)
 * and the specific state variables required for optimizers like SGD with Momentum
 * and Adam.
 */
public class Layer {

    /**
     * The weight matrix connecting the previous layer to this layer.
     * Dimensions: [outputSize][inputSize].
     */
    public final double[][] weights;

    /**
     * The bias vector for this layer's neurons.
     * Dimensions: [outputSize].
     */
    public final double[] biases;

    /**
     * Velocity accumulators for weights, used specifically for SGD with Momentum.
     */
    public final double[][] weightVelocities;

    /**
     * Velocity accumulators for biases, used specifically for SGD with Momentum.
     */
    public final double[] biasVelocities;

    /**
     * First moment estimates (m) for weights, used by the Adam optimizer.
     * Represents the running average of gradients.
     */
    public final double[][] m_weights;

    /**
     * Second moment estimates (v) for weights, used by the Adam optimizer.
     * Represents the running average of squared gradients.
     */
    public final double[][] v_weights;

    /**
     * First moment estimates (m) for biases, used by the Adam optimizer.
     */
    public final double[] m_biases;

    /**
     * Second moment estimates (v) for biases, used by the Adam optimizer.
     */
    public final double[] v_biases;

    /**
     * Constructs a new Layer with randomly initialized weights and small positive biases.
     * Initializes all optimizer velocity/moment arrays to zero.
     *
     * @param inputSize  The number of neurons in the previous layer (or input vector size).
     * @param outputSize The number of neurons in this layer.
     * @param random     The random number generator used for initialization (e.g., He Initialization).
     */
    public Layer(int inputSize, int outputSize, Random random) {
        this.weights = MathUtils.createRandomMatrix(outputSize, inputSize, random, inputSize);
        this.biases = MathUtils.createRandomVector(outputSize, random);

        this.weightVelocities = new double[outputSize][inputSize];
        this.biasVelocities = new double[outputSize];

        this.m_weights = new double[outputSize][inputSize];
        this.v_weights = new double[outputSize][inputSize];
        this.m_biases = new double[outputSize];
        this.v_biases = new double[outputSize];
    }
}