import java.util.Random;


public class Layer {

    /**
     * The matrix of connection weights.
     * Dimensions are [outputSize x inputSize].
     * Each row corresponds to a neuron in this layer, and each column corresponds
     * to a neuron in the previous layer.
     */
    public final double[][] weights;

    public final double[] biases;

    public Layer(int inputSize, int outputSize, Random random) {
        this.weights = MathUtils.createRandomMatrix(outputSize, inputSize, random);
        this.biases = MathUtils.createRandomVector(outputSize, random);
    }
}