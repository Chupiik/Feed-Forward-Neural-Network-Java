
import java.util.Random;

public class Layer {

    public final double[][] weights;
    public final double[] biases;

    public final double[][] weightVelocities;
    public final double[] biasVelocities;

    public final double[][] m_weights;
    public final double[][] v_weights;
    public final double[] m_biases;
    public final double[] v_biases;

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