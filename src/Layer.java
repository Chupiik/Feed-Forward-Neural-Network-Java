
import java.util.Random;

public class Layer {

    public final double[][] weights;
    public final double[] biases;

    public final double[][] weightVelocities;
    public final double[] biasVelocities;

    public Layer(int inputSize, int outputSize, Random random) {
        this.weights = MathUtils.createRandomMatrix(outputSize, inputSize, random, inputSize);
        this.biases = MathUtils.createRandomVector(outputSize, random);

        this.weightVelocities = new double[outputSize][inputSize];
        this.biasVelocities = new double[outputSize];
    }
}