import java.util.Random;


public class Layer {

    public final double[][] weights;

    public final double[] biases;

    public Layer(int inputSize, int outputSize, Random random) {
        this.weights = MathUtils.createRandomMatrix(outputSize, inputSize, random);
        this.biases = MathUtils.createRandomVector(outputSize, random);
    }
}