import java.util.Random;


/**
 * Static utility class containing mathematical functions for neural networks.
 * <p>
 * Includes activation functions (Sigmoid, ReLU, Leaky ReLU, Softmax), their derivatives,
 * and linear algebra operations (Matrix-Vector multiplication, Vector addition/subtraction).
 */
public final class MathUtils {

    /**
     * Private constructor to prevent instantiation of this utility class.
     */
    private MathUtils() {
    }

    /**
     * Calculates the Sigmoid activation function.
     * f(x) = 1 / (1 + exp(-x))
     *
     * @param x The input value.
     * @return The output between 0.0 and 1.0.
     */
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));

    }

    /**
     * Calculates the derivative of the Sigmoid function.
     * Note: This method expects the *output* of the sigmoid function, not the raw input.
     * f'(y) = y * (1 - y)
     *
     * @param sigmoidOutput The value already processed by the sigmoid function.
     * @return The gradient value.
     */
    public static double sigmoidDerivative(double sigmoidOutput) {
        return sigmoidOutput * (1 - sigmoidOutput);
    }

    /**
     * Calculates the ReLU (Rectified Linear Unit) activation function.
     * f(x) = max(0, x)
     *
     * @param x The input value.
     * @return x if positive, 0 otherwise.
     */
    public static double relu(double x) {
        return Math.max(0, x);
    }

    /**
     * Calculates the derivative of the ReLU function.
     *
     * @param reluOutput The output of the ReLU function (or the input, the sign is the same).
     * @return 1 if positive, 0 otherwise.
     */
    public static double reluDerivative(double reluOutput) {
        return reluOutput > 0 ? 1 : 0;
    }

    /**
     * Calculates the Leaky ReLU activation function.
     * Allows a small gradient when the unit is not active to prevent "dying ReLU".
     * f(x) = x if x > 0, else 0.01 * x
     *
     * @param x The input value.
     * @return x if positive, scaled x otherwise.
     */
    public static double leakyRelu(double x) {
        return x > 0 ? x : 0.01 * x;
    }

    /**
     * Calculates the derivative of the Leaky ReLU function.
     *
     * @param leakyReluOutput The value being differentiated.
     * @return 1 if positive, 0.01 otherwise.
     */
    public static double leakyReluDerivative(double leakyReluOutput) {
        return leakyReluOutput > 0 ? 1 : 0.01;
    }

    /**
     * Calculates the Softmax probability distribution for a vector of raw logits.
     * <p>
     * Implements numerical stability by subtracting the maximum logit value
     * from all logits before exponentiation to prevent overflow.
     *
     * @param logits The raw output values from the final layer.
     * @return An array of probabilities summing to 1.0.
     */
    public static double[] softmax(double[] logits) {
        double[] probabilities = new double[logits.length];

        double maxLogit = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        double sumExponentials = 0.0;
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = Math.exp(logits[i] - maxLogit);
            sumExponentials += probabilities[i];
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sumExponentials;
        }

        return probabilities;
    }


    /**
     * Performs Matrix-Vector multiplication.
     * Result = Matrix * Vector
     *
     * @param matrix The matrix [rows][cols].
     * @param vector The vector [cols].
     * @return A new vector of size [rows].
     * @throws IllegalArgumentException if matrix columns do not match vector length.
     */
    public static double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;

        if (rows == 0) {
            return new double[0];
        }

        int cols = matrix[0].length;

        if (cols != vector.length) {
            throw new IllegalArgumentException("Matrix columns (" + cols + ") must match vector length (" + vector.length + ").");
        }

        double[] result = new double[rows];

        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }

        return result;
    }


    /**
     * Adds two vectors element-wise.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @return A new vector containing a[i] + b[i].
     * @throws IllegalArgumentException if vectors have different lengths.
     */
    public static double[] addVectors(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length.");
        }

        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }


    /**
     * Subtracts vector b from vector a element-wise.
     *
     * @param a The vector to subtract from.
     * @param b The vector to subtract.
     * @return A new vector containing a[i] - b[i].
     * @throws IllegalArgumentException if vectors have different lengths.
     */
    public static double[] subtractVectors(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length.");
        }

        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    /**
     * Multiplies two vectors element-wise (Hadamard product).
     *
     * @param a The first vector.
     * @param b The second vector.
     * @return A new vector containing a[i] * b[i].
     * @throws IllegalArgumentException if vectors have different lengths.
     */
    public static double[] elementMultVectors(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have the same length.");
        }
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }


    /**
     * Transposes a given matrix.
     * Swaps rows and columns.
     *
     * @param matrix The input matrix [rows][cols].
     * @return A new transposed matrix [cols][rows].
     */
    public static double[][] transposeMatrix(double[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new double[0][0];
        }

        int rows = matrix.length;
        int cols = matrix[0].length;

        double[][] result = new double[cols][rows];


        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }

        return result;
    }

//    public static double[][] createRandomMatrix(int rows, int cols, Random random, int inputSize) {
//        double[][] result = new double[rows][cols];
//
//        double stddev = Math.sqrt(2.0 / inputSize);
//
//        for (int i = 0; i < rows; i++) {
//            for (int j = 0; j < cols; j++) {
//                result[i][j] = random.nextGaussian() * stddev;
//            }
//        }
//        return result;
//    }

    /**
     * Creates a matrix initialized with random values using He Initialization.
     * Suitable for layers using ReLU variants.
     * StdDev = sqrt(2.0 / inputSize).
     *
     * @param rows      Number of rows (output size).
     * @param cols      Number of columns (input size).
     * @param random    Random instance.
     * @param inputSize The number of input connections (fan-in) used for scaling.
     * @return A matrix initialized with He-distributed random values.
     */
    public static double[][] createRandomMatrix(int rows, int cols, Random random, int inputSize) {
        double[][] result = new double[rows][cols];
        double stddev = Math.sqrt(2.0 / inputSize);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = random.nextGaussian() * stddev;
            }
        }
        return result;
    }

    /**
     * Creates a vector initialized with a small positive constant.
     * Used for bias initialization to prevent dead neurons at start.
     *
     * @param size   The size of the vector.
     * @param random Random instance (unused in current implementation, but kept for interface consistency).
     * @return A vector filled with 0.1.
     */
    public static double[] createRandomVector(int size, Random random) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            //result[i] = random.nextDouble() - 0.5;
            result[i] = 0.1;
        }
        return result;
    }

}