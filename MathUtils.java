import java.util.Random;


public final class MathUtils {

    private MathUtils() {
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));

    }

    public static double sigmoidDerivative(double sigmoidOutput) {
        return sigmoidOutput * (1 - sigmoidOutput);
    }

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

    public static double[][] createRandomMatrix(int rows, int cols, Random random) {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = random.nextDouble() - 0.5;
            }
        }
        return result;
    }

    public static double[] createRandomVector(int size, Random random) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = random.nextDouble() - 0.5;
        }
        return result;
    }
}