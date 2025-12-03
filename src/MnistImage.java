/**
 * Represents a single image sample from the Fashion-MNIST dataset.
 * <p>
 * This is an Java Record that holds the input features
 * and the expected target label.
 *
 * @param pixels The flattened array of normalized pixel values (0.0 to 1.0).
 *               Expected length is 784 (28x28 image).
 * @param label  The integer class label representing the clothing category (0-9).
 */
public record MnistImage(double[] pixels, int label) {
}