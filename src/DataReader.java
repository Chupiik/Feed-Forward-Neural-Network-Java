import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;


/**
 * Utility class responsible for loading and parsing the Fashion-MNIST dataset
 * from CSV files.
 */
public class DataReader {

    /**
     * Loads image data and labels from the specified CSV files.
     * <p>
     * It reads both files simultaneously line-by-line. Pixel values are parsed
     * from the vector file and normalized from the range [0, 255] to [0.0, 1.0].
     *
     * @param vectorsPath The file path to the CSV containing the image pixel vectors.
     * @param labelsPath  The file path to the CSV containing the corresponding labels.
     * @return A list of MnistImage objects containing the normalized pixels and labels.
     */
    public static List<MnistImage> loadData(String vectorsPath, String labelsPath) {
        List<MnistImage> images = new ArrayList<>();

        try (BufferedReader brVectors = new BufferedReader(new FileReader(vectorsPath));
             BufferedReader brLabels = new BufferedReader(new FileReader(labelsPath))) {


            String vectorLine;
            String labelLine;

            while ((vectorLine = brVectors.readLine()) != null && (labelLine = brLabels.readLine()) != null) {
                int label = Integer.parseInt(labelLine);

                String[] pixelStrings = vectorLine.split(",");
                double[] pixels = new double[784];
                for (int i = 0; i < 784; i++) {
                    pixels[i] = Integer.parseInt(pixelStrings[i]) / 255.0;
                }

                images.add(new MnistImage(pixels, label));
            }

        } catch (IOException e) {
            System.err.println("Error reading data files");
            e.printStackTrace();
        } catch (NumberFormatException e) {
            System.err.println("Error parsing a number from the data files");
            e.printStackTrace();
        }

        return images;
    }


    /**
     * A simple test method to verify that data loading works correctly.
     * Prints statistics and details about the first loaded image to the console.
     *
     * @param args Command line arguments (not used).
     */
    public static void main(String[] args) {
        String trainVectorsPath = "data/fashion_mnist_train_vectors.csv";
        String trainLabelsPath = "data/fashion_mnist_train_labels.csv";

        System.out.println("Attempting to load data:");
        System.out.println("Vectors: " + trainVectorsPath);
        System.out.println("Labels:  " + trainLabelsPath);

        List<MnistImage> trainingImages = loadData(trainVectorsPath, trainLabelsPath);

        if (!trainingImages.isEmpty()) {
            System.out.println("\nSuccessfully loaded " + trainingImages.size() + " images.");

            MnistImage firstImage = trainingImages.get(0);
            System.out.println("Details of the first image:");
            System.out.println(" - Label: " + firstImage.label());
            System.out.println(" - Number of pixels: " + firstImage.pixels().length);
            System.out.println(" - First 10 pixels (normalized): " +
                    Arrays.toString(Arrays.copyOfRange(firstImage.pixels(), 0, 10)));
        } else {
            System.out.println("\nFailed to load any images");
        }
    }
}