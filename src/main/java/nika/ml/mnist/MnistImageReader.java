package nika.ml.mnist;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MnistImageReader {

    public static int[][] readImages(String imageFilePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(
                        MnistImageReader.class.getResourceAsStream(imageFilePath)))) {

            int magicNumber = dis.readInt();
            int numberOfImages = dis.readInt();
            int numRows = dis.readInt();
            int numCols = dis.readInt();

//            System.out.println("Magic Number: " + magicNumber);
            System.out.println("Number of Images: " + numberOfImages);
            System.out.println("Image Dimensions: " + numRows + "x" + numCols);

            int[][] images = new int[numberOfImages][784];
            for (int i = 0; i < numberOfImages; i++) {
                for (int j = 0; j < numRows * numCols; j++) {
                    images[i][j] = dis.readUnsignedByte(); // Read unsigned byte
                }
            }
            return images;
        }
    }

    public static int[] readLabels(String imageFilePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(
                        MnistImageReader.class.getResourceAsStream(imageFilePath)))) {

            int magicNumber = dis.readInt();
            int numberOfLabels = dis.readInt();

//            System.out.println("Magic Number: " + magicNumber);
            System.out.println("Number of Labels: " + numberOfLabels);

            int[] labels = new int[numberOfLabels];
            for (int i = 0; i < numberOfLabels; i++) {
                labels[i] = dis.readUnsignedByte();
            }
            return labels;
        }
    }
}
