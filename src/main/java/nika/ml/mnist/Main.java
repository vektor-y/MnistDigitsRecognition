package nika.ml.mnist;

import nika.ml.network.EvalResults;
import nika.ml.network.NeuralNetwork;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;


public class Main {
    public static void main(String[] args) {
        try {
            NeuralNetwork net = NeuralNetwork.load("models/NN-784-64-32-10.json");

            String imagesPath = "data/mnist/t10k-images.idx3-ubyte";
            String labelsPath = "data/mnist/t10k-labels.idx1-ubyte";

            if (Files.exists(Paths.get(imagesPath)) && Files.exists(Paths.get(labelsPath))) {
                // Full dataset
                System.out.println("✅ Loaded full dataset from /data.");
            } else {
                // Small demo
                imagesPath = "src/main/resources/demo/demo-images.idx3-ubyte";
                labelsPath = "src/main/resources/demo/demo-labels.idx1-ubyte";
                System.out.println("⚠️ Full dataset not found. Using tiny demo dataset.");
            }

            int[][] images = MnistImageReader.readImages(imagesPath);
            int [] labels = MnistImageReader.readLabels(labelsPath);
            EvalResults results = net.test(MnistTraining.normalizeInput(images), labels);
            System.out.println(results);
            DigitViewer.showDigits(results, 10);

        } catch (IOException e) {
            System.err.println("There was a problem finding files.");
        }
    }

}
