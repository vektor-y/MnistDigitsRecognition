package nika.ml.mnist;

import nika.ml.network.EvalResults;
import nika.ml.network.NeuralNetwork;
import static nika.ml.mnist.MnistTraining.normalizeInput;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import static java.lang.Math.min;

public class DigitViewer {
    public static void showDigits(EvalResults results, int size) {
        JPanel container = new JPanel();
        container.setLayout(new BoxLayout(container, BoxLayout.Y_AXIS));

        // Correct row
        container.add(new JLabel("✅ Correct Predictions"));
        JPanel correctRow = new JPanel();
        correctRow.setLayout(new BoxLayout(correctRow, BoxLayout.X_AXIS));
        for (int i = 0; i < min(size, results.getCorrect().size()); i++) {
            correctRow.add(new DigitPanel(results.getCorrect().get(i)));
        }
        container.add(correctRow);

        // Incorrect row
        container.add(new JLabel("❌ Incorrect Predictions"));
        JPanel wrongRow = new JPanel();
        wrongRow.setLayout(new BoxLayout(wrongRow, BoxLayout.X_AXIS));
        for (int i = 0; i < min(size, results.getWrong().size()); i++) {
            wrongRow.add(new DigitPanel(results.getWrong().get(i)));
        }
        container.add(wrongRow);

        JScrollPane scrollPane = new JScrollPane(container);
        scrollPane.setPreferredSize(new Dimension(1200, 450));

        JFrame frame = new JFrame("Digit Classification Results");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(scrollPane);
        frame.pack();
        frame.setVisible(true);
    }

    public static void showDigits(EvalResults results) {
        showDigits(results, 10);
    }

    public static void main(String[] args) {
        try {
            int[][] images;
            int[] labels;

            NeuralNetwork net = NeuralNetwork.load("src/main/resources/NN-784-64-32-10.json");

            images = MnistImageReader.readImages("/data/mnist/t10k-images.idx3-ubyte");
            labels = MnistImageReader.readLabels("/data/mnist/t10k-labels.idx1-ubyte");

            showDigits(net.test(normalizeInput(images), labels), 10);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}