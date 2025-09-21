package nika.ml.mnist;

import nika.ml.network.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;

public class DigitViewer {
    public static void showDigits(EvaluationResults results) {
        JPanel container = new JPanel();
        container.setLayout(new BoxLayout(container, BoxLayout.Y_AXIS));

        // Correct row
        container.add(new JLabel("✅ Correct Predictions"));
        JPanel correctRow = new JPanel();
        correctRow.setLayout(new BoxLayout(correctRow, BoxLayout.X_AXIS));
        for (int i = 0; i < results.getCorrect().size(); i++) {
            correctRow.add(new DigitPanel(results.getCorrect().get(i)));
        }
        container.add(correctRow);

        // Incorrect row
        container.add(new JLabel("❌ Incorrect Predictions"));
        JPanel wrongRow = new JPanel();
        wrongRow.setLayout(new BoxLayout(wrongRow, BoxLayout.X_AXIS));
        for (int i = 0; i < results.getWrong().size(); i++) {
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

    public static void main(String[] args) {

//        NeuralNetwork net = new NeuralNetwork(784, 16, 10);
//        NNTraining training = new NNTraining(net);
        try {
            int[][] images;
            int[] labels;

            NeuralNetwork net = NeuralNetwork.load("NN-784-16-10.json");

            net.enableVisualMode();
            images = MnistImageReader.readImages("/t10k-images.idx3-ubyte");
            labels = MnistImageReader.readLabels("/t10k-labels.idx1-ubyte");
            System.out.printf("Successful: %.2f %% \n", net.test(images, labels));

            showDigits(net.getResults());

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}