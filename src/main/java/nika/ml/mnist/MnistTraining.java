package nika.ml.mnist;

import nika.ml.network.EvalResults;
import nika.ml.network.Training;
import nika.ml.network.NeuralNetwork;

public class MnistTraining extends Training {

    public MnistTraining(NeuralNetwork neuralNetwork) {
        super(neuralNetwork);
    }

    static double[][] normalizeInput(int[][] images) {
        double[][] res = new double[images.length][images[0].length];
        for (int i = 0; i < images.length; ++i) {
            for (int j = 0; j < images[i].length; ++j) {
                res[i][j] = images[i][j] * 1.0 / 255;
            }
        }
        return res;
    }

    public void train(int[][] images, int[] labels,
                      double LEARNING_RATE, int EPOCHS, int MINI_BATCHES, int VALIDATION, int PATIENCE) {
        super.train(normalizeInput(images), labels, LEARNING_RATE, EPOCHS, MINI_BATCHES, VALIDATION, PATIENCE);
    }

    public EvalResults test(int[][] images, int[] labels) {
        return super.test(normalizeInput(images), labels);
    }
}
