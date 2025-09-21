package nika.ml.network;

import nika.ml.mnist.MnistImageReader;
import nika.ml.mnist.Sample;

import java.io.IOException;
import java.util.Arrays;

import static nika.ml.mnist.Sample.*;
import static nika.ml.network.LinAlgOperations.*;

public class NNTraining {

    private NeuralNetwork N;
    private NeuralNetwork bestN;
    private double[][][] WDeltas;
    private double[][] bDeltas;
    private double[][] grad;
    private double[] goal;
    private double totalRuntime;

    public NNTraining(NeuralNetwork neuralNetwork) {
        N = neuralNetwork;
        N.HeNormal();
        bestN = new NeuralNetwork(N.sizes);
        init();
    }

    public NeuralNetwork getNeuralNetwork() {
        return N;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        N = neuralNetwork;
        init();
    }

    private void init() {

        WDeltas = new double[N.LAYERS][][];
        bDeltas = new double[N.LAYERS][];
        grad = new double[N.LAYERS][];
        goal = new double[N.OUTPUT_DIM];

        for (int l = 0; l < N.LAYERS; ++l) {
            WDeltas[l] = new double[N.weights[l].length][N.weights[l][0].length];
            bDeltas[l] = new double[N.biases[l].length];
            grad[l] = new double[N.biases[l].length];
        }
    }

    // gradient of (cross entropy o softmax) with respect to z_l
    void gradCES_z() {
         sub(N.a[N.LAYERS - 1], goal, grad[N.LAYERS - 1]);
    }

    // compute the gradient for W_i and b_i, where i = layer
    void computeDelta(double[] x, int layer) {
        for (int i = 0; i < WDeltas[layer].length; ++i) {
            for (int j = 0; j < WDeltas[layer][i].length; ++j) {
                WDeltas[layer][i][j] = x[i] * grad[layer][j];
            }
        }
        System.arraycopy(grad[layer], 0, bDeltas[layer], 0, bDeltas[layer].length);
    }

    // compute the gradients w.r.t. weights and biases backwards
    void backPropagation() {
        gradCES_z();
        for (int l = N.LAYERS - 1; l > 0; --l) {
            computeDelta(N.a[l - 1], l);
            dot(N.weights[l], grad[l], grad[l - 1]);
            for (int j = 0; j < grad[l - 1].length; j++)
                grad[l - 1][j] *= (N.z[l - 1][j] > 0 ? 1 : 0);
        }
        computeDelta(N.input, 0);
    }

    // sets the goal vector for the specified label
    void setOutput(int label) {
        for (int i = 0; i < goal.length; ++i) {
            goal[i] = (i == label) ? 1 : 0;
        }
    }

    // reset a matrix
    void reset(double[][] a) {
        for (double[] doubles : a) {
            Arrays.fill(doubles, 0);
        }
    }

    // reset a 3-dimensional array
    void reset(double[][][] a) {
        for (double[][] doubles : a) {
            reset(doubles);
        }
    }

    // divide each entry by n
    void average(double[][][] a, int n) {
        for (double[][] doubles : a) {
            average(doubles, n);
        }
    }

    // divide each entry by n
    void average(double[][] a, int n) {
        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[i].length; ++j) {
                a[i][j] /= n;
            }
        }
    }

    // adjust weights and biases according to the average gradient in a mini-batch
    void adjustWb(double[][][] avWDeltas, double[][] avbDeltas, double t) {
        for (int l = 0; l < N.LAYERS; ++l) {
            for (int j = 0; j < N.weights[l][0].length; ++j) {
                for (int i = 0; i < N.weights[l].length; ++i) {
                    N.weights[l][i][j] -= t * avWDeltas[l][i][j];
                }
            }
        }
        for (int l = 0; l < N.LAYERS; ++l) {
            for (int i = 0; i < N.biases[l].length; ++i) {
                N.biases[l][i] -= t * avbDeltas[l][i];
            }
        }
    }

    void train(int[][] images, int[] labels,
               double LEARNING_RATE, int EPOCHS, int MINI_BATCHES, int VALIDATION, int PATIENCE) {

        Sample[] samples = getSamples(images, labels);

        int TRAINING = samples.length - VALIDATION;
        double bestAccuracy = 0;
        int step = 0;

        Sample[] main = new Sample[TRAINING];
        Sample[] validation = new Sample[VALIDATION];

        double[][][] sWDeltas = new double[N.LAYERS][][];
        double[][] sbDeltas = new double[N.LAYERS][];

        for (int l = 0; l < N.LAYERS; ++l) {
            sWDeltas[l] = new double[WDeltas[l].length][N.weights[l][0].length];
            sbDeltas[l] = new double[bDeltas[l].length];
        }

        for (int e = 0; e < EPOCHS; ++e) {

            System.out.println("Starting epoch " + (e + 1) + ".");
            long startTime = System.currentTimeMillis();

            shuffle(samples);
            System.arraycopy(samples, 0, main, 0, TRAINING);
            System.arraycopy(samples, TRAINING, validation, 0, VALIDATION);

            for (int i = 0; i < MINI_BATCHES; ++i) {
                reset(sWDeltas);
                reset(sbDeltas);
                for (int j = 0; j < TRAINING / MINI_BATCHES; ++j) {
                    int index = i * (TRAINING / MINI_BATCHES) + j;
                    N.normalizeInput(main[index].getImage());
                    N.feedForward();
                    setOutput(main[index].getLabel());
                    backPropagation();
                    add(sWDeltas, WDeltas);
                    add(sbDeltas, bDeltas);
                }
                average(sWDeltas, TRAINING / MINI_BATCHES);
                average(sbDeltas, TRAINING / MINI_BATCHES);
                adjustWb(sWDeltas, sbDeltas, LEARNING_RATE);
            }

            long endTime = System.currentTimeMillis();
            double currentAccuracy = N.test(validation);
            step++;

            if (currentAccuracy > bestAccuracy) {
                bestAccuracy = currentAccuracy;
                bestN.copy(N);
                step = 0;
            } else if (step >= PATIENCE){
                System.out.println("Stopping training because of no accuracy improvement.");
                N.copy(bestN);
                break;
            }
            System.out.println("Epoch " + (e + 1) + " finished with the success rate " + currentAccuracy
                    + ".\nRuntime: " + (endTime - startTime) * 1.0 / 1000 + "s\n" + "Step: " + step + "\n");
        }
    }

    public static void main(String[] args) {

        long startTime = System.currentTimeMillis();

        NeuralNetwork net = new NeuralNetwork(784, 128, 64, 32, 16, 10);
        NNTraining training = new NNTraining(net);
        try {
            int[][] images;
            int[] labels;

            images = MnistImageReader.readImages("/train-images.idx3-ubyte");
            labels = MnistImageReader.readLabels("/train-labels.idx1-ubyte");
            training.train(images, labels, 0.01, 50, 1000, 10_000, 10);

            long endTime = System.currentTimeMillis();

            System.out.println("Total runtime: " + (endTime - startTime) * 1.0 / 1000);

            net.save();

            //load("NN-784-6-10.json");

            images = MnistImageReader.readImages("/t10k-images.idx3-ubyte");
            labels = MnistImageReader.readLabels("/t10k-labels.idx1-ubyte");
            System.out.println("Successful: " + net.test(images, labels));

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}