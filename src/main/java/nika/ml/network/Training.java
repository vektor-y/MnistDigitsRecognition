package nika.ml.network;

import static nika.ml.network.LinAlgOperations.*;
import static nika.ml.network.Sample.getSamples;
import static nika.ml.network.Sample.shuffle;

public class Training {

    private NeuralNetwork N;
    private NeuralNetwork bestN;
    private double[][][] WDeltas;
    private double[][] bDeltas;
    private double[][] grad;
    private double[] goal;

    public Training(NeuralNetwork neuralNetwork) {
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
    private void gradCES_z() {
         sub(N.a[N.LAYERS - 1], goal, grad[N.LAYERS - 1]);
    }

    // compute the gradient for W_i and b_i, where i = layer
    private void computeDelta(double[] x, int layer) {
        for (int i = 0; i < WDeltas[layer].length; ++i) {
            for (int j = 0; j < WDeltas[layer][i].length; ++j) {
                WDeltas[layer][i][j] = x[i] * grad[layer][j];
            }
        }
        System.arraycopy(grad[layer], 0, bDeltas[layer], 0, bDeltas[layer].length);
    }

    // compute the gradients w.r.t. weights and biases backwards
    private void backPropagation() {
        gradCES_z();
        for (int l = N.LAYERS - 1; l > 0; --l) {
            computeDelta(N.a[l - 1], l);
            dot(N.weights[l], grad[l], grad[l - 1]);
            for (int j = 0; j < grad[l - 1].length; j++)
                grad[l - 1][j] *= (N.z[l - 1][j] > 0 ? 1 : 0);
        }
        computeDelta(N.input, 0);
    }

    // sets the goal vector for the specified label by creating a unit vector with v[label] = 1
    protected void setGoalOutput(int label) {
        for (int i = 0; i < goal.length; ++i) {
            goal[i] = (i == label) ? 1 : 0;
        }
    }

    // adjust weights and biases according to the average gradient in a mini-batch
    private void adjustWb(double[][][] avWDeltas, double[][] avbDeltas, double t) {
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

    public void train(double[][] inputs, int[] labels,
               double LEARNING_RATE, int EPOCHS, int MINI_BATCHES, int VALIDATION, int PATIENCE) {

        Sample[] samples = getSamples(inputs, labels);

        int TRAINING = samples.length - VALIDATION;
        double bestAccuracy = 0;
        int step = 0;

        Sample[] main = new Sample[TRAINING];
        Sample[] validation = new Sample[VALIDATION];
        System.arraycopy(samples, 0, main, 0, TRAINING);
        System.arraycopy(samples, TRAINING, validation, 0, VALIDATION);

        double[][][] sWDeltas = new double[N.LAYERS][][];
        double[][] sbDeltas = new double[N.LAYERS][];

        for (int l = 0; l < N.LAYERS; ++l) {
            sWDeltas[l] = new double[WDeltas[l].length][N.weights[l][0].length];
            sbDeltas[l] = new double[bDeltas[l].length];
        }

        for (int e = 0; e < EPOCHS; ++e) {

            System.out.println("Starting epoch " + (e + 1) + ".");
            long startTime = System.currentTimeMillis();

            shuffle(main);

            for (int i = 0; i < MINI_BATCHES; ++i) {
                reset(sWDeltas);
                reset(sbDeltas);
                for (int j = 0; j < TRAINING / MINI_BATCHES; ++j) {
                    int index = i * (TRAINING / MINI_BATCHES) + j;
                    N.setInput(main[index].getInput());
                    N.feedForward();
                    setGoalOutput(main[index].getLabel());
                    backPropagation();
                    add(sWDeltas, WDeltas);
                    add(sbDeltas, bDeltas);
                }
                average(sWDeltas, TRAINING / MINI_BATCHES);
                average(sbDeltas, TRAINING / MINI_BATCHES);
                adjustWb(sWDeltas, sbDeltas, LEARNING_RATE);
            }

            long endTime = System.currentTimeMillis();
            EvalResults results = N.test(validation);
            System.out.println(results);
            double currentAccuracy = results.getAccuracy();
            step++;

            if (currentAccuracy > bestAccuracy) {
                bestAccuracy = currentAccuracy;
                bestN.copy(N); // saves best validation accuracy model
                step = 0;
            } else if (step >= PATIENCE){
                System.out.println("Stopping training because of no accuracy improvement.");
                N.copy(bestN);
                break;
            }
            System.out.println("Runtime: " + (endTime - startTime) * 1.0 / 1000 + "s\n" + "Step: " + step + "\n");
        }
    }

    public EvalResults test(double[][] inputs, int[] labels) {
        return N.test(getSamples(inputs, labels));
    }
}