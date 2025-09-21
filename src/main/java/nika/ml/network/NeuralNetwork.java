package nika.ml.network;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import nika.ml.mnist.EvaluationResults;
import nika.ml.mnist.Sample;

import java.io.*;
import java.util.Random;

import static java.lang.Math.*;
import static nika.ml.mnist.Sample.*;

public class NeuralNetwork {

    int LAYERS;
    int INPUT_DIM;
    int OUTPUT_DIM;
    @JsonProperty
    int[] sizes;

    @JsonProperty
    double[][][] weights;
    @JsonProperty
    double[][] biases;
    double[][] z;
    double[][] a;
    double[] input;

    private static final ObjectMapper mapper = new ObjectMapper();

    boolean visualMode;
    @JsonIgnore
    EvaluationResults results;

    public NeuralNetwork() {

    }

    public NeuralNetwork(int ... sizes) {

        this.sizes = sizes;
        LAYERS = sizes.length - 1;

        weights = new double[LAYERS][][];
        biases = new double[LAYERS][];

        for (int l = 0; l < LAYERS; ++l) {
            weights[l] = new double[sizes[l]][sizes[l + 1]];
            biases[l] = new double[sizes[l + 1]];
        }

        init(sizes);
    }

    private void init(int ... sizes) {

        INPUT_DIM = sizes[0];
        OUTPUT_DIM = sizes[sizes.length - 1];
        z = new double[LAYERS][];
        a = new double[LAYERS][];
        input = new double[INPUT_DIM];
        visualMode = false;
        results = new EvaluationResults(10);

        for (int l = 0; l < LAYERS; ++l) {
            z[l] = new double[sizes[l + 1]];
            a[l] = new double[sizes[l + 1]];
        }
    }

    void HeNormal() {
        Random rand = new Random();
        double in = INPUT_DIM;

        for (int l = 0; l < LAYERS; ++l) {
            if (l > 0) {
                in = z[l - 1].length;
            }
            for (int i = 0; i < weights[l].length; ++i) {
                for (int j = 0; j < weights[l][i].length; ++j) {
                    weights[l][i][j] = rand.nextGaussian() * (Math.sqrt(2.0 / in));
                }
            }
        }
    }

    public void setVisualSize(int visualSize) {
        results.setVisualSize(visualSize);
    }

    public void enableVisualMode() {
        visualMode = true;
    }

    public void disableVisualMode() {
        visualMode = false;
    }

    public EvaluationResults getResults() {
        return results;
    }

    public void save(String filePath) throws IOException {
        mapper.writeValue(new File(filePath), this);
    }

    public void save() throws IOException {

        StringBuilder filename = new StringBuilder("NN");
        for (int layer : sizes) {
            filename.append("-").append(layer);
        }
        filename.append(".json");
        save(filename.toString());
    }

    public static NeuralNetwork load(String filePath) throws IOException {

        NeuralNetwork loaded = mapper.readValue(new File(filePath), NeuralNetwork.class);
        loaded.LAYERS = loaded.sizes.length - 1;
        loaded.init(loaded.sizes);
        return loaded;
    }

    public void copy(NeuralNetwork N) {
        for(int l = 0; l < LAYERS; ++l) {
            for (int i = 0; i < weights[l].length; ++i) {
                System.arraycopy(N.weights[l][i], 0, weights[l][i], 0, weights[l][i].length);
            }
            System.arraycopy(N.biases[l], 0, biases[l], 0, biases[l].length);
        }
    }

    private void computeLayer(int layer, double[] x) {
        for (int i = 0; i < weights[layer][0].length; ++i) {
            z[layer][i] = 0;
            for (int j = 0; j < weights[layer].length; ++j) {
                z[layer][i] += weights[layer][j][i] * x[j];
            }
            z[layer][i] += biases[layer][i];
        }
    }

    // applying weights and biases
    private void computeLayer(int layer) {
        if (layer > 0) {
            computeLayer(layer, a[layer - 1]);
        } else {
            computeLayer(layer, input);
        }
    }

    // activation function
    private void ReLu(int layer) {
        for (int i = 0; i < a[layer].length; ++i) {
            a[layer][i] = max(z[layer][i], 0);
        }
    }

    private double maxComponent(double[] v) {
        double max = Double.MIN_VALUE;
        for (double vi : v) {
            max = max(max, vi);
        }
        return max;
    }

    // final activation function
    private void SoftMax() {
        double denominator = 0;
        double max = maxComponent(z[LAYERS - 1]);
        for (int i = 0; i < z[LAYERS - 1].length; ++i) {
            denominator += exp(z[LAYERS - 1][i] - max);
            // subtract max to ensure the value is in (0, 1)
        }
        for (int i = 0; i < a[LAYERS - 1].length; ++i) {
            a[LAYERS - 1][i] = exp(z[LAYERS - 1][i] - max) / denominator;
        }
    }

    // compute the whole function
    void feedForward() {
        computeLayer(0);
        for (int l = 1; l < z.length; ++l) {
            ReLu(l - 1);
            computeLayer(l);
        }
        SoftMax();
    }

    // set the normalized input
    void normalizeInput(int[] pixels) {
        for (int i = 0; i < input.length; ++i) {
            input[i] = pixels[i] * 1.0 / 255;
        }
    }

    //
    double[] classify(double[] prediction) {
        double max = 0;
        double[] res = new double[2];
        for (int i = 0; i < prediction.length; ++i) {
            if (prediction[i] > max) {
                max = prediction[i];
                res[0] = i;
            }
        }
        res[1] = max * 100;
        return res;
    }

    public double test(Sample[] samples) {

        shuffle(samples);

        int count = 0;
        double rightConfidence = 0;
        double wrongConfidence = 0;

        for (Sample sample : samples) {
            normalizeInput(sample.getImage());
            feedForward();
            double[] res = classify(a[LAYERS - 1]);
            if (res[0] == sample.getLabel()) {
                count++;
                rightConfidence += res[1];
            } else {
                wrongConfidence += res[1];
            }
            if (visualMode) {
                results.addSample(res[0] == sample.getLabel(), sample.getImage(), res[0], res[1]);
            }
        }

        System.out.printf("Average confidence: %.2f %% \n" , (wrongConfidence + rightConfidence) / samples.length);
        System.out.printf("Average correct confidence: %.2f %% \n" , rightConfidence / count);
        System.out.printf("Average wrong confidence: %.2f %% \n" , wrongConfidence / (samples.length - count));

        return count * 100.0 / samples.length;

    }

    public double test(int[][] images, int[] labels) {

        return test(getSamples(images, labels));
    }

}
