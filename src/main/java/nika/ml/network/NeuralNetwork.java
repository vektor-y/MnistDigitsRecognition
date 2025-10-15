package nika.ml.network;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.*;
import java.util.Random;
import static java.lang.Math.*;
import static nika.ml.network.Sample.getSamples;
import static nika.ml.network.Sample.shuffle;

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

    @JsonIgnore
    EvalResults results;

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
        results = new EvalResults();

        for (int l = 0; l < LAYERS; ++l) {
            z[l] = new double[sizes[l + 1]];
            a[l] = new double[sizes[l + 1]];
        }
    }

    public void setInput(double[] newInput) {
        System.arraycopy(newInput, 0, input, 0, input.length);
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

    public EvalResults getResults() {
        return results;
    }

    public void save(String filePath) throws IOException {
        mapper.writeValue(new File(filePath), this);
    }

    public void save() throws IOException {

        StringBuilder filename = new StringBuilder("models/NN");
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

    // apply weights and biases
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

    // output interpretation function
    // chooses the biggest entry of the normalized output vector and returns its index
    public double[] interpret(double[] prediction) {
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

    public EvalResults test(Sample[] samples) {

        shuffle(samples);

        int count = 0;
        double correctConfidence = 0;
        double wrongConfidence = 0;

        for (Sample sample : samples) {
            setInput(sample.getInput());
            feedForward();
            double[] res = interpret(a[LAYERS - 1]);
            if (res[0] == sample.getLabel()) {
                count++;
                correctConfidence += res[1];
            } else {
                wrongConfidence += res[1];
            }
            results.addResult(res[0] == sample.getLabel(), new Result(sample, (int) round(res[0]), res[1]));
        }

        results.setAccuracy(count * 1.0 / samples.length);
        results.setConfidence((wrongConfidence + correctConfidence) / samples.length);
        results.setCorrectConfidence(correctConfidence / count);
        results.setWrongConfidence(wrongConfidence / (samples.length - count));

        return results;
    }

    public EvalResults test(double[][] inputs, int[] label) {
        return test(getSamples(inputs, label));
    }

}
