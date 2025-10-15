package nika.ml.network;

import java.util.Random;

public class Sample {

    private final double[] input;
    private final int label;

    public Sample(double[] input, int label) {
        this.input = input;
        this.label = label;
    }

    public double[] getInput() {
        return input;
    }

    public int getLabel() {
        return label;
    }

    // get Samples array from given inputs and labels
    public static Sample[] getSamples(double[][] inputs, int[] labels) {
        Sample[] res = new Sample[inputs.length];
        for (int i = 0; i < res.length; ++i) {
            res[i] = new Sample(inputs[i], labels[i]);
        }
        return res;
    }

    public static void shuffle(Sample[] samples) {

        Random rand = new Random();
        for (int i = samples.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1); // 0 ≤ j ≤ i
            // swap samples[i] and samples[j]
            Sample tmp = samples[i];
            samples[i] = samples[j];
            samples[j] = tmp;
        }
    }
}
