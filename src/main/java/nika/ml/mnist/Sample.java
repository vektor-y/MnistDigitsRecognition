package nika.ml.mnist;

import java.util.Random;

public class Sample {

    int[] image;
    int label;

    Sample(int[] image, int label) {
        this.image = image;
        this.label = label;
    }

    public int getLabel() {
        return label;
    }

    public int[] getImage() {
        return image;
    }

    // get Samples array from given images and labels
    public static Sample[] getSamples(int[][] images, int[] labels) {
        Sample[] res = new Sample[images.length];
        for (int i = 0; i < res.length; ++i) {
            res[i] = new Sample(images[i], labels[i]);
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
