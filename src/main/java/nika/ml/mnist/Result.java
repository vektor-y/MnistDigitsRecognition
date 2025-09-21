package nika.ml.mnist;

public class Result {
    int[] image;
    int label;
    double confidence;

    public Result(int[] image, int label, double confidence) {
        this.image = image;
        this.label = label;
        this.confidence = confidence;
    }

    public int[] getImage() {
        return image;
    }

    public int getLabel() {
        return label;
    }

    public double getConfidence() {
        return confidence;
    }
}
