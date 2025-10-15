package nika.ml.network;

public class Result {
    private final Sample sample;
    private final int prediction;
    private final double confidence;

    public Result(Sample sample, int prediction, double confidence) {
        this.sample = sample;
        this.prediction = prediction;
        this.confidence = confidence;
    }

    public int getLabel() {
        return sample.getLabel();
    }

    public double[] getInput() {
        return sample.getInput();
    }

    public int getPrediction() {
        return prediction;
    }

    public double getConfidence() {
        return confidence;
    }
}
