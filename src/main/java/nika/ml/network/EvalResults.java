package nika.ml.network;

import java.util.LinkedList;
import java.util.List;

public class EvalResults {
    private final List<Result> correct;
    private final List<Result> wrong;
    private double accuracy;
    private double confidence;
    private double correctConfidence;
    private double wrongConfidence;

    public EvalResults() {
        correct = new LinkedList<>();
        wrong = new LinkedList<>();
    }

    public List<Result> getCorrect() {
        return correct;
    }

    public List<Result> getWrong() {
        return wrong;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public double getConfidence() {
        return confidence;
    }

    public double getCorrectConfidence() {
        return correctConfidence;
    }

    public double getWrongConfidence() {
        return wrongConfidence;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public void setCorrectConfidence(double correctConfidence) {
        this.correctConfidence = correctConfidence;
    }

    public void setWrongConfidence(double wrongConfidence) {
        this.wrongConfidence = wrongConfidence;
    }

    public void addResult(boolean isCorrect, Result result) {

        List<Result> list = isCorrect ? correct : wrong;
        list.add(result);
    }

    @Override
    public String toString() {
        return  String.format("Accuracy: %.2f %% \n", accuracy * 100) +
                String.format("Average confidence: %.2f %% \n" , confidence) +
                String.format("Average correct confidence: %.2f %% \n" , correctConfidence) +
                String.format("Average wrong confidence: %.2f %%" , wrongConfidence);
    }
}
