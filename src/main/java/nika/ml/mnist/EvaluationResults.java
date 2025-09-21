package nika.ml.mnist;

import java.util.LinkedList;
import java.util.List;

import static java.lang.Math.round;

public class EvaluationResults {

    private int VISUALISATION_SIZE;
    List<Result> correct;
    List<Result> wrong;

    public EvaluationResults() { }

    public EvaluationResults(int VISUALISATION_SIZE) {

        this.VISUALISATION_SIZE = VISUALISATION_SIZE;
        correct = new LinkedList<>();
        wrong = new LinkedList<>();
    }

    public int getVisualSize() {
        return VISUALISATION_SIZE;
    }

    public void setVisualSize(int VISUALISATION_SIZE) {
        this.VISUALISATION_SIZE = VISUALISATION_SIZE;
    }

    public List<Result> getCorrect() {
        return correct;
    }

    public List<Result> getWrong() {
        return wrong;
    }

    public void addSample(boolean isCorrect, int[] image, double label, double confidence) {

        if (isCorrect && correct.size() < VISUALISATION_SIZE) {
            correct.add(new Result(image, (int) round(label), confidence));
            return;
        }
        if (!isCorrect && wrong.size() < VISUALISATION_SIZE) {
            wrong.add(new Result(image, (int) round(label), confidence));
        }
    }



}
