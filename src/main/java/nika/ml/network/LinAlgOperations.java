package nika.ml.network;

import java.util.Arrays;

public final class LinAlgOperations {

    static void dot(double[][] A, double[] b, double[] res) {

        if (A[0].length != b.length) {
            System.out.println("LAO.dot(M, v) dimensions incompatible.");
        }

        for (int i = 0; i < A.length; ++i) {
            res[i] = dot(A[i], b);
        }
    }

    static double dot(double[] a, double[] b) {

        if (a.length != b.length) {
            System.out.println("LAO.dot(v, w): vector and vector dimensions incompatible.");
        }

        double sum = 0;
        for (int i = 0; i < a.length; ++i) {
            sum += a[i] * b[i];
        }
        if (Double.isNaN(sum)) {
            System.out.println("LAO.dot(v, w): NaN");
        }
        return sum;
    }

    static void add(double[][] a, double[][] b) {

        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[i].length; ++j) {
                a[i][j] += b[i][j];
            }
        }
    }

    static void add(double[][][] a, double[][][] b) {

        for (int i = 0; i < a.length; ++i) {
            add(a[i], b[i]);
        }
    }

    static void sub(double[] a, double[] b, double[] res) {
        for (int i = 0; i < a.length; ++i) {
            res[i] = a[i] - b[i];
        }
    }

    // reset a matrix
    static void reset(double[][] a) {
        for (double[] doubles : a) {
            Arrays.fill(doubles, 0);
        }
    }

    // reset a 3-dimensional array
    static void reset(double[][][] a) {
        for (double[][] doubles : a) {
            reset(doubles);
        }
    }

    // divide each entry by n
    static void average(double[][][] a, int n) {
        for (double[][] doubles : a) {
            average(doubles, n);
        }
    }

    // divide each entry by n
    static void average(double[][] a, int n) {
        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[i].length; ++j) {
                a[i][j] /= n;
            }
        }
    }
}
