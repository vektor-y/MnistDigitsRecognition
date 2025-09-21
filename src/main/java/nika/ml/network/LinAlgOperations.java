package nika.ml.network;

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

    static double[] column(double[][] A, int n) {
        double[] res = new double[A.length];
        for (int i = 0; i < A.length; ++i) {
            res[i] = A[i][n];
        }
        return res;
    }

    static void setColumn(double[][] M, double[] v, int n) {
        for (int i = 0; i < M.length; ++i) {
            M[i][n] = v[i];
        }
    }


    static double[] add(double[] a, double[] b) {

        assert a.length == b.length;

        double[] res = new double[a.length];
        for (int i = 0; i < res.length; ++i) {
            res[i] = a[i] + b[i];
        }
        return res;
    }

    static double[][] trans(double [][]A) {
        double[][] res = new double[A[0].length][A.length];
        for (int i = 0; i < res.length; ++i) {
            for (int j = 0; j < res[i].length; ++j) {
                res[i][j] = A[j][i];
            }
        }
        return res;
    }

    //    static double[][] dot(double[][] A, double[][] B) {
//
//        assert A[0].length == B.length;
//
//        double[][] res = new double[A.length][B[0].length];
//        for (int i = 0; i < B[0].length; ++i) {
//            setColumn(res, dot(A, column(B, i)), i);
//        }
//        return res;
//    }

}
