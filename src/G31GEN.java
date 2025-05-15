import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

public class G31GEN {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_points num_clusters");
        }

        int N = Integer.parseInt(args[0]);
        int K = Integer.parseInt(args[1]);
        generatePoints(N, K);
    }

    static void generatePoints(int N, int K) {
        int clusterPoints = N / K;
        int NA = (int)(0.8 * clusterPoints);
        int NB = clusterPoints - NA;
        double[][] pointsA = new double[NA*K][2];
        double[][] pointsB = new double[NB*K][2];

        double xCenter = 10.0;
        double yA = 60.0;
        double yB = 20.0;

        for (int i = 0; i < K; i++) {

            for (int j = 0; j < NA; j++) {
                double x = generateRandomNumber(xCenter - 10.0, xCenter + 10.0);
                double y = generateRandomNumber(yA - 10.0, yA + 10.0);
                pointsA[j + i*NA] = new double[]{x, y};
            }

            for (int j = 0; j < NB; j++) {
                double x = generateRandomNumber(xCenter - 10.0, xCenter + 10.0);
                double y = generateRandomNumber(yB - 10.0, yB + 10.0);
                pointsB[j + i*NB] = new double[]{x, y};
            }

            xCenter += 100.0;
        }

        DecimalFormatSymbols symbols = new DecimalFormatSymbols(Locale.US);
        symbols.setDecimalSeparator('.');
        symbols.setGroupingSeparator(' ');

        DecimalFormat df = new DecimalFormat("#.######", symbols);
        df.setGroupingUsed(false);

        System.out.format("NA=%d, NB=%d%n", NA, NB);
        for (double[] point : pointsA) {
            System.out.format("%s,%s,A%n", df.format(point[0]), df.format(point[1]));
        }

        for (double[] point : pointsB) {
            System.out.format("%s,%s,B%n", df.format(point[0]), df.format(point[1]));
        }
    }

    static double generateRandomNumber(double min, double max) {  return min + (Math.random() * (max - min)); }
}
