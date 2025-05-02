import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class G31HW1 {
    public static void main(String[] args) throws IOException {
        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: file_path num_partitions num_clusters num_iterations");
        }

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        SparkConf conf = new SparkConf(true).setAppName("G31HW");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("OFF");

        int L = Integer.parseInt(args[1]), K = Integer.parseInt(args[2]), M = Integer.parseInt(args[3]);

        System.out.println("Input file = " + args[0] + ", L = " + args[1] + ", K = " + args[2] + ", M = " + args[3]);

        JavaRDD<String> inputPoints = sc.textFile(args[0]).repartition(L);
        JavaPairRDD<Vector, String> U = inputPoints
                .mapToPair((point) -> {
                    String[] data = point.split(",");
                    double[] coordinates = new double[data.length - 1];
                    for (int i = 0; i < data.length - 1; i++) {
                        coordinates[i] = Double.parseDouble(data[i]);
                    }
                    return new Tuple2<>(Vectors.dense(coordinates), data[data.length - 1]);
                })
                .cache();

        long NA = U.filter((pointPair) -> pointPair._2().equals("A")).count();
        long NB = U.filter((pointPair) -> pointPair._2().equals("B")).count();
        System.out.println("N = " + U.count() + ", NA = " + NA + ", NB = " + NB);

        KMeansModel clusters = KMeans.train(U.keys().rdd(), K, M, "kmeans||", 1);
        Vector[] C = clusters.clusterCenters();

        System.out.format("Delta(U,C) = %.6f%n", MRComputeStandardObjective(U, C));
        System.out.format("Phi(A,B,C) = %.6f%n", MRComputeFairObjective(U, C));

        MRPrintStatistics(U, C);
    }

    static Double MRComputeStandardObjective(JavaPairRDD<Vector, String> U, Vector[] C) {
        long N = U.count();

        return U
                .mapToPair((element) -> {
                    double closestDistance = Vectors.sqdist(element._1(), C[0]);
                    for (int i = 1; i < C.length; i++) {
                        double distance = Vectors.sqdist(element._1(), C[i]);
                        if (distance < closestDistance) {
                            closestDistance = distance;
                        }
                    }
                    return new Tuple2<>(closestDistance, element._2());
                })
                .mapPartitionsToPair((element) -> {
                    double distancesSum = 0.0;
                    while (element.hasNext()) {
                        Tuple2<Double, String> tuple = element.next();
                        distancesSum += tuple._1();
                    }
                    ArrayList<Tuple2<Long, Double>> pairs = new ArrayList<>();
                    pairs.add(new Tuple2<>(0L, distancesSum));
                    return pairs.iterator();
                })
                .groupByKey()
                .mapValues((it) -> {
                    double totalSum = 0.0;
                    for (double sum : it) {
                        totalSum += sum;
                    }
                    return totalSum / N;
                })
                .values()
                .first();
    }

    static Double MRComputeFairObjective(JavaPairRDD<Vector, String> U, Vector[] C) {
        HashMap<String, Long> cardinalities = new HashMap<>();
        cardinalities.put("A", U.filter((point) -> point._2().equals("A")).count());
        cardinalities.put("B", U.filter((point) -> point._2().equals("B")).count());

        return U
                .mapToPair((element) -> {
                    double closestDistance = Vectors.sqdist(element._1(), C[0]);
                    for (int i = 1; i < C.length; i++) {
                        double distance = Vectors.sqdist(element._1(), C[i]);
                        if (distance < closestDistance) {
                            closestDistance = distance;
                        }
                    }
                    return new Tuple2<>(closestDistance, element._2());
                })
                .mapPartitionsToPair((element) -> {
                    HashMap<String, Double> distances = new HashMap<>();
                    while (element.hasNext()) {
                        Tuple2<Double, String> tuple = element.next();
                        distances.put(tuple._2(), tuple._1() + distances.getOrDefault(tuple._2(), 0.0));
                    }
                    ArrayList<Tuple2<String, Double>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Double> e : distances.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()
                .mapToPair((element) -> {
                    double cardinality = cardinalities.get(element._1());
                    Iterator<Double> it = element._2().iterator();
                    double totalSum = 0.0;
                    while (it.hasNext()) {
                        double sum = it.next();
                        totalSum += sum;
                    }
                    return new Tuple2<>(0, totalSum / cardinality);
                })
                .reduceByKey(Math::max)
                .values()
                .first();
    }

    static void MRPrintStatistics(JavaPairRDD<Vector, String> U, Vector[] C) {
        Map<Integer, Tuple2<Long, Long>> centerCounts = U
                .mapToPair((element) -> {
                    double closestDistance = Vectors.sqdist(element._1(), C[0]);
                    int closestCenterIndex = 0;
                    for (int i = 1; i < C.length; i++) {
                        double distance = Vectors.sqdist(element._1(), C[i]);
                        if (distance < closestDistance) {
                            closestDistance = distance;
                            closestCenterIndex = i;
                        }
                    }
                    if (element._2().equals("A")) {
                        return new Tuple2<>(closestCenterIndex, new Tuple2<>(1L, 0L));
                    }
                    return new Tuple2<>(closestCenterIndex, new Tuple2<>(0L, 1L));
                })
                .mapPartitionsToPair((element) -> {
                    HashMap<Integer, Tuple2<Long, Long>> centerStats = new HashMap<>();
                    while (element.hasNext()) {
                        Tuple2<Integer, Tuple2<Long, Long>> tuple = element.next();
                        Tuple2<Long, Long> currentVal = centerStats.getOrDefault(tuple._1(), new Tuple2<>(0L, 0L));
                        Tuple2<Long, Long> sum = new Tuple2<>(tuple._2()._1() + currentVal._1(), tuple._2()._2() + currentVal._2());
                        centerStats.put(tuple._1(), sum);
                    }
                    ArrayList<Tuple2<Integer, Tuple2<Long, Long>>> pairs = new ArrayList<>();
                    for (Map.Entry<Integer, Tuple2<Long, Long>> e : centerStats.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .reduceByKey((center1, center2) -> new Tuple2<>(center1._1() + center2._1(), center1._2() + center2._2()))
                .collectAsMap();

        for (int i = 0; i < C.length; i++) {
            Tuple2<Long, Long> counts = centerCounts.get(i);

            System.out.format("i = %d, center = (", i);
            for (int j = 0; j < C[i].size() - 1; j++) {
                System.out.format("%.6f,", C[i].apply(j));
            }
            System.out.format("%.6f), NA%d = %d, NB%d = %d%n",
                    C[i].apply(C[i].size() - 1), i, counts._1(), i, counts._2());
        }
    }
}
