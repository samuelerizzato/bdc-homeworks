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
                    double x = Double.parseDouble(data[0]);
                    double y = Double.parseDouble(data[1]);
                    return new Tuple2<>(Vectors.dense(x, y), data[2]);
                })
                .cache();

        long NA = U.filter((point) -> point._2().equals("A")).count();
        long NB = U.filter((point) -> point._2().equals("B")).count();
        System.out.println("N = " + U.count() + ", NA = " + NA + ", NB = " + NB);

        KMeansModel clusters = KMeans.train(U.map(Tuple2::_1).rdd(), K, M, "kmeans||", 1);
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
        JavaPairRDD<Integer, HashMap<String, Long>> result = U.mapToPair((element) -> {
            double closestDistance = Vectors.sqdist(element._1(), C[0]);
            int closestCenterIndex = 0;
            for (int i = 1; i < C.length; i++) {
                double distance = Vectors.sqdist(element._1(), C[i]);
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestCenterIndex = i;
                }
            }
            return new Tuple2<>(closestCenterIndex, element._2());
        }).mapPartitionsToPair((element) -> {
            HashMap<Integer, HashMap<String, Long>> centerStats = new HashMap<>();
            while (element.hasNext()) {
                Tuple2<Integer, String> tuple = element.next();
                centerStats.computeIfAbsent(tuple._1(), k -> new HashMap<>());
                HashMap<String, Long> groupCenters = centerStats.get(tuple._1());
                groupCenters.put(tuple._2(), 1L + groupCenters.getOrDefault(tuple._2(), 0L));
            }
            ArrayList<Tuple2<Integer, HashMap<String, Long>>> pairs = new ArrayList<>();
            for (Map.Entry<Integer, HashMap<String, Long>> e : centerStats.entrySet()) {
                pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
            }
            return pairs.iterator();
        }).reduceByKey((center1, center2) -> {
            HashMap<String, Long> centerGroupSum = new HashMap<>();
            for (Map.Entry<String, Long> e : center1.entrySet()) {
                centerGroupSum.put(e.getKey(), e.getValue() + centerGroupSum.getOrDefault(e.getKey(), 0L));
            }
            for (Map.Entry<String, Long> e : center2.entrySet()) {
                centerGroupSum.put(e.getKey(), e.getValue() + centerGroupSum.getOrDefault(e.getKey(), 0L));
            }
            return centerGroupSum;
        });

        Map<Integer, HashMap<String, Long>> centersCount = result.collectAsMap();

        for (int i = 0; i < C.length; i++) {
            HashMap<String, Long> counts = centersCount.get(i);
            long NAi = counts.get("A");
            long NBi = counts.get("B");
            double[] centerCoordinates = C[i].toArray();

            System.out.format("i = %d, center = (%.6f,%.6f), NA%d = %d, NB%d = %d%n",
                    i, centerCoordinates[0], centerCoordinates[1], i, NAi, i, NBi);
        }
    }
}
