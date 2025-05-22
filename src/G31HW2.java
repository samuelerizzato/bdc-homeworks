import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class G31HW2 {
    static double NA;
    static double NB;

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

        System.out.println("Input file = " + args[0] + ", L = " + L + ", K = " + K + ", M = " + M);

        JavaPairRDD<Vector, String> inputPoints = sc
                .textFile(args[0])
                .repartition(L)
                .mapToPair((point) -> {
                    String[] data = point.split(",");
                    double[] coordinates = new double[data.length - 1];
                    for (int i = 0; i < data.length - 1; i++) {
                        coordinates[i] = Double.parseDouble(data[i]);
                    }
                    return new Tuple2<>(Vectors.dense(coordinates), data[data.length - 1]);
                })
                .cache();

        NA = inputPoints.filter((pointPair) -> pointPair._2().equals("A")).count();
        NB = inputPoints.filter((pointPair) -> pointPair._2().equals("B")).count();
        System.out.println("N = " + inputPoints.count() + ", NA = " + (long)NA + ", NB = " + (long)NB);

        long start = System.currentTimeMillis();
        KMeansModel model = KMeans.train(inputPoints.keys().rdd(), K, M, "kmeans||", 1);
        long end = System.currentTimeMillis();
        long stdCentDuration = end - start;
        Vector[] cStand = model.clusterCenters();

        start = System.currentTimeMillis();
        Vector[] cFair = MRFairLloyd(inputPoints, K, M);
        end = System.currentTimeMillis();
        long fairCentDuration = end - start;

        start = System.currentTimeMillis();
        double stdObjective = MRComputeFairObjective(inputPoints, cStand);
        end = System.currentTimeMillis();

        long stdObjectiveDuration = end - start;

        start = System.currentTimeMillis();
        double fairObjective = MRComputeFairObjective(inputPoints, cFair);
        end = System.currentTimeMillis();

        long fairObjectiveDuration = end - start;


        System.out.format("Fair Objective with Standard Centers = %.6f%n", stdObjective);
        System.out.format("Fair Objective with Fair Centers = %.6f%n", fairObjective);
        System.out.format("Time to compute standard centers = %d ms%n", stdCentDuration);
        System.out.format("Time to compute fair centers = = %d ms%n", fairCentDuration);
        System.out.format("Time to compute objective with standard centers = %d ms%n", stdObjectiveDuration);
        System.out.format("Time to compute objective with fair centers = %d ms%n", fairObjectiveDuration);
    }

    static Vector[] MRFairLloyd(JavaPairRDD<Vector, String> inputPoints, int K, int M) {
        KMeansModel model = KMeans.train(inputPoints.keys().rdd(), K, 0, "kmeans||", 1);
        Vector[] C = model.clusterCenters();

        for (int it = 0; it < M; it++) {
            Vector[] finalC = C;
            JavaPairRDD<Integer, Tuple2<Vector, String>> clusters = inputPoints.mapToPair(pointPair -> {
                double closestDistance = Vectors.sqdist(pointPair._1(), finalC[0]);
                int closestCenterIndex = 0;
                for (int i = 1; i < finalC.length; i++) {
                    double distance = Vectors.sqdist(pointPair._1(), finalC[i]);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestCenterIndex = i;
                    }
                }
                return new Tuple2<>(closestCenterIndex, pointPair);
            }).cache();

            C = runCentroidSelection(clusters, C, K);
        }

        return C;
    }

    static Vector[] runCentroidSelection(JavaPairRDD<Integer, Tuple2<Vector, String>> clusters, Vector[] C, int K) {
        Vector[] fairC = new Vector[C.length];

        Map<Integer, Tuple2<Long, Long>> centerCounts = clusters
                .mapToPair(pointPair -> {
                    String group = pointPair._2()._2();
                    Tuple2<Long, Long> pointUnit = group.equals("A") ? new Tuple2<>(1L, 0L) : new Tuple2<>(0L, 1L);
                    return new Tuple2<>(pointPair._1(), pointUnit);
                })
                .mapPartitionsToPair(partition -> {
                    HashMap<Integer, Tuple2<Long, Long>> clusterCount = new HashMap<>();
                    while (partition.hasNext()) {
                        Tuple2<Integer, Tuple2<Long, Long>> tuple = partition.next();
                        Tuple2<Long, Long> currentVal = clusterCount.getOrDefault(tuple._1(), new Tuple2<>(0L, 0L));
                        Tuple2<Long, Long> sum = new Tuple2<>(tuple._2()._1() + currentVal._1(), tuple._2()._2() + currentVal._2());
                        clusterCount.put(tuple._1(), sum);
                    }
                    ArrayList<Tuple2<Integer, Tuple2<Long, Long>>> pairs = new ArrayList<>();
                    for (Map.Entry<Integer, Tuple2<Long, Long>> e : clusterCount.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .reduceByKey((c1, c2) -> new Tuple2<>(c1._1() + c2._1(), c1._2() + c2._2()))
                .collectAsMap();

        if (centerCounts.size() < K) {

            for (int k = 0; k < K; k++) {
                if (!centerCounts.containsKey(k)) {
                    int finalK = k;

                    Tuple2<Integer, Tuple2<Vector, String>> minDistance = clusters
                            .mapToPair(pointPair -> {
                                double distance = Vectors.sqdist(pointPair._2()._1(), C[finalK]);
                                return new Tuple2<>(distance, pointPair);
                            })
                            .mapPartitionsToPair(partition -> {
                                if (partition.hasNext()) {
                                    Tuple2<Double, Tuple2<Integer, Tuple2<Vector, String>>> minPointDistance = partition.next();
                                    while (partition.hasNext()) {
                                        Tuple2<Double, Tuple2<Integer, Tuple2<Vector, String>>> pDistance = partition.next();
                                        if (pDistance._1() < minPointDistance._1()) {
                                            minPointDistance = pDistance;
                                        }
                                    }
                                    ArrayList<Tuple2<Integer, Tuple2<Double, Tuple2<Integer, Tuple2<Vector, String>>>>> pair = new ArrayList<>();
                                    pair.add(new Tuple2<>(0, minPointDistance));
                                    return pair.iterator();
                                }
                                return Collections.emptyIterator();
                            })
                            .reduceByKey((p1, p2) -> p1._1() < p2._1() ? p1 : p2)
                            .values()
                            .first()._2();

                    clusters = clusters.mapToPair(pair ->
                            pair.equals(minDistance) ? new Tuple2<>(finalK, pair._2()) : pair);

                }
            }

        }

        Map<Integer, Tuple2<Vector, Vector>> clusterSums = clusters
                .mapPartitionsToPair(partition -> {
                    HashMap<Integer, Tuple2<Vector, Vector>> partialClusterSums = new HashMap<>();
                    while (partition.hasNext()) {
                        Tuple2<Integer, Tuple2<Vector, String>> tuple = partition.next();
                        Vector currentPoint = tuple._2()._1();
                        Vector defaultVector = Vectors.zeros(currentPoint.size());
                        Tuple2<Vector, Vector> vectorSum = partialClusterSums.getOrDefault(tuple._1(), new Tuple2<>(defaultVector, defaultVector));
                        if (tuple._2()._2().equals("A")) {
                            partialClusterSums.put(tuple._1(), new Tuple2<>(sumVectors(vectorSum._1(), currentPoint), vectorSum._2()));
                        } else {
                            partialClusterSums.put(tuple._1(), new Tuple2<>(vectorSum._1(), sumVectors(vectorSum._2(), currentPoint)));
                        }
                    }
                    ArrayList<Tuple2<Integer, Tuple2<Vector, Vector>>> pairs = new ArrayList<>();
                    for (Map.Entry<Integer, Tuple2<Vector, Vector>> e : partialClusterSums.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .reduceByKey((c1, c2) -> new Tuple2<>(sumVectors(c1._1(), c2._1()), sumVectors(c1._2(), c2._2())))
                .collectAsMap();

        double[] alpha = new double[K];
        double[] beta = new double[K];
        double[] ell = new double[K];
        Vector[] aStdCentroids = new Vector[K];
        Vector[] bStdCentroids = new Vector[K];

        for (int j = 0; j < K; j++) {
            Tuple2<Long, Long> counts = centerCounts.get(j);
            Tuple2<Vector, Vector> sums = clusterSums.get(j);
            alpha[j] = counts._1() / NA;
            beta[j] = counts._2() / NB;

            Vector other = counts._1() > 0L ? divideVector(sums._1(), counts._1()) : divideVector(sums._2(), counts._2());
            Vector aCentroid = counts._1() > 0L ? divideVector(sums._1(), counts._1()) : other;
            Vector bCentroid = counts._2() > 0L ? divideVector(sums._2(), counts._2()) : other;

            aStdCentroids[j] = aCentroid;
            bStdCentroids[j] = bCentroid;

            ell[j] = Math.sqrt(Vectors.sqdist(aCentroid, bCentroid));
        }

        Map<String, Double> groupCosts = clusters
                .mapToPair(pointPair -> {
                    Vector point = pointPair._2()._1();
                    Vector centroid = pointPair._2()._2().equals("A") ? aStdCentroids[pointPair._1()] : bStdCentroids[pointPair._1()];
                    double distance = Vectors.sqdist(centroid, point);
                    return new Tuple2<>(pointPair._2()._2(), distance);
                })
                .mapPartitionsToPair(partition -> {
                    HashMap<String, Double> distances = new HashMap<>();
                    while (partition.hasNext()) {
                        Tuple2<String, Double> tuple = partition.next();
                        distances.put(tuple._1(), tuple._2() + distances.getOrDefault(tuple._1(), 0.0));
                    }
                    ArrayList<Tuple2<String, Double>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Double> e : distances.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .reduceByKey(Double::sum)
                .collectAsMap();

        double fixedA = groupCosts.get("A") / NA;
        double fixedB = groupCosts.get("B") / NB;

        double[] X = computeVectorX(fixedA, fixedB, alpha, beta, ell, K);

        for (int i = 0; i < K; i++) {
            if (ell[i] == 0) {
                fairC[i] = aStdCentroids[i];
            } else {
                fairC[i] = computeFairCentroid(aStdCentroids[i], bStdCentroids[i], ell[i], X[i]);
            }
        }

        return fairC;
    }

    static Vector computeFairCentroid(Vector centroidA, Vector centroidB, double ell, double x) {
        int pointSize = centroidA.size();
        double[] coordinates = new double[pointSize];
        for (int i = 0; i < pointSize; i++) {
            coordinates[i] = ((ell - x) * centroidA.apply(i) + x * centroidB.apply(i)) / ell;
        }
        return Vectors.dense(coordinates);
    }

    static Vector sumVectors(Vector v1, Vector v2) {
        double[] values = new double[v1.size()];
        for (int i = 0; i < v1.size(); i++) {
            values[i] = v1.apply(i) + v2.apply(i);
        }
        return Vectors.dense(values);
    }

    static Vector divideVector(Vector v, double scalar) {
        double[] coordinates = new double[v.size()];
        for (int i = 0; i < v.size(); i++) {
            coordinates[i] = v.apply(i) / scalar;
        }
        return Vectors.dense(coordinates);
    }

    static double[] computeVectorX(double fixedA, double fixedB, double[] alpha, double[] beta, double[] ell, int K) {
        double gamma = 0.5;
        double[] xDist = new double[K];
        double fA, fB;
        double power = 0.5;
        int T = 10;
        for (int t = 1; t <= T; t++) {
            fA = fixedA;
            fB = fixedB;
            power = power / 2;
            for (int i = 0; i < K; i++) {
                double temp = (1 - gamma) * beta[i] * ell[i] / (gamma * alpha[i] + (1 - gamma) * beta[i]);
                xDist[i] = temp;
                fA += alpha[i] * temp * temp;
                temp = (ell[i] - temp);
                fB += beta[i] * temp * temp;
            }
            if (fA == fB) {
                break;
            }
            gamma = (fA > fB) ? gamma + power : gamma - power;
        }
        return xDist;
    }

    static double MRComputeFairObjective(JavaPairRDD<Vector, String> U, Vector[] C) {
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
}
