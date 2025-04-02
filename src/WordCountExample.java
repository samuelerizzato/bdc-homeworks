import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class WordCountExample{

    public static void main(String[] args) throws IOException {

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are: num_partitions, <path_to_file>
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("WordCount");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("OFF");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Read number of partitions
        int L = Integer.parseInt(args[0]);

        // Read input file and subdivide it into L random partitions
        // cache function is used because of back propagation
        JavaRDD<String> docs = sc.textFile(args[1]).repartition(L).cache();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SETTING GLOBAL VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long numdocs, numwords;
        numdocs = docs.count();
        System.out.println("Number of documents = " + numdocs);
        JavaPairRDD<String, Long> wordCounts;
        Random randomGenerator = new Random();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 1-ROUND WORD COUNT
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        wordCounts = docs.flatMapToPair(myMethods::wordCountPerDoc) // <-- MAP PHASE (R1)
                .reduceByKey((x, y) -> x+y);    // <-- REDUCE PHASE (R1)
        numwords = wordCounts.count();
        System.out.println("Number of distinct words in the documents = " + numwords);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 2-ROUND WORD COUNT - RANDOM KEYS ASSIGNED IN MAP PHASE
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        wordCounts = docs
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<Integer, Tuple2<String, Long>>> pairs = new ArrayList<>();
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(randomGenerator.nextInt(L), new Tuple2<>(e.getKey(), e.getValue())));
                    }
                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R1), in this case reduceKey doesn't work since we expect to do ops on tuples not longs
                .flatMapToPair(myMethods::gatherPairs) // <-- REDUCE PHASE (R1)
                .reduceByKey((x, y) -> x+y); // <-- REDUCE PHASE (R2)
        numwords = wordCounts.count();
        System.out.println("Number of distinct words in the documents = " + numwords);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 2-ROUND WORD COUNT - RANDOM KEYS ASSIGNED ON THE FLY
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        wordCounts = docs.flatMapToPair(myMethods::wordCountPerDoc)
                .groupBy((wordcountpair) -> randomGenerator.nextInt(L))  // <-- KEY ASSIGNMENT+SHFFLE+GROUPING
                .flatMapToPair(myMethods::gatherPairs)
                .reduceByKey((x, y) -> x+y); // <-- REDUCE PHASE (R2)
        numwords = wordCounts.count();
        System.out.println("Number of distinct words in the documents = " + numwords);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 2-ROUND WORD COUNT - SPARK PARTITIONS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        wordCounts = docs.flatMapToPair(myMethods::wordCountPerDoc)
                .mapPartitionsToPair((element) -> {    // <-- REDUCE PHASE (R1)
                    HashMap<String, Long> counts = new HashMap<>();
                    while (element.hasNext()){
                        Tuple2<String, Long> tuple = element.next();
                        counts.put(tuple._1(), tuple._2() + counts.getOrDefault(tuple._1(), 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()     // <-- SHUFFLE+GROUPING
                .mapValues((it) -> { // <-- REDUCE PHASE (R2)
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                }); // Obs: one could use reduceByKey in place of groupByKey and mapValues
        numwords = wordCounts.count();
        System.out.println("Number of distinct words in the documents = " + numwords);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // COMPUTE AVERAGE WORD LENGTH
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        int avgwordlength = wordCounts
                .map((tuple) -> tuple._1().length())
                .reduce((x, y) -> x+y);
        System.out.println("Average word length = " + (double) avgwordlength/ (double) numwords);

    }

}

class myMethods {
    public static Iterator<Tuple2<String,Long>> wordCountPerDoc(String document) {
        String[] tokens = document.split(" ");
        HashMap<String, Long> counts = new HashMap<>();
        ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
        for (String token : tokens) {
            counts.put(token, 1L + counts.getOrDefault(token, 0L));
        }
        for (Map.Entry<String, Long> e : counts.entrySet()) {
            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
        }
        return pairs.iterator();
    }
    public static Iterator<Tuple2<String,Long>> gatherPairs(Tuple2<Integer,Iterable<Tuple2<String, Long>>> element) {
        HashMap<String, Long> counts = new HashMap<>();
        for (Tuple2<String, Long> c : element._2()) {
            counts.put(c._1(), c._2() + counts.getOrDefault(c._1(), 0L));
        }
        ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
        for (Map.Entry<String, Long> e : counts.entrySet()) {
            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
        }
        return pairs.iterator();
    }
}
