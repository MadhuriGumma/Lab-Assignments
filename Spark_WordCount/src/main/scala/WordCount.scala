import org.apache.spark.{SparkConf, SparkContext}



/**
  * Created by gumma on 2/1/2017.
  */
object WordCount {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir", "F:\\HaddopForSpark");

    val sparkConf = new SparkConf().setAppName("WordCount").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    val inputFile = sc.textFile("input.txt")
    val file_input = inputFile.flatMap(line=>{line.split(" ").filter(_.nonEmpty)})
    val apply_filter = (file_input.filter(e=>e.stripPrefix(",").stripSuffix(",").trim.length%2 !=0))
    val stripped_data = apply_filter.map(e=>e.stripPrefix(",").stripSuffix(",").stripPrefix(".").stripSuffix(".").trim)
    val console_data = stripped_data.collect()
    stripped_data.saveAsTextFile("outputFile")
    println(console_data.mkString("\n"))
  }
}
