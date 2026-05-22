package ru.itmo.bigdata

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.FloatType
import java.sql.SQLException
import java.io.IOException

object DataMartApp {

  private val RequiredColumns = Seq(
    "ENERGY_100G", "FAT_100G", "CARBOHYDRATES_100G",
    "SUGARS_100G", "PROTEINS_100G", "SALT_100G"
  )

  def main(args: Array[String]): Unit = {
    val dbUrl = sys.env.getOrElse("DB_URL", "jdbc:oracle:thin:@oracle-db:1521/XEPDB1")
    val dbUser = sys.env.getOrElse("DB_USER", "SYSTEM")
    val dbPassword = sys.env.getOrElse("DB_PASSWORD", "oracle_password")
    val dbDriver = "oracle.jdbc.driver.OracleDriver"

    val spark = SparkSession.builder()
      .appName("Scala_Spark_Data_Mart")
      .master("local[*]")
      .getOrCreate()

    try {
      val rawDf = spark.read
        .format("jdbc")
        .option("url", dbUrl)
        .option("dbtable", "SYSTEM.RAW_FOOD_DATA")
        .option("user", dbUser)
        .option("password", dbPassword)
        .option("driver", dbDriver)
        .load()

      val normalizedDf = rawDf.select(rawDf.columns.map(c => col(c).as(c.toUpperCase)): _*)

      validateSchema(normalizedDf)
      val validatedDf = filterValidRecords(normalizedDf, spark)
      val preprocessedDf = preprocessData(validatedDf)

      preprocessedDf.write
        .format("jdbc")
        .option("url", dbUrl)
        .option("dbtable", "SYSTEM.PREPROCESSED_FOOD_DATA")
        .option("user", dbUser)
        .option("password", dbPassword)
        .option("driver", dbDriver)
        .mode("overwrite")
        .save()

      println("✅ Витрина данных успешно сформирована на Scala!")

    } catch {
      case e: SQLException =>
        System.err.println(s"Ошибка СУБД: ${e.getMessage}")
        sys.exit(1)
      case e: IllegalArgumentException =>
        System.err.println(s"Ошибка валидации схемы: ${e.getMessage}")
        sys.exit(2)
      case e: IOException =>
        System.err.println(s"Ошибка ввода-вывода Spark: ${e.getMessage}")
        sys.exit(3)
    } finally {
      spark.stop()
    }
  }

  def validateSchema(df: DataFrame): Unit = {
    val columns = df.columns.toSet
    val missing = RequiredColumns.filterNot(columns.contains)
    if (missing.nonEmpty) {
      throw new IllegalArgumentException(s"Отсутствуют колонки: ${missing.mkString(", ")}")
    }
  }

  def filterValidRecords(df: DataFrame, spark: SparkSession): DataFrame = {
    val schema = df.schema
    val validatedRdd = df.rdd.filter { row =>
      try {
        RequiredColumns.forall { colName =>
          val idx = row.fieldIndex(colName)
          if (row.isNullAt(idx)) false
          else {
            val value = row.get(idx)
            value.isInstanceOf[Double] || value.isInstanceOf[Float] || value.isInstanceOf[java.math.BigDecimal] || value.isInstanceOf[Int]
          }
        }
      } catch {
        case _: Exception => false
      }
    }
    spark.createDataFrame(validatedRdd, schema)
  }

  def preprocessData(df: DataFrame): DataFrame = {
    var cleanDf = df
    RequiredColumns.foreach(c => {
      cleanDf = cleanDf.withColumn(c, col(c).cast(FloatType))
    })
    cleanDf = cleanDf.na.drop(RequiredColumns)
    val percentageCols = Seq("FAT_100G", "CARBOHYDRATES_100G", "PROTEINS_100G", "SALT_100G", "SUGARS_100G")
    percentageCols.foldLeft(cleanDf) { (tempDf, colName) =>
      tempDf.filter(col(colName) >= 0.0 && col(colName) <= 100.0)
    }.filter(col("ENERGY_100G") >= 0.0)
  }
}