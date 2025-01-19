import findspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, IntegerType, StringType, StructType
from pyspark.sql.functions import col, from_json, when
from pyspark.ml import PipelineModel
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import col
from pyspark.sql.functions import lower


def preprocess_data(df):
    # Remove punctuation
    df = df.withColumn("tweet", regexp_replace(col("tweet"), "[^\w\s]", ""))
    # Convert text to lowercase
    df = df.withColumn("tweet", lower(col("tweet")))
    # Drop rows with null values
    df = df.dropna()
    return df



if __name__ == "__main__":
    findspark.init()
    pretrained_model = '../Pretrained_LogisticRegression.pkl'
    spark = SparkSession.builder.master("local[*]").appName("TwitterSentimentAnalysis").config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2,org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1").config("spark.ui.enabled", "false").getOrCreate()

    # Schema for reading data from Kafka
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("game", StringType(), True),
        StructField("sentiment", StringType(), True),
        StructField("tweet", StringType(), True)
    ])

    # Read data from Kafka
    data = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:29092").option("subscribe", "twitter").option("startingOffsets", "latest").option("header", "true").load().selectExpr("CAST(value AS STRING) as message").withColumn("value", from_json("message", schema)).select("value.*")

    #preprocessing our data
    preprocessed_data = preprocess_data(data)

    
    loaded_model = PipelineModel.load(pretrained_model)

    # Make predictions
    predictions = loaded_model.transform(preprocessed_data)

    # Translate prediction to sentiment categories
    predictions = predictions.withColumn("sentiment_prediction",
                                         when(col("prediction") == 0, "negative")
                                         .when(col("prediction") == 1, "positive")
                                         .when(col("prediction") == 2, "neutral")
                                         .when(col("prediction") == 3, "irrelevant")
                                         .otherwise("unknown"))
    
    selected_predictions = predictions.select("id", "game", "tweet","sentiment_prediction", "prediction")

    def write_to_mongodb(df, epoch_id):
        df.write.format("mongo").mode("append").option("uri", "mongodb://localhost:27017/TwitterSentimentAnalysis.TweetPrediction").save()
        #print console
        df.show(truncate=False)

    # Output selected predictions to MongoDB using foreachBatch
    query = selected_predictions.writeStream.foreachBatch(write_to_mongodb).outputMode("append").start().awaitTermination()



