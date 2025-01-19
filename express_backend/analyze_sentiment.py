import sys
import time
import json
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

def analyse_sentiment(text, pretrained_path):
    start = time.time()

    # Initialize SparkSession
    spark = SparkSession.builder \
    .master("local[*]") \
    .appName("TwitterSentimentAnalysis") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2,org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.ui.enabled", "false") \
    .config("spark.executor.extraJavaOptions", "-Dlog4j.configuration=file:/path/to/log4j.properties") \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:/path/to/log4j.properties") \
    .getOrCreate()

    # Suppress Spark logs
    spark.sparkContext.setLogLevel("ERROR")


    # Create DataFrame with input text
    tweet = spark.createDataFrame([(text,)], ["tweet"])

    # Load the pretrained model
    model = PipelineModel.load(pretrained_path)

    # Transform the data through the pipeline
    transformed_data = model.stages[0].transform(tweet)
    transformed_data = model.stages[1].transform(transformed_data)
    transformed_data = model.stages[2].transform(transformed_data)
    predictions = model.stages[-1].transform(transformed_data)

    # Extract prediction and sentiment
    prediction = predictions.selectExpr(
        "prediction",
        "CASE \
            WHEN prediction = 1.0 THEN 'positive' \
            WHEN prediction = 0.0 THEN 'negative' \
            WHEN prediction = 2.0 THEN 'neutral' \
            ELSE 'irrelevant' \
        END as sentiment"
    ).collect()[0]

    return {
        "prediction": prediction["prediction"],
        "sentiment": prediction["sentiment"]
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 analyze_sentiment.py <text> <pretrained_model_path>", file=sys.stderr)
        sys.exit(1)

    input_text = sys.argv[1]
    pretrained_model = sys.argv[2]

    try:
        result = analyse_sentiment(input_text, pretrained_model)
        # Print only the JSON result to stdout
        print(json.dumps(result))
    except Exception as e:
        # Print errors to stderr
        print(json.dumps({"error": str(e)}), file=sys.stderr)
