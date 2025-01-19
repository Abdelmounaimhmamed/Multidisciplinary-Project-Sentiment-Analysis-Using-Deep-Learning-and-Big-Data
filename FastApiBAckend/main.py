from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
from typing import Dict
import json
import time
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app 
app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify your frontend URL (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# MongoDB Connection
client = MongoClient("mongodb://localhost:27017")
db = client["TwitterSentimentAnalysis"]
collection = db["UserResults"]
collection2 = db["TweetPrediction"]

# Initialize SparkSession (global to reuse across requests)
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("TwitterSentimentAnalysis") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2,org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Pydantic model for request validation
class SentimentRequest(BaseModel):
    tweet: str

# Sentiment Analysis Function
def analyse_sentiment(text: str, pretrained_path: str) -> Dict:
    start = time.time()

    # Create DataFrame with input text
    tweet = spark.createDataFrame([(text,)], ["tweet"])

    # Load the pretrained model
    model = PipelineModel.load(pretrained_path)

    # Transform the data through the pipeline
    transformed_data = model.stages[0].transform(tweet)
    transformed_data = model.stages[1].transform(transformed_data)
    transformed_data = model.stages[2].transform(transformed_data)
    predictions = model.stages[-1].transform(transformed_data)
    print(predictions)

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

    end = time.time()
    print(f"Time Taken: {end - start:.2f} seconds")

    return {
        "prediction": prediction["prediction"],
        "sentiment": prediction["sentiment"]
    }

# Sentiment Analysis Endpoint
@app.post("/analyze-sentiment")
async def analyze_sentiment(request: SentimentRequest):
    if not request.tweet:
        raise HTTPException(status_code=400, detail="Tweet content cannot be empty")

    try:
        # Analyze sentiment using the Spark model
        result = analyse_sentiment(request.tweet, "../Pretrained_LogisticRegression.pkl")

        # Save to MongoDB
        data = {
            "timestamp": datetime.utcnow(),
            "tweet": request.tweet,
            "prediction": result["prediction"],
            "sentiment": result["sentiment"]
        }
        collection.insert_one(data)

        return {
            "message": "Sentiment analyzed successfully",
            "result": result
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing sentiment")


# Custom JSON encoder for MongoDB ObjectId and datetime
def serialize_document(doc):
    return {
        "_id": str(doc["_id"]),
        "timestamp": doc["timestamp"].isoformat() if "timestamp" in doc and isinstance(doc["timestamp"], datetime) else None,
        "tweet": doc["tweet"],
        "prediction": doc["prediction"],
        "sentiment": doc["sentiment"],
    }
    
@app.get("/tweets")
async def get_tweets():
    try:
        records = collection2.find().limit(10)
        tweets = []
        for record in records:
            try:
                tweets.append(serialize_document(record))
            except Exception as e:
                print(f"Error serializing document: {record}, Error: {str(e)}")
        return {"tweets": tweets}
    except Exception as e:
        return {"error": str(e)}


# Test Endpoint
@app.get("/")
def root():
    return {"message": "FastAPI Sentiment Analysis Server is running"}



