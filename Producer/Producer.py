import pandas as pd
from kafka import KafkaProducer
import logging
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# Kafka configuration
producer = KafkaProducer(bootstrap_servers='kafka1:9092')

topic_name = 'twitter'  # Modify this according to your topic name

# Define column names if your CSV file doesn't have a header row
column_names = ['id', 'game', 'sentiment', 'tweet']

def read_csv_and_produce(csv_file, batch_size=1, sleep_time=1):
    df = pd.read_csv(csv_file, header=None, names=column_names)
    num_rows = len(df)
    current_batch = 0

    while current_batch < num_rows:
        batch_data = df.iloc[current_batch:current_batch+batch_size].to_dict(orient='records')

        for tweet_data in batch_data:
            # Send the data to Kafka
            producer.send(topic_name, value=json.dumps(tweet_data).encode('utf-8'))
            logging.info(f"Data sent to Kafka: {tweet_data}")

        current_batch += batch_size
        time.sleep(sleep_time)

if __name__ == "__main__":
    csv_file_path = '/twitter_validation.csv/twitter_validation.csv/twitter_validation.csv'  # Change this to the path of your CSV file
    read_csv_and_produce(csv_file_path)
