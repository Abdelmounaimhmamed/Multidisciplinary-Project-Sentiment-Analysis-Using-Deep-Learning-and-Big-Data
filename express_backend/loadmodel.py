import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import TextVectorization, Input
import tensorflow as tf

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the saved model
model = load_model("sentiment_analysis_model.h5", compile=False)

# Input tweets
example_text = "I love programming"
example_text_tensor = tf.constant(example_text)
# Convert tweets to a NumPy array (dtype must match TextVectorization input)


# Predict sentiments
predictions = model.predict(example_text_tensor)

# Decode predictions
sentiment_mapping = {0: "negative", 1: "positive"}
decoded_predictions = [sentiment_mapping[int(round(pred[0]))] for pred in predictions]

# Display results
for tweet, sentiment in zip(tweets, decoded_predictions):
    print(f"Tweet: {tweet} => Sentiment: {sentiment}")
