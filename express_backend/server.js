const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
require("dotenv").config();
const { MongoClient } = require('mongodb');
const cors = require("cors")


const app = express();
const PORT = 5000;

app.use(express.json());
app.enable(cors());
app.use(cors());


mongoose.connect("mongodb://localhost:27017/TwitterSentimentAnalysis", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});
const db = mongoose.connection;
db.on("error", console.error.bind(console, "MongoDB connection error:"));
db.once("open", () => {
  console.log("Connected to MongoDB");
});

// Define User Schema
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
});

const User = mongoose.model("User", userSchema);

const tweetSchema = new mongoose.Schema({
  timestamp: { type: Date, default: Date.now },
  tweet: String,
  prediction: Number,
  sentiment: String,
});



const tweetPredictionSchema = new mongoose.Schema({
  id: Number,
  game: String,
  tweet: String,
  sentiment_prediction: String,
  prediction: Number,
});


const uri = 'mongodb://localhost:27017'; 
const dbName = 'TwitterSentimentAnalysis';
const collectionName = 'TweetPrediction';



MongoClient.connect(uri, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(client => {
    console.log('Connected to MongoDB');
    let db = client.db(dbName);
  })
  .catch(error => {
    console.error('Error connecting to MongoDB:', error);
  });


app.get('/tweets', async (req, res) => {
  try {
    const collection = db.collection(collectionName);
    const tweets = await collection.find({}).toArray();
    res.status(200).json(tweets);
  } catch (error) {
    console.error('Error fetching tweets:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});
app.get('/tweetify', async (req, res) => {
  try {
    const collection = db.collection("UserResults");
    const tweets = await collection.find({}).toArray(); 
    res.status(200).json(tweets);
  } catch (error) {
    console.error('Error fetching tweets:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


// User Signup
app.post("/signup", async (req, res) => {
  const { name, email, password } = req.body;

  if (!name || !email || !password) {
    return res.status(400).json({ error: "All fields are required" });
  }

  try {
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: "Email is already registered" });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ name, email, password: hashedPassword });
    await newUser.save();

    res.status(201).json({ message: "User registered successfully" });
  } catch (error) {
    console.error("Error during signup:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/login", async (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ error: "All fields are required" });
  }

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ error: "Invalid email or password" });
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(400).json({ error: "Invalid email or password" });
    }

    const token = jwt.sign({ userId: user._id }, "loggers123", {
      expiresIn: "1h",
    });

    res.json({ message: "Login successful", token, user });
  } catch (error) {
    console.error("Error during login:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});




app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
