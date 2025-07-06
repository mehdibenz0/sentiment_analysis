<p align="right">
  <b>Powered by      </b>
</p>

<p align="right">
  <img src="images/techspark_logo.png" alt="TechSpark Academy Logo" width="210"/>
</p>




# Sentiment Analysis Project

In this fun and beginner-friendly machine learning project, you'll build your own text sentiment classifier using Python! The goal? To teach a computer how to understand if a piece of text expresses a positive or negative sentiment — like movie reviews, tweets, or product feedback.

## Project Overview
In this project, you will:

 - Collect and prepare text data: Gather text samples labeled as positive or negative from datasets or other sources. Clean and preprocess the text for machine learning.
 - Train a sentiment analysis model: Use TF-IDF vectorization and classical machine learning algorithms like Support Vector Machines (SVM) to classify sentiments.
 - Test and evaluate the model: Measure how well your model predicts sentiment on unseen data.
 - Predict sentiment on custom input: Enter your own text and see how the model classifies its sentiment in real time!

## Files Description

| File             | Description |
|------------------|-------------|
| `collect_data.py` | Script to download or gather raw text data (e.g., reviews) |
| `utils/data_preprocessing.py` | Functions for cleaning text and vectorizing using TF-IDF |
| `utils/model.py` | Functions to train, tune, and evaluate the machine learning model |
| `utils/predict.py` | 	Functions to predict sentiment on new, custom text |
| `main.py` | Main script to run the entire pipeline interactively. |


##  Your Tasks
 - Run data collection and preprocessing to prepare your dataset.
 - Train the sentiment model with hyperparameter tuning.
 - Evaluate the model’s accuracy.
 - Enter custom text to get live sentiment predictions.

##  Clone this Project
To download all the code and files from this project, run:
```bash
git clone https://github.com/mehdibenz0/sentiment_analysis.git
```
This will create a new folder with all the files you need — including:
 - Python scripts for data collection, preprocessing, training, and prediction
 - This README file
 - Required dependencies listed in requirements.txt
   
##  Project Steps
1. Install requirements
```bash
pip install -r requirements.txt
```

2. Collect or update text data
```bash
python collect_data.py
```
Run this once or whenever you want fresh data:

3. Prepare the data
This step resizes all images to a fixed size and saves them in a format the model can use:
```bash
python main.py
```
The script also trains and evaluates your model.

4. Complete the TODOs in train_model_student.py, then run the following to train your model:
```bash
python train_model_student.py
```
This script will process the images and train a model called object_detector.h5.

5. Run real-time prediction!
```bash
python predict_webcam.py
```
Try holding up different objects in front of the webcam. The program will tell you what it sees!

## Clean-Up Tip
If the model isn't working well, consider:
 - Collecting more data
 - Trying other classical models (e.g., SVM instead of Logistic Regression)

## Learning Goals
 - Understand the basics of text preprocessing for NLP
 - Learn how to vectorize text data using TF-IDF
 - Gain experience training and tuning classical machine learning models for sentiment classification
 - Build an interactive sentiment prediction tool for custom input


## Helpful Hints

 - Ensure your dataset labels are consistent and balanced between positive and negative samples.
 - Preprocess your text thoroughly — remove punctuation, stopwords, and consider lemmatization.
 - Test your model with a variety of sentences, including ambiguous or neutral ones.
 - The more diverse your training data, the better your model will generalize.
