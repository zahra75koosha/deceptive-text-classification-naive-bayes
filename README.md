# Deceptive Text Classification using Naive Bayes

A Python implementation of a Naive Bayes text classifier for distinguishing between **truthful** and **deceptive** statements.

## Overview

This project implements a Naive Bayes classifier from scratch and evaluates its performance on the **Deception Dataset**, which contains 196 English statements about the best friend, equally divided into truthful and deceptive texts.

The classifier estimates word probabilities for each class and predicts the class of unseen statements based on their calculated probabilities.

## Dataset

- **Dataset:** Deception Dataset
- **Total samples:** 196
- **Classes:** Truthful and Deceptive
- **Language:** English

The class label is encoded in the filename of each text file.

## Methodology

The project includes the following steps:

1. Load and preprocess the text data.
2. Tokenize the statements and extract word frequencies.
3. Build a Naive Bayes classifier.
4. Apply **One-Add (Laplace) Smoothing** to avoid zero probabilities.
5. Classify unseen statements as either truthful or deceptive.
6. Evaluate the classifier under different preprocessing configurations.

### Preprocessing Configurations

The classifier is evaluated under four configurations:

- Original text
- Stopword removal
- Stemming
- Stopword removal + stemming

For stemming, the **Snowball Stemmer** is used.

## Implementation

The main components include:

- `trainNaiveBayes`: trains the classifier and calculates class and word probabilities.
- `testNaiveBayes`: predicts the class of an unseen statement.
- Text preprocessing using tokenization, stopword removal, and stemming.
- Accuracy evaluation for different preprocessing configurations.

## Technologies

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn

## Evaluation

The program calculates classification accuracy for each preprocessing configuration and saves the results to an output file.

## Purpose

This project was developed as part of a **Text Classification (Naive Bayes)** assignment and demonstrates the implementation of a probabilistic NLP classifier, text preprocessing, feature extraction, and model evaluation.
