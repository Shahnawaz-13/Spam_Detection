## SPAM DETECTION
Spam detection is a machine learning process that uses Natural Language Processing (NLP) to automatically categorize incoming messages as either "Spam" (unsolicited/harmful) or "Ham" (legitimate).

## 📌 Project Essence (About Section)
* This project implements a Multinomial Naive Bayes classifier to identify unsolicited spam messages with high precision.
* Leverages the probabilistic independence assumption of Bayes' Theorem to perform efficient text classification on high-dimensional data.
* Designed to handle the discrete word counts typical of SMS and email datasets.

## 🛠️ The Technical Workflow
* Utilizes Natural Language Processing (NLP) techniques, including tokenization, stop-word removal, and Lemmatization to clean raw text.
* Converts text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) to weigh the importance of specific 'spammy' keywords.
* The model calculates the posterior probability of a message being 'Spam' vs 'Ham' based on the frequency of words in the training corpus.

## 📈 Key Performance Metrics
* Achieved an accuracy of [ 0.98162475] on the test dataset.
* Prioritized Precision over Recall to ensure that important 'Ham' messages are rarely misclassified as Spam (minimizing False Positives).

## 💻 Implementation Highlights
Developed using the scikit-learn library for model training and NLTK for linguistic preprocessing.
The script includes a real-time prediction function that allows users to input custom strings for instant classification.

## 💡 Why Naive Bayes?
Chosen for its computational efficiency and ability to perform well even with limited training data.
Particularly effective for text-based features where the feature space (vocabulary) is large.

## DEMO URL
https://sms-email-spamdetection.streamlit.app/
