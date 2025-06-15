# 📧 Email Spam Detection Model

A Machine Learning project developed as part of a semester project to classify emails as **Spam** or **Not Spam** using natural language processing (NLP) and various classification algorithms.

## 🔍 Project Overview

This project aims to build an efficient spam detection model that can automatically classify incoming emails based on their content. It leverages **Natural Language Processing (NLP)** for text preprocessing and **Machine Learning** for classification.

## 📂 Features

- Preprocessing of raw email data (tokenization, stopword removal, vectorization)
- Multiple ML models trained and evaluated
- Performance comparison of models
- Confusion matrix and classification report for evaluation
- User input testing (optional)
- Jupyter Notebook-based implementation

## 🧠 Machine Learning Models Used

- Multinomial Naive Bayes ✅ *(Best Performing)*
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

## ⚙️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- NLTK (Natural Language Toolkit)
- Jupyter Notebook

## 🛠️ Project Structure

email-spam-detection/
│
├── data/
│ └── spam.csv # Dataset used for training/testing
│
├── notebooks/
│ └── spam_detection.ipynb # Main model implementation
│
├── models/
│ └── trained_model.pkl # (Optional) Serialized model file
│
├── README.md # Project documentation
└── requirements.txt # Python dependencies

## 📊 Results

- **Accuracy:** ~97% with Multinomial Naive Bayes
- **Precision/Recall/F1-score:** High on spam classification
- **Fast and efficient model suitable for real-time deployment**

## 🚀 How to Run

