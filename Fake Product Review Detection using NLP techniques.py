#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

#load dataset
data = pd.read_csv('d:/python_ka_chilla/AI Projects/Fake Product Review Detection using NLP/product_reviews.csv')  

# display first few rows of the dataset
print(data.head())

#define features and labels
X = data['text_']
y = data['label']  # 'fake' or 'real'

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

#train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

#make predictions on the test set
y_pred = model.predict(X_test_vectors)

#evaluate the model
print("Model Evaluation Results:")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

