import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



# Load and preprocess the dataset
raw_mail_data = pd.read_csv('mail_data.csv')
mail_data = raw_mail_data.fillna('')  # Replace NaN with empty strings
mail_data['Category'] = mail_data['Category'].replace({'spam': 0, 'ham': 1})  # Encode labels

# Split the dataset into training, validation, and testing
train, validate, test = np.split(
    mail_data.sample(frac=1, random_state=42),  # Shuffle data
    [int(0.6 * len(mail_data)), int(0.8 * len(mail_data))]
)

# Define a function to scale and optionally oversample the dataset
def scale_dataset(dataframe, oversample=False):
    X = dataframe['Message'].values  # Extract text data
    y = dataframe['Category'].values  # Extract labels
    
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X.reshape(-1, 1), y)
        X = X.flatten()  # Flatten back to 1D array
    
    return X, y

# Scale datasets and vectorize text data
X_train, Y_train = scale_dataset(train, oversample=True)
X_validate, Y_validate = scale_dataset(validate, oversample=False)
X_test, Y_test = scale_dataset(test, oversample=False)

# Convert text to numeric features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit-transform on training data
X_test_tfidf = vectorizer.transform(X_test)  # Transform test data
X_validate_tfidf = vectorizer.transform(X_validate)  # Transform validation data

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_tfidf, Y_train)

# Predict and evaluate the model
Y_pred = model.predict(X_test_tfidf)
print(classification_report(Y_test, Y_pred))  # Use Y_test as ground truth
# Generate the confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'])
plt.title("Confusion Matrix for Predictions")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
