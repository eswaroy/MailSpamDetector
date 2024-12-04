Spam Detection using Random Forest Classifier
📄 Overview
This project is a machine learning-based spam detection system that classifies emails as either "Spam" or "Ham" (not spam). The dataset is preprocessed, and a Random Forest Classifier is used to perform the classification. The project includes oversampling to handle class imbalances, TF-IDF vectorization for text processing, and performance evaluation with metrics and a confusion matrix.

🚀 Features
Preprocessing of text data including handling missing values and label encoding.
Oversampling of minority classes using imblearn's RandomOverSampler.
TF-IDF vectorization for converting text data to numeric features.
Random Forest Classifier for robust classification.
Performance evaluation using:
Classification report (accuracy, precision, recall, and F1 score).
Confusion matrix visualization.
🛠️ Technologies Used
Python: Programming language.
NumPy & Pandas: Data manipulation and analysis.
Scikit-learn: Machine learning algorithms and utilities.
Imbalanced-learn: Oversampling for handling imbalanced datasets.
Matplotlib & Seaborn: Data visualization.
📂 Project Structure
mail_data.csv: The dataset containing email messages and their labels (spam or ham).
spam_detection.py: Main script for preprocessing, training, and evaluating the model.
📊 Dataset
The dataset, mail_data.csv, contains two columns:

Message: The text of the email.
Category: Label indicating whether the email is spam (0) or ham (1).
📋 Requirements
Install the required Python libraries before running the project:


pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
🏃‍♂️ How to Run
Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/spam-detection.git
Navigate to the project directory:

cd spam-detection
Run the script:

python spam_detection.py
📈 Results
Classification Report: Detailed metrics for model performance.
Confusion Matrix: Visual representation of the predictions vs. actual labels.
🔍 Insights
Oversampling effectively balances the dataset for better model training.
TF-IDF vectorization captures important text features.
Random Forest Classifier provides interpretable results and strong performance.
🤝 Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve this project.

⚖️ License
This project is licensed under the MIT License. See the LICENSE file for details.
