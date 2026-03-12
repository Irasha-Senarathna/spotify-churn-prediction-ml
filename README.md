# spotify-churn-prediction-ml

Spotify Churn Prediction (Machine Learning Project)
Project Overview

This project aims to predict whether a Spotify user will churn (leave the service) or stay, using machine learning techniques. By analyzing user listening behavior and account information, the model can identify patterns that indicate potential churn.

The project is implemented using Python and common machine learning libraries within the field of Machine Learning.

Objectives

Perform data preprocessing and cleaning

Explore and prepare the dataset for modeling

Train a Logistic Regression model

Evaluate model performance using classification metrics

Visualize results using plots

Project Structure
spotify-churn-prediction-ml/
│
├── data/
│   └── spotify_churn.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_logistic_regression.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── train_logistic.py
│
├── results/
│   ├── metrics.txt
│   └── plots/
│       └── confusion_matrix_logreg.png
│
├── venv/
├── .gitignore
└── README.md
Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

Jupyter Notebook

These tools help implement algorithms used in Data Science and predictive modeling.

Machine Learning Model
Logistic Regression

The primary model used in this stage is Logistic Regression, which is commonly used for binary classification problems.

Steps performed:

Data preprocessing

Train-test split

Feature scaling

Model training

Prediction

Performance evaluation

Evaluation Metrics

The model performance is evaluated using:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

These metrics help determine how well the model predicts churn.

Visualization

A confusion matrix heatmap is generated to visualize prediction results.

Example output:

results/plots/confusion_matrix_logreg.png
How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/spotify-churn-prediction-ml.git
cd spotify-churn-prediction-ml
2. Create Virtual Environment
python -m venv venv
3. Activate Virtual Environment

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate
4. Install Dependencies
pip install -r requirements.txt
5. Run Notebooks
jupyter notebook

Then open:

notebooks/02_preprocessing.ipynb
notebooks/03_logistic_regression.ipynb
Team Contributions
Role	Task
Student 1	Data preprocessing and Logistic Regression model
Student 2	Random Forest model implementation
Future Improvements

Implement Random Forest model

Perform hyperparameter tuning

Improve class imbalance handling

Add more visualization and analysis
