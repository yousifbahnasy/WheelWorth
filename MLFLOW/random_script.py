import os
from pyexpat import features
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore')

# Function to clean, process, and log the data
def process_data():
    file_path = os.path.join(os.getcwd(), 'cardataset.csv')

    with mlflow.start_run():
        # Load the dataset
        df = pd.read_csv(file_path)

        # Rename the columns according to the new requirements
        df = df.rename(columns={
            "Engine HP": "HP",
            "Engine Cylinders": "Cylinders",
            "Transmission Type": "Transmission",
            "Driven_Wheels": "Drive Mode",
            "highway MPG": "MPG-H",
            "city mpg": "MPG-C",
            "MSRP": "Price"
        })

        # Handle missing values
        df['Engine Fuel Type'] = df['Engine Fuel Type'].fillna(df['Engine Fuel Type'].mode()[0])
        df['Market Category'] = df['Market Category'].fillna('Unknown')
        df['HP'] = df['HP'].fillna(df['HP'].mean())
        df['Cylinders'] = df['Cylinders'].fillna(df['Cylinders'].mean())
        df['Number of Doors'] = df['Number of Doors'].fillna(df['Number of Doors'].mean())

        # Drop the 'Market Category' column
        df = df.drop(columns=['Market Category'], errors='ignore')

        # Normalize numerical features
        scaler = StandardScaler()
        df[['HP', 'Cylinders', 'Number of Doors', 'MPG-H', 'MPG-C']] = scaler.fit_transform(
            df[['HP', 'Cylinders', 'Number of Doors', 'MPG-H', 'MPG-C']]
        )

        # Encode categorical variables
        label_encoders = {}
        for column in ['Make', 'Model', 'Engine Fuel Type', 'Transmission', 'Drive Mode', 
                       'Vehicle Size', 'Vehicle Style']:
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

        # Calculate the age of the car
        current_year = 2024
        df['Age_of_Car'] = current_year - df['Year']

        # Define features and target
        X = df.drop(columns=['Price'], axis=1)
        median_price = df['Price'].median()
        y = (df['Price'] > median_price).astype(int)  # Binary target variable

        # Split to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45)

        return X_train, X_test, y_train, y_test

## --------------------- Modeling ---------------------------- ##
def train_model(X_train, y_train, X_test, y_test, plot_name, n, d):
    mlflow.set_experiment('CarPrices-processing')
    with mlflow.start_run() as run:
        mlflow.set_tag('classifier', 'random_forest')

        # Train Random Forest model with given n and d
        clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=45)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

        # metrics
        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)

        # Create an example input for the model
        input_example = X_train[0:1]  # Get the first row as an example

        # Log params, metrics, and model 
        mlflow.log_params({'n_estimators': n, 'max_depth': d})
        mlflow.log_metrics({'accuracy': acc_test, 'f1-score': f1_test})

        # Log the model with input example
        mlflow.sklearn.log_model(clf, f'{clf.__class__.__name__}/{plot_name}', 
                                  input_example=input_example)

        # Plot confusion matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='d', cmap='Blues')
        plt.title(f'{plot_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save the plot to MLflow
        conf_matrix_fig = plt.gcf()
        mlflow.log_figure(figure=conf_matrix_fig, artifact_file=f'{plot_name}_conf_matrix.png')
        plt.close()

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{plot_name} ROC Curve')
        plt.legend(loc='lower right')

        # Save the plot to MLflow
        roc_curve_fig = plt.gcf()
        mlflow.log_figure(figure=roc_curve_fig, artifact_file=f'{plot_name}_roc_curve.png')
        plt.close()

def main(n: int, d: int):
    # Process the data and obtain the train/test splits
    X_train, X_test, y_train, y_test = process_data()

    # --------------------- Data Balancing ---------------------------- ##
    ros = RandomOverSampler(random_state=45)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

    # Train the model with balanced data
    train_model(X_train=X_train_balanced, y_train=y_train_balanced, X_test=X_test, y_test=y_test,
                plot_name='balanced_data', n=n, d=d)

    # Train the model with original imbalanced data
    train_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                plot_name='imbalanced_data', n=n, d=d)

if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=350)  # Number of estimators
    parser.add_argument('--d', type=int, default=15)  # Maximum depth
    args = parser.parse_args()

    ## Call the main function
    main(n=args.n, d=args.d)
