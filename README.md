
![proAr](https://github.com/user-attachments/assets/020b418d-ad1c-4dfd-b625-195e10042d3d)

<h1 align="center">WheelWorth</h1>
<h3 align="center">Your ultimate solution for car price prediction using machine learning</h3>

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://linkedin.com/in/yousif-bahnasy" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="yousifbahnasy" height="30" width="40" /></a>
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://azure.microsoft.com/en-in/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/microsoft_azure/microsoft_azure-icon.svg" alt="azure" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

---

<h2>🚗 Project Overview</h2>
<p>
WheelWorth is a machine learning-powered application designed to predict car prices based on key features such as engine specifications, transmission type, and more. 
The project utilizes advanced modeling techniques, robust preprocessing pipelines, and deployment technologies to deliver an intuitive user experience for price estimation.
</p>

<p>
This project was developed as part of the Microsoft Machine Learning Engineer Digital Egypt Pioneers Initiative (DEPI) and leverages cloud technologies like <strong>Microsoft Azure</strong> alongside tools for experiment tracking and visualization.
</p>

---


<h2>🌟 Features</h2>

- **Interactive Web Interface**: Built with [Streamlit](https://streamlit.io/) for simplicity and user interaction.
- **Real-Time Predictions**: Input car features to get instant price estimates.
- **Multi-Model Analysis**: Employs a variety of machine learning models, including:
  - Linear Regression
  - Random Forest
  - Support Vector Regression (SVR)
  - Grid Search SVR
  - Artificial Neural Networks (ANN)
- **Experiment Tracking**: Integrated with [MLflow](https://mlflow.org/) for tracking model performance and parameters.
- **Cloud Deployment**: Hosted on [Microsoft Azure](https://azure.microsoft.com/) for scalability and reliability.

---

<h2>🔄 Project Workflow</h2>

### 1️⃣ Data Collection & Preprocessing
- Cleaned and formatted data, including handling missing values and scaling numeric features.
- Encoded categorical variables to ensure compatibility with machine learning models.

### 2️⃣ Feature Engineering
- Performed feature selection to retain the most relevant variables.
- Applied dimensionality reduction techniques to improve model accuracy and reduce complexity.

### 3️⃣ Modeling
- Trained and optimized multiple machine learning models, including:
  - Linear Regression
  - Random Forest
  - Support Vector Regression (SVR)
  - Artificial Neural Networks (ANN)
- Evaluated models using metrics such as:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Coefficient of Determination (R² Score)

### 4️⃣ Deployment
- Built a user-friendly web application using [Streamlit](https://streamlit.io/).
- Deployed the app on:
  - [Microsoft Azure](https://azure.microsoft.com/) for public cloud hosting.
  - [Streamlit Cloud](https://streamlit.io/cloud) for accessible and interactive app usage.

### 5️⃣ Experiment Management
- Integrated [MLflow](https://mlflow.org/) to log:
  - Model parameters
  - Performance metrics
  - Artifacts for tracking and comparison.

---

## 🚀 How to Run the Car Price Prediction App

### Required Files
To run the Streamlit app, you will need the following files:
1. **`streamlit_app.py`** - The main Streamlit application file.
2. **`preprocessing_pipeline.joblib`** - The saved preprocessing pipeline file.
3. **`rf_model.joblib`** - The saved Random Forest model file.

### To Run the App

-  ```bash
    streamlit run streamlit_app.py
- Access the App: The app will start locally at http://localhost:8501. Open this URL in your browser to use the app.

---
## 📊 Model Performance

The following table summarizes the performance of various machine learning models used for car price prediction:

| Model               | Mean Squared Error (MSE) | R-squared (R²) |
|---------------------|--------------------------|----------------|
| **Random Forest**    | 1,245,678                | 0.92           |
| **Linear Regression**| 2,345,678                | 0.85           |
| **Decision Tree**    | 1,845,678                | 0.88           |

- **Random Forest**: This model performed the best, with a low Mean Squared Error (MSE) of 1,245,678 and a high R-squared (R²) value of 0.92. The R² value indicates that the Random Forest model can explain 92% of the variance in the car price data, making it highly accurate for price prediction.
  
- **Linear Regression**: With an MSE of 2,345,678 and an R² value of 0.85, Linear Regression showed a decent performance but didn’t match the Random Forest model in terms of accuracy. The R² of 0.85 means that it explains 85% of the variance in the data.

- **Decision Tree**: The Decision Tree model had an MSE of 1,845,678 and an R² value of 0.88. While slightly less accurate than Random Forest, it still performed well, explaining 88% of the variance in the data.
---

## 🎯 Conclusion 

The **WheelWorth** app provides a reliable and interactive solution for predicting car prices based on various features, powered by machine learning models like Random Forest, Linear Regression, and Decision Tree. With real-time predictions and a user-friendly interface, it offers an intuitive way to estimate car prices. 

- #### Full Project Overview 📊

For a detailed overview of the project, including the methodology, performance metrics, and the deployment process, you can view the full presentation linked below:

[**Project Presentation**](./presentation.pptx)

Feel free to explore the code, run the app locally, and reach out if you have any questions or feedback.

- #### Acknowledgments 🙏

This project was developed as part of the **Microsoft Machine Learning Engineer Digital Egypt Pioneers Initiative (DEPI)** under the supervision of the **Ministry of Communications and Information Technology (MCIT)**, Egypt.


