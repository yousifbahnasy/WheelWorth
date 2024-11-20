# Car Price Prediction

This project utilizes machine learning techniques to predict car prices based on a set of features.

## MLFLOW PROJECT
``` bash
# Get conda channels
conda config --show channels

# Build a MLFlow project, if you use one entry point with name (main)
mlflow run . --experiment-name <exp-name> # here it is {CarPrices-processing}

# If you have multiple entry points
mlflow run -e forest . --experiment-name churn-detection
mlflow run -e logistic . --experiment-name churn-detection
mlflow run -e xgboost . --experiment-name churn-detection

# If you want some params instead of default values

mlflow run -e logistic . --experiment-name churn-detection -P c=3.5 -P p="l2"
mlflow run -e xgboost . --experiment-name churn-detection -P n=250 -P lr=0.15 -P d=22

```

```
## MLFLOW Models
``` bash
# serve the model via REST
mlflow models serve -m "path" --port 8000 --env-manager=local
mlflow models serve -m "file:///C:/Users/ALNOUR/Desktop/MLPROJECT/mlruns/204090361731620430/8b1fd11af10d4729a4a14c7b49da0715/artifacts/RandomForestClassifier/balanced_data" --port 8000 --env-manager=local

# it will open in this link
http://localhost:8000/invocations
```
# exmaple of data to be sent
{
    "dataframe_split": {
        "columns": [
            "Make",
            "Model",
            "Year",
            "Engine Fuel Type",
            "Engine HP",
            "Engine Cylinders",
            "Transmission",
            "Drive Mode",
            "Number of Doors",
            "Vehicle Size",
            "Vehicle Style",
            "highway MPG",
            "city mpg",
            "Popularity",
            "Price"
        ],
        "data": [
            ["BMW", "1 Series M", 2011, "premium unleaded (required)", 335.0, 6.0, "MANUAL", "rear wheel drive", 2.0, "Compact", "Coupe", 26, 19, 3916, 46135],
            ["BMW", "1 Series", 2011, "premium unleaded (required)", 300.0, 6.0, "MANUAL", "rear wheel drive", 2.0, "Compact", "Convertible", 28, 19, 3916, 40650],
            ["BMW", "1 Series", 2011, "premium unleaded (required)", 300.0, 6.0, "MANUAL", "rear wheel drive", 2.0, "Compact", "Coupe", 28, 20, 3916, 36350]
        ]
    }
}

from mlflow.models import validate_serving_input

model_uri = 'runs:/8b1fd11af10d4729a4a14c7b49da0715/RandomForestClassifier/balanced_data'

# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'
serving_payload = """{
  "dataframe_split": {
    "columns": [
      "Make",
      "Model",
      "Year",
      "Engine Fuel Type",
      "HP",
      "Cylinders",
      "Transmission",
      "Drive Mode",
      "Number of Doors",
      "Vehicle Size",
      "Vehicle Style",
      "MPG-H",
      "MPG-C",
      "Popularity",
      "Age_of_Car"
    ],
    "data": [
      [
        3,
        84,
        2017,
        2,
        -0.269917351364784,
        -0.9159772819170414,
        3,
        0,
        -1.6299675173478536,
        2,
        8,
        0.6050704377378582,
        0.3634796924835182,
        3105,
        7
      ]
    ]
  }
}"""
from mlflow.models import validate_serving_input

model_uri = 'runs:/8b1fd11af10d4729a4a14c7b49da0715/RandomForestClassifier/balanced_data'



# exmaple of data to be sent

{
  "dataframe_split": {
    "columns": [
      "Make", "Model", "Year", "Engine Fuel Type", "HP", "Cylinders",
      "Transmission", "Drive Mode", "Number of Doors", "Vehicle Size",
      "Vehicle Style", "MPG-H", "MPG-C", "Popularity", "Age_of_Car"
    ],
    "data": [
      [3, 84, 2017, 2, -0.269917351364784, -0.9159772819170414, 3, 0, -1.6299675173478536, 2, 8, 0.6050704377378582, 0.3634796924835182, 3105, 7],
      [1, 50, 2015, 1, 0.5543456721295187, -0.1223457125374189, 2, 1, -0.3247981765237643, 1, 7, 0.4584792853722914, 0.2893649876349128, 4872, 9],
      [2, 102, 2019, 3, 0.8796456781947354, 1.245789614237891, 0, 1, 0.2489172356235411, 2, 6, 0.3297987652368492, 0.4124983658295412, 2789, 5],
      [4, 75, 2016, 0, 1.0759784395178492, 0.5234891784512945, 1, 0, -0.8372541789217482, 1, 9, 0.5491234981273625, 0.3859765282736549, 3201, 8]
    ]
  }
}
