## Setting up a Python Virtual Environment and Generating a Requirements.txt File

Before we dive into the documentation for the house price prediction app, let's first set up a Python virtual environment and generate a requirements.txt file.

### Creating a Virtual Environment

1. Open your terminal or command prompt.
2. Navigate to the directory where you want to create your project.
3. Run the following command to create a new virtual environment:

   ```
   python -m venv env
   ```

   This will create a new directory called `env` in your project directory, which will contain the Python interpreter and all the packages you install.

4. Activate the virtual environment:
   - On Windows, run:
     ```
     env\Scripts\activate
     ```
   - On macOS or Linux, run:
     ```
     source env/bin/activate
     ```

### Generating a Requirements.txt File

1. With your virtual environment activated, install the necessary packages for your project:
   ```
   pip install streamlit scikit-learn pandas numpy matplotlib
   ```
2. Once you've installed all the required packages, generate a `requirements.txt` file by running:
   ```
   pip freeze > requirements.txt
   ```
   This will create a `requirements.txt` file in your project directory, which contains a list of all the installed packages and their versions.

Now that you have your virtual environment set up and a `requirements.txt` file generated, let's move on to the documentation for the house price prediction app.

Based on the search results, here is how to run the Streamlit project:

## Running the Streamlit Application

To run the Streamlit application, execute the following command in the terminal:

```
streamlit run house-ml-app.py
```

Replace `house-ml-app.py` with the name of your Python script if it's different.

The key steps are:

1. Open your terminal or command prompt.
2. Navigate to the directory where your Streamlit app code is located.
3. Run the following command:
   ```
   streamlit run house-ml-app.py
   ```
   This will start the Streamlit server and open your app in your default web browser.


## House Price Prediction API with Streamlit and Random Forest

This project provides a user-friendly web application for predicting house prices using Streamlit and a Random Forest model. The application allows users to input specific features of a house and returns the predicted price based on the given input.

### Choosing the Model

#### Comparing Regression Models

The code sample in the `house_predict.ipynb` file demonstrates the comparison of different regression models using the scikit-learn (sklearn) library in Python.

##### Description

The code imports several machine learning models from the sklearn library, including `LinearRegression`, `RandomForestRegressor`, and `KNeighborsRegressor`. It then creates a list of these models and iterates through them, fitting each model to a training dataset (`Xtrain`, `Ytrain`) and evaluating its performance on a test dataset (`Xtest`, `Ytest`) using the R-squared (R^2) score.

The results of the model evaluation are stored in a dictionary, which is then converted into a Pandas DataFrame. Finally, a bar plot is created to visualize the R^2 scores for each model.

##### Features

- Comparison of multiple regression models
- Evaluation of model performance using R-squared score
- Visualization of model performance using a bar plot

##### Usage

To use this code, you will need to have the following libraries installed:

- `sklearn`
- `numpy`
- `matplotlib`
- `pandas`

You will also need to have access to your training and test datasets, which should be stored in the variables `Xtrain`, `Ytrain`, `Xtest`, and `Ytest`.

Once you have everything set up, you can run the code to compare the performance of the different regression models and visualize the results.

### API Interface With RandomForestRegressor and Streamlit

The Streamlit application, `house-ml-app.py`, is designed to provide a user-friendly interface for predicting house prices based on user input.

#### Requirements

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas
- NumPy

#### Installation

You can install the required libraries using pip:

```
pip install streamlit scikit-learn pandas numpy
```

#### Model Training

The Random Forest model is trained on a dataset containing various features related to house prices. The model is trained using the Scikit-learn library.

```python
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest model
rf = RandomForestRegressor(n_estimators=100, max_features='sqrt')

# Fit the model to the training data
rf.fit(X_train, y_train)
```

#### Streamlit Application

The Streamlit application allows users to input various features of a house, and the application will predict the house price based on the trained Random Forest model.

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest model
rf = RandomForestRegressor(n_estimators=100, max_features='sqrt')

# Load the trained model
rf.fit(X_train, y_train)

st.write("""
    # House Price Prediction App
""")

# User input parameters
LotArea = st.sidebar.slider('LotArea', 5000, 20000, 10000)
OverallCond = st.sidebar.slider('OverallCond', 5, 10, 7)
YearBuilt = st.sidebar.slider('YearBuilt', 1900, 2022, 1980)
TotalBsmtSF = st.sidebar.slider('TotalBsmtSF', 0, 3000, 1000)

# Create a DataFrame with user input
df = pd.DataFrame({'LotArea': [LotArea],
                   'OverallCond': [OverallCond],
                   'YearBuilt': [YearBuilt],
                   'TotalBsmtSF': [TotalBsmtSF]})

# Predict the house price
prediction = rf.predict(df)

st.subheader("Prediction")
st.write(prediction[0])

st.subheader("Prediction Probability")
# You can add the probability calculation here
```

#### Running the Application

To run the Streamlit application, execute the following command in the terminal:

```
streamlit run house-ml-app.py
```

Replace `house-ml-app.py` with the name of your Python script if it's different.

#### Model Evaluation

The Random Forest model is evaluated using the R2 score, which measures the model's performance in predicting the house prices.

```python
from sklearn.metrics import r2_score

# Predict the house prices for the test dataset
test_predictions = rf.predict(X_test)

# Calculate the R2 score
r2 = r2_score(y_test, test_predictions)

print("R2 Score:", r2)
```

Replace `X_train`, `y_train`, and `X_test`, `y_test` with your actual training and testing datasets.