import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
    # Simple House Price Prediction app
        
         """)

st.sidebar.header('User Input Parameters')


analytic_fields = ['LotArea', 'OverallCond', 'YearBuilt', 'TotalBsmtSF']
def user_input_features():
    LotArea = st.sidebar.slider('LotArea', 5000, 20000, 10000)   #('LotArea', 4.3, 7.9, 5.4)
    OverallCond = st.sidebar.slider('OverallCond', 5, 10, 7)#('OverallCond', 2.0, 4.4, 3.4)
    YearBuilt = st.sidebar.slider('YearBuilt', 1900, 2022, 1980) #('YearBuilt', 1.0, 6.9, 1.3)
    TotalBsmtSF = st.sidebar.slider('TotalBsmtSF', 0, 3000, 1000)
#('TotalBsmtSF', 0.1, 2.5, 0.2)
    data = { 'LotArea' : LotArea,
            'OverallCond' : OverallCond,
            'YearBuilt': YearBuilt,
            'TotalBsmtSF': TotalBsmtSF
    }
    featuers = pd.DataFrame(data, index=[0])
    return featuers
  
df =  user_input_features() 

st.subheader('User Input Parameter')
st.write(df)

data = pd.read_csv("data.csv")

data.head()

data['SalePrice'] = data['SalePrice'].fillna(
  data['SalePrice'].mean())

data = data.dropna()

from sklearn.model_selection import train_test_split

analytic_fields = ['LotArea', 'OverallCond', 'YearBuilt', 'TotalBsmtSF']

X = data[analytic_fields]
Y = data['SalePrice']

model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
model.fit(X, Y)

prediction = model.predict(df)

#Names of the different categories of flowers
#found in our dataset
st.header('Class labels and their corresponding index number')
st.write(data['SalePrice'])

#Prints out the prediction based on the user input
st.subheader("Prediction")
st.write(prediction)

#Give the probability of the accuracy of the prediction
st.subheader("Prediction Probality")

