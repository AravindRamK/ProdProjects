import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
filep = "C:/Users/karav_867vu4n/OneDrive/Desktop/HOUSE_PRICE_DATASET.csv"
data= pd.read_csv(filep)
data=data.iloc[1:]
cleanedfile= "C:/Users/karav_867vu4n/OneDrive/Desktop/Cleaned_House_data.csv"
data.to_csv(cleanedfile,index=False)
X= data[['Square Footage','Number of Bedrooms','Number of Bathrooms','House Age','Distance to City Centre']]
Y=data['Price']
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)
y_pred= model.predict(X_test)
mse= mean_squared_error(y_pred,Y_test)
r2=r2_score(y_pred,Y_test)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
