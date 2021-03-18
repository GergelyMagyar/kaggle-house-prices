import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# reading data from csv
raw_data = pd.read_csv("./sources/train.csv")


# dealing with NaN values
#print(raw_data.isnull().sum())

raw_data["LotFrontage"] = raw_data["LotFrontage"].fillna(raw_data["LotFrontage"].mean())
raw_data["Alley"] = raw_data["Alley"].fillna("NoAlley")
raw_data["BsmtQual"] = raw_data["BsmtQual"].fillna("NoAlley")
raw_data["BsmtCond"] = raw_data["Alley"].fillna("NoBasement")
raw_data["BsmtExposure"] = raw_data["BsmtExposure"].fillna("NoBasement")
raw_data["BsmtFinType1"] = raw_data["BsmtFinType1"].fillna("NoBasement")
raw_data["BsmtFinType2"] = raw_data["BsmtFinType2"].fillna("NoBasement")
raw_data["FireplaceQu"] = raw_data["FireplaceQu"].fillna("NoFireplace")
raw_data["GarageType"] = raw_data["GarageType"].fillna("NoGarage")
raw_data["GarageYrBlt"] = raw_data["GarageYrBlt"].fillna(raw_data["GarageYrBlt"].mean())
raw_data["GarageFinish"] = raw_data["GarageFinish"].fillna("NoGarage")
raw_data["GarageQual"] = raw_data["GarageQual"].fillna("NoGarage")
raw_data["GarageCond"] = raw_data["GarageCond"].fillna("NoGarage")
raw_data["PoolQC"] = raw_data["PoolQC"].fillna("NoPool")
raw_data["Fence"] = raw_data["Fence"].fillna("NoFence")
raw_data["MiscFeature"] = raw_data["MiscFeature"].fillna("NoMisk")

raw_data = raw_data.dropna(axis=0)

#print(raw_data.isnull().sum().any())


# splitting data to label and features
y = raw_data[["SalePrice"]]
x = raw_data.drop(["Id","SalePrice"], axis=1)


# splitting to test and dev sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)


# correlations

train = x_train.copy()
train["SalePrice"] = y
correlation = train.corr()
#print(correlation["SalePrice"].sort_values(ascending=False))


# Text data

text_values = x_train.select_dtypes(exclude=['int64','float64'])
text_values_columns = text_values.columns

new_columns = []
for column in text_values_columns:
    new_columns.append(pd.get_dummies(x_train[column]).astype(float))
x_train = x_train.select_dtypes(include=['int64', 'float64']).astype(float)
new_columns.insert(0, x_train)
x_train = pd.concat(new_columns, axis=1)
#print(x_train.head(3))
#print(len(x_train.columns))



# TRAIN MODEL

print("Min value:",y_train["SalePrice"].min())
print("Max value:",y_train["SalePrice"].max())
print("Mean value:",y_train["SalePrice"].mean())

def display_scores(name, scores):
    print(name,"cross validation:")
    print("\t" + name,"error mean:",scores.mean())
    print("\t" + name, "error standard deviation:", scores.std())


# Linear Regression

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

lin_pred = lin_reg.predict(x_train)
lin_mse = mean_squared_error(y_train, lin_pred)
lin_rmse = np.sqrt(lin_mse)
print("Linear Regression RMSE:",lin_rmse)

lin_scores = cross_val_score(lin_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

display_scores("Linear Regression",lin_rmse_scores)


# Decision Tree

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)

tree_pred = tree_reg.predict(x_train)
tree_mse = mean_squared_error(y_train, tree_pred)
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree RMSE:",tree_rmse)

tree_scores = cross_val_score(tree_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

display_scores("Decision Tree",tree_rmse_scores)