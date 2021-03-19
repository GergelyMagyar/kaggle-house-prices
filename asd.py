import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestRegressor
import torch
from torch import nn
from torch import optim

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


# reading data from csv
raw_data = pd.read_csv("./sources/train.csv")


def clean_data(raw_data, transformer, only_test = False):
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


    # Text data

    text_values = x.select_dtypes(exclude=['int64','float64'])
    text_values_columns = text_values.columns

    new_columns = []
    for column in text_values_columns:
        new_columns.append(pd.get_dummies(x[column]).astype(float))
    x = x.select_dtypes(include=['int64', 'float64']).astype(float)
    new_columns.insert(0, x)
    x = pd.concat(new_columns, axis=1)
    #print(x.head(3))
    #print(len(x.columns))


    if(not only_test):
        # splitting to test and dev sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)


    # Normalizing

    if(only_test):
        transformer.fit(x)
    else:
        transformer.fit_transform(x_train)
        transformer.fit(x_test)

    if(only_test):
        return x, None, y, None, None
    else:
        return x_train, x_test, y_train, y_test, transformer



x_train, x_test, y_train, y_test, transformer = clean_data(raw_data, Normalizer(), False);



# TRAIN MODEL

try_linear = False
try_tree = False

print("Min value:",y_train["SalePrice"].min())
print("Max value:",y_train["SalePrice"].max())
print("Mean value:",y_train["SalePrice"].mean())

def display_scores(name, scores):
    print(name,"cross validation:")
    print("\t" + name,"error mean:",scores.mean())
    print("\t" + name, "error standard deviation:", scores.std())


# Linear Regression

if try_linear:
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    lin_pred = lin_reg.predict(x_train)
    lin_mse = mean_squared_error(y_train, lin_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("Linear Regression train RMSE:",lin_rmse)

    lin_pred = lin_reg.predict(x_test)
    lin_mse = mean_squared_error(y_test, lin_pred)
    lin_rmse = np.sqrt(lin_mse)
    print("Linear Regression test RMSE:",lin_rmse)

    lin_scores = cross_val_score(lin_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)

    display_scores("Linear Regression",lin_rmse_scores)


# Decision Tree

if try_tree:
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(x_train, y_train)

    tree_pred = tree_reg.predict(x_train)
    tree_mse = mean_squared_error(y_train, tree_pred)
    tree_rmse = np.sqrt(tree_mse)
    print("Decision Tree train RMSE:",tree_rmse)

    tree_pred = tree_reg.predict(x_test)
    tree_mse = mean_squared_error(y_test, tree_pred)
    tree_rmse = np.sqrt(tree_mse)
    print("Decision Tree test RMSE:",tree_rmse)

    tree_scores = cross_val_score(tree_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)

    display_scores("Decision Tree",tree_rmse_scores)


# Neural Network

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    dev = "cpu"

learning_rate = 0.1
n_features = len(x_train.columns)
print(n_features)
model = nn.Sequential(
    nn.Linear(n_features, 250),
    nn.Sigmoid(),
    nn.Linear(250, 100),
    nn.Sigmoid(),
    nn.Linear(100, 50),
    nn.Sigmoid(),
    nn.Linear(50, 10),
    nn.Sigmoid(),
    nn.Linear(10, 1)
).to(device)
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
decay_rate = 0.985
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

x_train_t = torch.from_numpy(x_train.values.astype(np.float32)).to(device)
y_train_t = torch.from_numpy(y_train.values.astype(np.float32)).to(device)
num_epocs = 1000
for epoch in range(num_epocs):
    y_hat = model(x_train_t)
    l = torch.sqrt(loss(y_hat, y_train_t))
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    lr_scheduler.step()

    if (epoch + 1) % 8 == 0:
        print(f'epoch {epoch + 1}/{num_epocs}, loss {l:.4f}')

x_test_t = torch.from_numpy(x_test.values.astype(np.float32)).to(device)
y_test_t = torch.from_numpy(y_test.values.astype(np.float32)).to(device)
with torch.no_grad():
    y_hat = model(x_test_t)
    l = torch.sqrt(loss(y_hat, y_test_t))
    print(l)
