import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sb

sb.set_style('whitegrid')
np.random.seed(1)

# Loading Data
data = pd.read_csv("insurance.csv")


# display(data)

def plot_data(data):
    for i in data.columns:
        if i != 'bmi' and i != 'charges':
            dt = data.groupby(i).size().sort_values(ascending=False)
            plt.figure(figsize=(5, 5))
            dt.plot(kind='bar', title=i, ylabel='Times', xlabel=i)
        elif i == 'bmi':
            plt.figure()
            sb.regplot(x='bmi', y='charges', data=data).set(title='bmi and charges')
    return None


plot_data(data)


# Data PreProcessing
def convert_categorical_to_numerical(data, unique_val=10):
    obj_df = data.select_dtypes(include=[object])
    unique_list = [f for f in obj_df.columns if obj_df[f].unique().shape[0] >= unique_val]
    obj_df_sel = obj_df.drop(columns=unique_list)
    for col in obj_df_sel.columns:
        data[col] = pd.factorize(data[col])[0]
    return data


data = convert_categorical_to_numerical(data=data)
print("conversion successfully")
display(data)

# Correlation Plot
plt.figure(figsize=(25, 10))
sb.heatmap(data.corr(), annot=True)
display(data.describe().T)

# Splitting Data into Train and Test data
X = data.iloc[:, :6]
y = data['charges']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
data.to_csv('finaldata.csv', index=False)
x_test.to_csv('testing.csv', index=False)


# Creating a function to evaluate model performance
def model_performance(model, model_name, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    training_score = np.round(model.score(x_train, y_train), 3)
    testing_score = np.round(model.score(x_test, y_test), 3)

    mse_training = np.round(mean_squared_error(y_train, y_train_pred), 3)
    mse_testing = np.round(mean_squared_error(y_test, y_test_pred), 3)

    mae_training = np.round(mean_absolute_error(y_train, y_train_pred), 3)
    mae_testing = np.round(mean_absolute_error(y_test, y_test_pred), 3)

    r2_training = np.round(r2_score(y_train, y_train_pred), 3)
    r2_testing = np.round(r2_score(y_test, y_test_pred), 3)

    print("Model Performance for:", model_name)
    print("")

    print("Training Score:", training_score)
    print("Testing Score:", testing_score)
    print("")

    print("Training Data Mean Squared Error:", mse_training)
    print("Testing Data Mean Squared Error:", mse_testing)
    print("")

    print("Training Data Mean Absolute Error:", mae_training)
    print("Testing Data Mean Absolute Error:", mae_testing)
    print("")

    print("Training Data r2_score:", r2_training)
    print("Testing Data r2_score:", r2_testing)
    print("")

    plt.show()

    return training_score, testing_score, mse_training, mse_testing, mae_training, mae_testing, r2_training, r2_testing


# Creating Machine Learning Models (LinearRegression, RandomForestRegressor, Decision_Tree)
model1 = LinearRegression()
model1.fit(x_train, y_train)
Linear_regression_performance = model_performance(model1, model_name='LinearRegression')

model2 = RandomForestRegressor(max_depth=4, min_samples_split=2, min_samples_leaf=2)
model2.fit(x_train, y_train)
Random_Forest_performance = model_performance(model2, model_name='RandomForestRegressor')

param_grid = {'max_depth': np.arange(1, 10), 'min_samples_split': np.arange(2, 10),
              'min_samples_leaf': np.arange(1, 10)}
grid = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, cv=4)
grid.fit(X, y)
model3 = grid.best_estimator_
model3.fit(x_train, y_train)
Decision_Tree_performance = model_performance(model3, model_name='Decision_Tree')

import pickle

# Saving model to disk
pickle.dump(model1, open('model1LR.pkl', 'wb'))
pickle.load(open('model1LR.pkl', 'rb'))

pickle.dump(model2, open('model2RFR.pkl', 'wb'))
pickle.load(open('model2RFR.pkl', 'rb'))

pickle.dump(model3, open('model3DT.pkl', 'wb'))
pickle.load(open('model3DT.pkl', 'rb'))

model_performance = [["Linear Regression", Linear_regression_performance[0], Linear_regression_performance[1],
                      Linear_regression_performance[2], Linear_regression_performance[3],
                      Linear_regression_performance[4], Linear_regression_performance[5],
                      Linear_regression_performance[6], Linear_regression_performance[7]],
                     ["Random Forest", Random_Forest_performance[0], Random_Forest_performance[1],
                      Random_Forest_performance[2], Random_Forest_performance[3], Random_Forest_performance[4],
                      Random_Forest_performance[5], Random_Forest_performance[6], Random_Forest_performance[7]],
                     ["Decision Tree", Decision_Tree_performance[0], Decision_Tree_performance[1],
                      Decision_Tree_performance[2], Decision_Tree_performance[3], Decision_Tree_performance[4],
                      Decision_Tree_performance[5], Decision_Tree_performance[6], Decision_Tree_performance[7]]]

performance = pd.DataFrame(model_performance,
                           columns=['Model_Name', "Train Score", "Test Score", "Train MSE", 'Test MSE', 'Train MAE',
                                    "Test MAE", "Train R2", "Test R2"])

display(performance)
