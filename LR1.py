import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

## import dataset

df = pd.read_csv(r'C:\\Users\\vinic\\Downloads\\Salary_Data.csv')

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

## split test treino = 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state = 0)

### training usando regressor
regressor = LinearRegression()
regressor.fit(x_train, y_train)

## predict the test set
y_pred = regressor.predict(x_test)


# visualizing training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary x Experience (training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

## visualizing the test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue') ## para RL, os resultados derivam de uma funcao unica epor isso n precisa trocar para x_test
plt.title('Salary x Experience (test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

##assessing the model performance
r2 = r2_score (y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f'R2 = {r2}')
print(f'MAE = {mae}')
print(f'MSE = {mse}')
print(f'RMSE = {rmse}')
