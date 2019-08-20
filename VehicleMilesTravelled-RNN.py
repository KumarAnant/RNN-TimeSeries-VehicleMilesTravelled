import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\akumar\\OneDrive\\AmateurWork\\Python\\Udemy\\TimeSeries\\TSA_COURSE_NOTEBOOKS\\Data\\VehicleMilesTraveled.csv', index_col = 'DATE', parse_dates = True)
df.columns = ['Value']
df.head()
df.plot()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['Value'])
result.plot()
plt.show()
result.trend.plot()
plt.show()
result.seasonal.plot()
plt.show()
result.resid.plot()
plt.show()
# Check for any missing value
print(len(df))
df.isna().any()

# Train / Test split
test_length = 24
train = df[:-test_length]
test = df[-test_length:]

# Normalise the values
from sklearn.preprocessing import MinMaxScaler

ScalerModel = MinMaxScaler().fit(train)
train_scaled = ScalerModel.transform(train)
test_scaled = ScalerModel.transform(test)

# Time series generator
from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 12
n_feature = 1

train_generator = TimeseriesGenerator(train_scaled, train_scaled, length = n_input, batch_size=1)
len(train_generator)
len(train)
len(test)
X, y = train_generator[1]

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(units = 250, activation = 'relu', input_shape = (n_input, n_feature)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')

model.summary()

model.fit_generator(train_generator, epochs = 50)
model.history.history.keys()
plt.plot(range(len(model.history.history['loss'])), model.history.history['loss'])
plt.show()

first_eval_bath = train_scaled[-n_input:]
first_eval_batch = first_eval_bath.reshape(1, n_input, n_feature)
model.predict(first_eval_batch)

# Forecast using RNN model
test_predictions = []
# Input for 1st prediction
first_eval_batch = train_scaled[-n_input:]
curr_batch = first_eval_batch.reshape(1, n_input, n_feature)

for i in range(len(test)):
    print(i)
    current_pred = model.predict(curr_batch)[0]
    test_predictions.append(current_pred)
    curr_batch = np.append(curr_batch[:, 1:, :], [[current_pred]], axis= 1)


print(test_predictions)
print(len(test))
print(len(test_predictions))
true_predictions = ScalerModel.inverse_transform(test_predictions)
print(true_predictions)
print(test)
test['Predictions'] = true_predictions
print(test)

test.plot()
plt.show() 

model.save('PredictMilesTraveled.h5')

from sklearn.metrics import mean_squared_error
rsme = np.sqrt(mean_squared_error(test['Value'], test['Predictions']))
print('RMSE: ', rsme)