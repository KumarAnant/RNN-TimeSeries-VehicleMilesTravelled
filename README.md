# RNN-TimeSeries-VehicleMilesTravelled
Plot of Vehicle miles travelled over time

![Miles travelled](./images/1Figure_1.png)

The plot seems to be having seasonal component along with trend and residual components.

![Resdual components](./images/2Figure_1.png)

Trend part of the residual component
![Resdual component of the distribution](./images/3Trend.png)

Seasonal component of the decomposed
![Seasonal component](./images/4Seasonal.png)

Residual part of the distribution
![Residual part of the distribution](./images/5Residual.png)

Newral Network model was devlooped using RNN with Tensorflow and tested with the test data. Following is the test Loss of the model

![Loss of the model](./images/6Loss.png)

Original vs predicted values of the test dataset is platted below. The model seems to be very accurate.
![Prediction vs Original](./images/7Prediction.png)
