from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "datasets"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Dataset Loaded Successfully")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(14,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

print("Training Accuracy:", history.history['accuracy'][-1])

loss, accuracy = model.evaluate(val_data)

print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

model.save("ecosystem_cnn_model.h5")

from tensorflow.keras.models import load_model

model = load_model("ecosystem_cnn_model.h5")

print("Model loaded successfully")


from tensorflow.keras.models import Model

feature_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

features = feature_model.predict(val_data)

print("Feature shape:", features.shape)


import numpy as np
from scipy.stats import entropy

entropy_values = []

for feature_vector in features:
    
    prob = np.abs(feature_vector) / np.sum(np.abs(feature_vector))
    
    ent = entropy(prob)
    
    entropy_values.append(ent)

print("Sample entropy values:", entropy_values[:10])


import matplotlib.pyplot as plt

plt.plot(entropy_values)

plt.title("Ecosystem Complexity (Entropy)")

plt.xlabel("Image Index")

plt.ylabel("Entropy")

plt.show()

entropy_array = np.array(entropy_values)

threshold = np.mean(entropy_array) + np.std(entropy_array)

tipping_points = np.where(entropy_array > threshold)[0]

print("Potential tipping points detected at indices:", tipping_points)

plt.plot(entropy_array, label="Entropy")

plt.scatter(
    tipping_points,
    entropy_array[tipping_points],
    color="red",
    label="Tipping Point"
)

plt.title("Ecosystem Tipping Point Detection")

plt.xlabel("Image Index")

plt.ylabel("Entropy")

plt.legend()

plt.savefig("tipping_point_result.png")

plt.show()


from sklearn.preprocessing import MinMaxScaler

entropy_series = np.array(entropy_values).reshape(-1,1)

scaler = MinMaxScaler()

entropy_scaled = scaler.fit_transform(entropy_series)

sequence_length = 10

X = []

y = []

for i in range(len(entropy_scaled) - sequence_length):
    
    X.append(entropy_scaled[i:i+sequence_length])
    
    y.append(entropy_scaled[i+sequence_length])

X = np.array(X)

y = np.array(y)

print("Sequence shape:", X.shape)


from tensorflow.keras.layers import LSTM

lstm_model = Sequential()

lstm_model.add(
    LSTM(
        50,
        activation='relu',
        input_shape=(sequence_length,1)
    )
)

lstm_model.add(Dense(1))

lstm_model.compile(
    optimizer='adam',
    loss='mse'
)


history_lstm = lstm_model.fit(
    X,
    y,
    epochs=10,
    batch_size=32
)

future_steps = 20

last_sequence = X[-1]

future_predictions = []

for _ in range(future_steps):
    
    pred = lstm_model.predict(
        last_sequence.reshape(1,sequence_length,1)
    )
    
    future_predictions.append(pred[0][0])
    
    last_sequence = np.append(last_sequence[1:], pred, axis=0)

future_predictions = np.array(future_predictions).reshape(-1,1)

future_predictions = scaler.inverse_transform(future_predictions)

print("Future predicted entropy:", future_predictions[:10])

plt.figure(figsize=(10,5))

plt.plot(entropy_series, label="Original Entropy")

future_indices = range(
    len(entropy_series),
    len(entropy_series) + future_steps
)

plt.plot(
    future_indices,
    future_predictions,
    'ro--',   # red circles with dashed line
    label="Predicted Future Entropy"
)

plt.title("Future Ecosystem Entropy Prediction")

plt.xlabel("Time")

plt.ylabel("Entropy")

plt.legend()

plt.show()

import pandas as pd

entropy_series_pd = pd.Series(entropy_values)

rolling_variance = entropy_series_pd.rolling(window=50).var()

plt.figure(figsize=(10,5))

plt.plot(entropy_series_pd, label="Entropy")

plt.plot(rolling_variance, color='orange', label="Rolling Variance")

plt.title("Early Warning Signal: Variance Increase")

plt.xlabel("Time")

plt.ylabel("Value")

plt.legend()

plt.show()

autocorr = entropy_series_pd.autocorr(lag=1)

print("Lag-1 Autocorrelation:", autocorr)

plt.figure(figsize=(10,5))

plt.plot(rolling_variance, label="Rolling Variance")

plt.title("Variance-Based Early Warning Signal")

plt.xlabel("Time")

plt.ylabel("Variance")

plt.legend()

plt.show()

import numpy as np
import pandas as pd

def multiscale_entropy(series, max_scale=10):

    mse_values = []

    series = np.array(series)

    for scale in range(1, max_scale+1):

        coarse_grained = [
            np.mean(series[i:i+scale])
            for i in range(0, len(series)-scale+1, scale)
        ]

        coarse_grained = np.array(coarse_grained)

        prob = np.abs(coarse_grained) / np.sum(np.abs(coarse_grained))

        ent = -np.sum(prob * np.log(prob))

        mse_values.append(ent)

    return mse_values


mse = multiscale_entropy(entropy_values)

print("Multiscale entropy:", mse)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(range(1,len(mse)+1), mse, marker='o')

plt.title("Multiscale Entropy Analysis")

plt.xlabel("Scale")

plt.ylabel("Entropy")

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

entropy_series_pd = pd.Series(entropy_values)

rolling_variance = entropy_series_pd.rolling(window=50).var()

plt.figure(figsize=(10,5))

plt.plot(entropy_series_pd, label="Entropy")

plt.plot(rolling_variance, color='orange', label="Rolling Variance")

plt.title("Early Warning Signal: Variance Increase")

plt.xlabel("Time")

plt.ylabel("Value")

plt.legend()

plt.show()
lag1_autocorr = entropy_series_pd.autocorr(lag=1)

print("Lag-1 Autocorrelation:", lag1_autocorr)

from sklearn.model_selection import train_test_split

X_flat = X.reshape(X.shape[0], X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X_flat,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rf_model = RandomForestRegressor()

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("Random Forest RMSE:", rf_rmse)

from sklearn.svm import SVR

svr_model = SVR()

svr_model.fit(X_train, y_train.ravel())

svr_pred = svr_model.predict(X_test)

svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))

print("SVR RMSE:", svr_rmse)

lstm_pred = lstm_model.predict(X_test.reshape(X_test.shape[0], sequence_length,1))

lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))

print("LSTM RMSE:", lstm_rmse)

print("\nModel Comparison")

print("---------------------------")

print("Random Forest RMSE:", rf_rmse)

print("SVR RMSE:", svr_rmse)

print("LSTM RMSE:", lstm_rmse)

rolling_autocorr = entropy_series_pd.rolling(window=50).apply(
    lambda x: x.autocorr(lag=1),
    raw=False
)

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(entropy_series_pd)
plt.title("Entropy (System Complexity)")

plt.subplot(3,1,2)
plt.plot(rolling_variance, color='orange')
plt.title("Rolling Variance (Instability Indicator)")

plt.subplot(3,1,3)
plt.plot(rolling_autocorr, color='green')
plt.title("Rolling Lag-1 Autocorrelation (Critical Slowing Down)")

plt.tight_layout()

plt.show()
