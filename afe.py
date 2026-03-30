import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import entropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
dataset_path = "datasets"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

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
def build_cnn_model():
    model = Sequential()

    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(14,activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
cnn_model = build_cnn_model()

history = cnn_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

cnn_model.save("ecosystem_cnn_model.h5")

print("CNN Training Done")

cnn_model = load_model("ecosystem_cnn_model.h5")

feature_model = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)

features = feature_model.predict(val_data)

print("Feature shape:", features.shape)

def compute_fractal_dimension(data):
    data = np.array(data)
    N = len(data)
    if N < 5:
        return 0

    L = []
    for k in range(1, 4):
        Lk = []
        for m in range(k):
            Lmk = 0
            for i in range(1, int((N - m) / k)):
                Lmk += abs(data[m + i*k] - data[m + (i-1)*k])
            if int((N - m) / k) != 0:
                Lmk = (Lmk * (N - 1)) / (int((N - m) / k) * k)
                Lk.append(Lmk)
        if len(Lk) > 0:
            L.append(np.mean(Lk))

    if len(L) < 2:
        return 0

    L = np.log(L)
    k_vals = np.log(range(1, len(L)+1))
    coeffs = np.polyfit(k_vals, L, 1)

    return -coeffs[0]


def compute_AFE(feature_vector):
    prob = np.abs(feature_vector) / (np.sum(np.abs(feature_vector)) + 1e-10)

    H = entropy(prob)
    var = np.var(feature_vector)
    FD = compute_fractal_dimension(feature_vector)

    mid = len(feature_vector)//2

    H1 = entropy(np.abs(feature_vector[:mid]) / (np.sum(np.abs(feature_vector[:mid]))+1e-10))
    H2 = entropy(np.abs(feature_vector[mid:]) / (np.sum(np.abs(feature_vector[mid:]))+1e-10))

    dH = H2 - H1

    AFE = 0.25*H + 0.25*FD + 0.25*var + 0.25*dH

    return AFE
afe_values = []

for vec in features:
    afe_values.append(compute_AFE(vec))

afe_array = np.array(afe_values)

print("AFE Sample:", afe_array[:10])
threshold = np.mean(afe_array) + 1.5*np.std(afe_array)

tipping_points = np.where(afe_array > threshold)[0]

print("Tipping Points:", tipping_points)
plt.plot(afe_array, label="AFE")

plt.scatter(tipping_points, afe_array[tipping_points], color="red", label="Tipping")

plt.title("AFE-Based Tipping Detection")

plt.legend()

plt.show()
scaler = MinMaxScaler()

series = afe_array.reshape(-1,1)

scaled = scaler.fit_transform(series)

sequence_length = 10

X, y = [], []

for i in range(len(scaled) - sequence_length):
    X.append(scaled[i:i+sequence_length])
    y.append(scaled[i+sequence_length])

X, y = np.array(X), np.array(y)

lstm_model = Sequential()

lstm_model.add(LSTM(50, activation='relu', input_shape=(sequence_length,1)))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.fit(X, y, epochs=10)

future_steps = 20

last_seq = X[-1]

future = []

for _ in range(future_steps):
    pred = lstm_model.predict(last_seq.reshape(1,sequence_length,1))
    future.append(pred[0][0])
    last_seq = np.append(last_seq[1:], pred, axis=0)

future = scaler.inverse_transform(np.array(future).reshape(-1,1))

series_pd = pd.Series(afe_array)

rolling_var = series_pd.rolling(window=50).var()

rolling_autocorr = series_pd.rolling(window=50).apply(lambda x: x.autocorr(lag=1))

plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(series_pd)
plt.title("AFE")

plt.subplot(3,1,2)
plt.plot(rolling_var)
plt.title("Variance")

plt.subplot(3,1,3)
plt.plot(rolling_autocorr)
plt.title("Autocorrelation")

plt.show()
