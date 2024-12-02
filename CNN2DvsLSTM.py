import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Cargar y preprocesar los datos
data = pd.read_csv('preprocessed_data.csv')

# Eliminar la columna de índice (ID)
data = data.drop(columns=['Unnamed: 0'], errors='ignore')

# Seleccionar características (X) y objetivo (y)
X = data[['Radiacion', 'Temperatura', 'Temperatura panel']].values
y = data['Potencia'].values

# Escalar los datos
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled,
                                                    test_size=0.2, random_state=42)

# Redimensionar datos para CNN2D y LSTM
X_train_cnn2d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], 1))
X_test_cnn2d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], 1))

X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Función para crear la red CNN 2D
def create_cnn2d():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(1, 2), activation='relu',
                     input_shape=(1, X_train.shape[1], 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    return model

# Función para crear la red LSTM
def create_lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, X_train.shape[1])))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    return model

# Crear los modelos
models = {
    "CNN2D": create_cnn2d(),
    "LSTM": create_lstm()
}

# Entrenar y evaluar los modelos
results = {}
histories = {}
for name, model in models.items():
    print(f"Training {name} model...")
    if name == "CNN2D":
        history = model.fit(
            X_train_cnn2d, y_train,
            epochs=50, batch_size=32,
            validation_split=0.2, verbose=0
        )
        y_pred = model.predict(X_test_cnn2d)
    elif name == "LSTM":
        history = model.fit(
            X_train_lstm, y_train,
            epochs=50, batch_size=32,
            validation_split=0.2, verbose=0
        )
        y_pred = model.predict(X_test_lstm)

    # Guardar el historial de entrenamiento
    histories[name] = history

    # Desescalar predicciones
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_test_rescaled = scaler_y.inverse_transform(y_test)

    # Calcular métricas
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    results[name] = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'y_pred': y_pred_rescaled,
        'y_test': y_test_rescaled
    }

# Mostrar los resultados
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R2: {metrics['R2']:.4f}")
    print("-" * 30)

# Gráfico comparativo de las curvas de pérdida
plt.figure(figsize=(12, 6))
for name, history in histories.items():
    plt.plot(history.history['loss'], label=f'{name} Training Loss')
    plt.plot(history.history['val_loss'], label=f'{name} Validation Loss')

plt.title('Curvas de Pérdida de Entrenamiento y Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Loss)')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de valores reales vs. predichos para ambos modelos
plt.figure(figsize=(12, 6))
for name, metrics in results.items():
    plt.scatter(metrics['y_test'], metrics['y_pred'], label=f'{name} Predictions', alpha=0.5)

plt.plot([min(y_test_rescaled), max(y_test_rescaled)],
         [min(y_test_rescaled), max(y_test_rescaled)],
         color='red', linestyle='--', label='Perfect Prediction')

plt.title('Valores Reales vs. Predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.legend()
plt.grid(True)
plt.show()
