import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ===== DATOS =====
data = {
    'Horas': [10, 2, 15, 5, 10],
    'Asistencia': [0.8, 0.4, 0.9, 0.5, 0.8],
    'Aprobo': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# ===== LIMPIEZA =====
df = df.drop_duplicates()
print("1. Datos limpios (sin duplicados):")
print(df)

# ===== NORMALIZACION =====
scaler = MinMaxScaler()
df[['Horas']] = scaler.fit_transform(df[['Horas']])
print("\n2. Datos normalizados (0-1):")
print(df)

# ===== RED NEURONAL SIMPLE =====
class Red:
    def __init__(self):
        self.w = np.random.rand(2, 1) * 0.5  # 2 pesos (Horas, Asistencia)
        self.b = 0  # bias
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predecir(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)
    
    def entrenar(self, X, y, epochs=100):
        for _ in range(epochs):
            # Forward
            pred = self.predecir(X)
            
            # Backward (actualizar pesos)
            error = pred - y.reshape(-1, 1)
            self.w -= 0.1 * np.dot(X.T, error) / len(X)
            self.b -= 0.1 * np.mean(error)

# ===== ENTRENAR =====
X = df[['Horas', 'Asistencia']].values
y = df['Aprobo'].values

red = Red()
red.entrenar(X, y)

# ===== PREDICCIONES =====
prediciones = (red.predecir(X) > 0.5).astype(int).flatten()
accuracy = np.mean(prediciones == y)

print(f"\n3. Predicciones:")
print(f"Real:       {y}")
print(f"Predicción: {prediciones}")
print(f"Accuracy:   {accuracy:.0%}")
