# Red Neuronal Simplificada

## 📊 Paso 1: Limpieza
```python
df = df.drop_duplicates()  # Elimina filas iguales
```

## 📊 Paso 2: Normalización
```python
scaler = MinMaxScaler()
df[['Horas']] = scaler.fit_transform(df[['Horas']])
```
Convierte valores grandes a rango 0-1 para que la red aprenda mejor.

## 🧠 Paso 3: Red Neuronal

**Estructura:**
```
Entrada (Horas, Asistencia) → Red → Salida (0 o 1)
```

**Pesos (w):** Números que se ajustan durante el entrenamiento

**Sigmoid:** Función que convierte cualquier número a 0-1

## 🚀 Paso 4: Entrenar
- Repite 100 veces
- Calcula error
- Ajusta pesos

## ✅ Resultado
```
Accuracy: 100%
```
