
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score

st.title("Predicción de calificaciones")
st.write("Regresión lineal múltiple")

df = pd.read_csv("student_exam_scores.csv")

# Variables independientes
X2 = df[['hours_studied', 'sleep_hours',
         'attendance_percent', 'previous_scores']]

# Variable dependiente
y2 = df['exam_score']

# Dividir datos
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.30, random_state=0
)

# Crear y entrenar modelo
LR2 = LinearRegression()
LR2.fit(X2_train, y2_train)

# Predicciones
pred2 = LR2.predict(X2_test)

# Resultados del modelo
st.subheader("Resultados del modelo")
st.write("R2:", round(r2_score(y2_test, pred2), 4))
st.write("MSE:", round(metrics.mean_squared_error(y2_test, pred2), 4))

# Datos del alumno
st.subheader("Ingrese los datos del alumno")

horas_estudio = st.number_input("Horas de estudio", min_value=0.0, value=5.0)
horas_sueno = st.number_input("Horas de sueño", min_value=0.0, value=7.0)
asistencia = st.number_input("Porcentaje de asistencia", min_value=0.0, max_value=100.0, value=80.0)
examen_anterior = st.number_input("Calificación anterior", min_value=0.0, max_value=100.0, value=70.0)

# Calcular predicción
if st.button("Predecir calificación"):

    alumno = pd.DataFrame([[
        horas_estudio,
        horas_sueno,
        asistencia,
        examen_anterior
    ]], columns=X2.columns)

    calificacion = LR2.predict(alumno)

    st.success(f"Calificación estimada: {calificacion[0]:.2f}")

# Comparar resultados
st.subheader("Calificaciones reales y estimadas")

comp = pd.DataFrame({
    'Real': y2_test,
    'Predicción': pred2
})

st.dataframe(comp)
