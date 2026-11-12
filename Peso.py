import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción del peso  ''')
st.image("bascula.jpg", caption="Predicción del peso de una persona.")

st.header('Datos personales')

def user_input_features():
  # Entrada
  Edad = st.number_input('Edad:', min_value=0, max_value=100, value = 0, step = 1)
  Genero = st.number_input('Género (Mujer = 0, Hombre = 1):',  min_value=0, max_value=1, value = 0, step = 1)
  Altura = st.number_input('Altura (en cm):', min_value=0, max_value=230, value = 0, step = 1)
  Cintura = st.number_input('Circunferencia de la cintura (en cm):', min_value=0, max_value=140, value = 0, step = 1)
  Brazo = st.number_input('Circunferencia del brazo (en cm):', min_value=0, max_value=50, value = 0, step = 1)
  IMC = st.number_input('Índice de masa corporal:', min_value=0, max_value=50, value = 0, step = 1)


  user_input_data = {'AGE': Edad,
                     'GENDER (1=M)': Genero,
                     'HEIGHT': Altura,
                     'WAIST': Cintura,
                     'ARM CIRC': Brazo,
                     'BMI': IMC,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()
prediccion=0
datos =  pd.read_csv('BodyData_df.csv', encoding='latin-1')
X = datos.drop(columns='WEIGHT')
y = datos['WEIGHT']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['AGE'] + b1[1]*df['GENDER (1=M)'] + b1[2]*df['HEIGHT'] + b1[3]*df['WAIST'] + b1[4]*df['ARM CIRC'] + b1[5]*df['BMI']

st.subheader('Cálculo del peso')
st.write('El peso de la persona es ', prediccion)
