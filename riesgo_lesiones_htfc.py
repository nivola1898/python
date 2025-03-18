import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Carga de Datos (Simulados)
@st.cache_data  # Cache para mejorar el rendimiento
def load_data():
    # Simulación de datos con más variables relevantes y realistas
    data = {
        'horas_trabajo': [40, 35, 45, 50, 30, 40, 35, 45, 50, 30, 40, 35, 45, 50, 30],
        'horas_sueño': [7, 6, 8, 5, 9, 7, 6, 8, 5, 9, 7, 6, 8, 5, 9],
        'dieta': [4, 3, 5, 2, 4, 4, 3, 5, 2, 4, 4, 3, 5, 2, 4],
        'grasa_visceral': [12, 15, 10, 18, 11, 12, 15, 10, 18, 11, 12, 15, 10, 18, 11],
        'imc': [24, 26, 22, 28, 23, 24, 26, 22, 28, 23, 24, 26, 22, 28, 23],
        'edad': [25, 28, 23, 30, 22, 25, 28, 23, 30, 22, 25, 28, 23, 30, 22],
        'historial_lesiones': [0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0], # 0=No, 1=Si, 2=Grave
        'posicion': ['Delantero', 'Defensa', 'Mediocampista', 'Delantero', 'Defensa',
                     'Delantero', 'Defensa', 'Mediocampista', 'Delantero', 'Defensa',
                     'Delantero', 'Defensa', 'Mediocampista', 'Delantero', 'Defensa'],
        'lesionado': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0] # 0=No, 1=Sí
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

# 2. Preparación de Datos
# Convertir variables categóricas a numéricas
df['posicion'] = df['posicion'].astype('category').cat.codes

# Separar características y variable objetivo
features = ['horas_trabajo', 'horas_sueño', 'dieta', 'grasa_visceral', 'imc', 'edad', 'historial_lesiones', 'posicion']
target = 'lesionado'

# 3. Entrenamiento del Modelo (Random Forest)
# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Predicción y Evaluación
# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Precisión del modelo: {accuracy:.2f}")

# Mostrar reporte de clasificación
report = classification_report(y_test, y_pred)
st.text(f"Reporte de Clasificación:\n{report}")

# 5. Interfaz de Usuario con Streamlit
st.title('Dashboard de Riesgo de Lesiones - Harborough Town FC')

# Entradas del usuario
st.sidebar.header("Parámetros del Jugador")
horas_trabajo = st.sidebar.slider('Horas de trabajo/semana', 0, 60, 40)
horas_sueño = st.sidebar.slider('Horas de sueño/noche', 0, 12, 6)
dieta = st.sidebar.slider('Calidad de dieta (1-5)', 1, 5, 3)
grasa_visceral = st.sidebar.slider('Grasa visceral (%)', 5, 30, 15)
imc = st.sidebar.slider('Índice de Masa Corporal (IMC)', 15, 40, 25)
edad = st.sidebar.slider('Edad', 18, 40, 25)
historial_lesiones = st.sidebar.selectbox('Historial de lesiones', [0, 1, 2], index=0)
posicion = st.sidebar.selectbox('Posición', ['Delantero', 'Defensa', 'Mediocampista'], index=0)
posicion = ['Delantero', 'Defensa', 'Mediocampista'].index(posicion) # Convertir a numérico

# 6. Predicción con el Modelo Entrenado
# Crear un DataFrame con los datos del jugador
jugador_data = pd.DataFrame({
    'horas_trabajo': [horas_trabajo],
    'horas_sueño': [horas_sueño],
    'dieta': [dieta],
    'grasa_visceral': [grasa_visceral],
    'imc': [imc],
    'edad': [edad],
    'historial_lesiones': [historial_lesiones],
    'posicion': [posicion]
})

# Predecir el riesgo de lesión
riesgo = model.predict_proba(jugador_data[features])[0][1] * 100

# 7. Mostrar Resultados
st.header("Resultado de la Predicción")
st.metric("Riesgo de Lesión", f"{riesgo:.2f}%")

# 8. Interpretación
if riesgo > 70:
    st.error("¡ALTO RIESGO! Se recomienda una evaluación médica y un plan de prevención intensivo.")
elif riesgo > 50:
    st.warning("RIESGO MODERADO. Considerar ajustes en el entrenamiento y monitoreo cercano.")
else:
    st.success("BAJO RIESGO. Mantener las prácticas actuales y seguir monitoreando.")

# 9. Análisis del FC United
st.header("Lecciones del FC United of Manchester")
st.write("""
El FC United of Manchester es un ejemplo de cómo una gran base social y una fuerte identidad comunitaria no son suficientes para garantizar el éxito a largo plazo. 
Su historia nos enseña la importancia de:
*   **La gestión financiera prudente:** Evitar deudas excesivas y diversificar las fuentes de ingresos.
*   **La inversión inteligente en el equipo:** Priorizar el desarrollo de jugadores jóvenes y evitar fichajes costosos y arriesgados.
*   **La planificación a largo plazo:** Establecer objetivos realistas y evitar decisiones impulsivas.
""")
