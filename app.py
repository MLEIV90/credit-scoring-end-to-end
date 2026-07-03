import joblib, json
import pandas as pd
import streamlit as st
from explicador import evaluar, generar_explicacion, preparar_solicitante

st.set_page_config(page_title="Credit Scoring + Reason Codes", page_icon="💳")
ESCENARIO = "Moderado"   # política de riesgo del banco (fija, no la elige el usuario)

@st.cache_resource
def cargar():
    modelo     = joblib.load("modelo_log_final.pkl")
    calibrador = joblib.load("calibrador.pkl")
    with open("params.json") as f:
        params = json.load(f)
    ejemplos = pd.read_csv("ejemplos.csv")
    return modelo, calibrador, params, ejemplos

modelo, calibrador, params, ejemplos = cargar()

st.title("💳 Evaluación de crédito con explicación")
st.caption("El modelo decide; la IA generativa solo redacta el motivo.")

def mostrar(res):
    color = "green" if res["decision"] == "APROBADA" else "red"
    st.markdown(f"### Decisión: :{color}[{res['decision']}]")
    c1, c2 = st.columns(2)
    c1.metric("Score", res["score"])
    c2.metric("Corte", f"≥ {res['corte']}")
    with st.spinner("Generando explicación…"):
        st.info(generar_explicacion(res["decision"], res["drivers"]))

tab_form, tab_ej = st.tabs(["📝 Cargar solicitante", "🎲 Ejemplo aleatorio"])

with tab_form:
    c1, c2 = st.columns(2)
    with c1:
        income   = st.number_input("Ingreso anual (USD)", min_value=0, value=50000, step=1000)
        emp_len  = st.number_input("Antigüedad laboral (años)", min_value=0.0, value=5.0, step=0.5)
        amnt     = st.number_input("Monto del préstamo (USD)", min_value=0, value=10000, step=500)
        int_rate = st.number_input("Tasa de interés (%)", min_value=0.0, value=11.0, step=0.1)
    with c2:
        grade  = st.selectbox("Grado crediticio", list("ABCDEFG"))
        intent = st.selectbox("Destino del préstamo",
                   ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
        home   = st.selectbox("Situación habitacional", ["RENT","OWN","MORTGAGE","OTHER"])
        deflt  = st.selectbox("¿Impago previo en buró?", ["N","Y"])

    if st.button("Evaluar solicitud", type="primary"):
        datos = {'person_income': income, 'person_emp_length': emp_len,
                 'loan_amnt': amnt, 'loan_int_rate': int_rate,
                 'loan_grade': grade, 'loan_intent': intent,
                 'person_home_ownership': home, 'cb_person_default_on_file': deflt}
        sol = preparar_solicitante(datos, params["cols"])
        mostrar(evaluar(sol, modelo, calibrador, params, ESCENARIO))

with tab_ej:
    if st.button("🎲 Evaluar un solicitante de ejemplo"):
        sol = ejemplos.sample(1).reset_index(drop=True)
        mostrar(evaluar(sol, modelo, calibrador, params, ESCENARIO))