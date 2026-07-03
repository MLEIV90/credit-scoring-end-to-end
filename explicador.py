import os
import numpy as np
import pandas as pd
from groq import Groq

MAPA = {
    'person_income_log':       'nivel de ingresos',
    'loan_grade_enc':          'grado / historial crediticio',
    'loan_percent_income_log': 'peso de la cuota sobre el ingreso',
    'is_high_dti':             'nivel de endeudamiento (DTI)',
    'loan_int_rate_log':       'tasa de interés del préstamo',
    'person_emp_length_log':   'antigüedad laboral',
    'cb_default_on_file_enc':  'antecedente de impago en buró',
}

def humanizar(v):
    if v.startswith('loan_intent_'):
        return f"destino del préstamo ({v.split('_')[-1].lower()})"
    if v.startswith('person_home_ownership_'):
        return f"situación habitacional ({v.split('_')[-1].lower()})"
    return MAPA.get(v, v)

def factores_decision(ap, decision):
    sel = ap[ap['contribucion'] > 0] if decision == "RECHAZADA" else ap[ap['contribucion'] < 0]
    sel = sel.reindex(sel['contribucion'].abs().sort_values(ascending=False).index)
    return [humanizar(v) for v in sel['variable'].head(4)]

SYSTEM = """Sos un asistente que redacta, para el solicitante, el motivo de una decisión de crédito.
Reglas estrictas:
- Usá EXCLUSIVAMENTE los factores que se te dan. No inventes datos ni cifras.
- No menciones nombres técnicos, números del modelo ni probabilidades.
- Español claro y respetuoso. EXACTAMENTE 2 oraciones: la primera comunica la decisión, la segunda lista los factores. Sin saludos de cierre ni frases de relleno.
- PROHIBIDO sugerir que la decisión podría cambiar en el futuro. No uses frases como "no es el momento adecuado", "en este momento" ni similares. La decisión es final, no temporal.
- Los factores indican QUÉ se evaluó, no su magnitud. No afirmes si un valor es "alto" o "bajo" salvo que sea evidente por la decisión (un rechazo implica factores desfavorables; una aprobación, favorables)."""

def generar_explicacion(decision, drivers):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    user = (f"Decisión: {decision}.\n"
            f"Factores que más pesaron, de mayor a menor: {', '.join(drivers)}.\n"
            f"Redactá el motivo para el solicitante.")
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0, max_tokens=120,
    )
    texto = r.choices[0].message.content
    for f in ["en este momento", "no es el momento", "por ahora", "por el momento", "en esta ocasión"]:
        texto = texto.replace(f, "")
    return texto.replace("  ", " ").strip()

def evaluar(solicitante, modelo, calibrador, params, escenario="Moderado"):
    """solicitante: DataFrame de 1 fila con las columnas del modelo."""
    cols   = params["cols"]
    scaler = modelo.named_steps['scaler']
    clf    = modelo.named_steps['clf']

    contrib = clf.coef_[0] * scaler.transform(solicitante[cols])[0]
    ap = pd.DataFrame({'variable': cols, 'contribucion': contrib}) \
           .sort_values('contribucion', key=abs, ascending=False)

    pdc = calibrador.predict_proba(solicitante[cols])[0, 1]
    score = int(params["offset"] - params["factor"] * np.log(pdc / (1 - pdc)))
    score = max(380, min(863, score))   # clamp al rango válido de la scorecard
    corte = params["corte"][escenario]
    decision = "APROBADA" if score >= corte else "RECHAZADA"
    drivers = factores_decision(ap, decision)

    return {"pd": pdc, "score": score, "decision": decision,
            "corte": corte, "drivers": drivers}
    
MAPA_GRADE = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

def preparar_solicitante(datos, cols):
    """Reproduce el feature engineering del cap. 5 para un solicitante crudo."""
    income   = float(datos['person_income'])
    emp_len  = float(datos['person_emp_length'])
    amnt     = float(datos['loan_amnt'])
    int_rate = float(datos['loan_int_rate'])
    pct_income = amnt / income if income > 0 else 0.0

    fila = {c: 0 for c in cols}                       # todo en 0 (incluye dummies)
    fila['person_income_log']       = np.log1p(income)
    fila['loan_percent_income_log'] = np.log1p(pct_income)
    fila['person_emp_length_log']   = np.log1p(emp_len)
    fila['loan_int_rate']           = int_rate
    fila['loan_amnt']               = amnt
    fila['loan_grade_enc']          = MAPA_GRADE[datos['loan_grade']]
    fila['cb_default_on_file_enc']  = 1 if datos['cb_person_default_on_file'] == 'Y' else 0
    fila['is_high_dti']             = 1 if pct_income > 0.40 else 0

    col_intent = f"loan_intent_{datos['loan_intent']}"
    if col_intent in fila: fila[col_intent] = 1          # si es la ref. queda en 0 (correcto)
    col_home = f"person_home_ownership_{datos['person_home_ownership']}"
    if col_home in fila: fila[col_home] = 1

    return pd.DataFrame([fila])[cols]