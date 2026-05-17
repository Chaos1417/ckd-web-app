import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CKD Risk Analyzer", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS: Vibrant Green Medical Theme ────────────────────────────────────────
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:12px 0 4px">
      <span style="font-size:2.4rem">🧬</span>
      <h2 style="font-size:1.2rem;font-weight:800;margin:6px 0 2px">CKD Risk Analyzer</h2>
      <p style="font-size:.78rem;opacity:.65;margin:0">AI-Powered Clinical Decision Support</p>
    </div><hr>""", unsafe_allow_html=True)

    st.markdown("""<div class="fcard"><div class="ft">🌍 Global</div>
      <p>Globally, <b>1 in 10</b> people live with Chronic Kidney Disease, and many
      are unaware until advanced stages.</p></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="fcard"><div class="ft">🇵🇰 Pakistan</div>
      <p>Pakistan ranks <b>8th globally</b> in kidney diseases. CKD prevalence is
      estimated at <b>21.2%</b>, driven by Diabetes and Hypertension.</p></div>""",
        unsafe_allow_html=True)

    st.markdown("""<hr><p style="font-size:.75rem;opacity:.45;text-align:center">
      Model: Extra Trees · 12 RFE Features<br>⚠️ Educational use only</p>""",
        unsafe_allow_html=True)

# ── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = os.path.dirname(os.path.abspath(__file__))
    m = joblib.load(os.path.join(base, "ckd_extra_trees_model.pkl"))
    s = joblib.load(os.path.join(base, "ckd_scaler.pkl"))
    e = joblib.load(os.path.join(base, "ckd_encoder.pkl"))
    return m, s, e

model, scaler, encoder = load_assets()

# ── Synced Slider + Number Input ────────────────────────────────────────────
def dual_input(label, mn, mx, default, step, key):
    sk, nk = f"_s_{key}", f"_n_{key}"
    if sk not in st.session_state:
        st.session_state[sk] = default
    if nk not in st.session_state:
        st.session_state[nk] = default
    def _fs():
        st.session_state[nk] = st.session_state[sk]
    def _fn():
        st.session_state[sk] = max(mn, min(mx, st.session_state[nk]))
    c1, c2 = st.columns([4, 1.3])
    with c1:
        st.slider(label, min_value=mn, max_value=mx, step=step, key=sk, on_change=_fs)
    with c2:
        st.number_input("✏️", min_value=mn, max_value=mx, step=step, key=nk,
                        on_change=_fn, label_visibility="collapsed")
    return st.session_state[sk]

# ── Header + Banner ─────────────────────────────────────────────────────────
st.markdown("""<div class="hdr">
  <h1>🧬 Chronic Kidney Disease Risk Analyzer</h1>
  <p>Enter your clinical biomarkers below. Our Extra Trees AI model analyzes
  12 key features to assess kidney-health risk instantly.</p>
</div>""", unsafe_allow_html=True)

st.image("https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=1200&h=300&fit=crop",
         use_container_width=True, caption="AI-Powered Clinical Health Assessment")

# ── Input Form ──────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="stitle">🩸 Blood Metrics</div>', unsafe_allow_html=True)
b1, b2 = st.columns(2)
with b1:
    hemo = dual_input("Hemoglobin (g/dL)", 3.0, 18.0, 15.0, 0.1, "hemo")
    bgr = dual_input("Blood Glucose Random (mg/dL)", 50, 500, 120, 1, "bgr")
    bu = dual_input("Blood Urea (mg/dL)", 10, 400, 40, 1, "bu")
with b2:
    sc = dual_input("Serum Creatinine (mg/dL)", 0.4, 25.0, 1.2, 0.1, "sc")
    sod = dual_input("Sodium (mEq/L)", 100, 165, 135, 1, "sod")
    pcv = dual_input("Packed Cell Volume (%)", 9, 60, 40, 1, "pcv")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="stitle">🔬 Urine & Cell Metrics</div>', unsafe_allow_html=True)
u1, u2 = st.columns(2)
with u1:
    sg = st.select_slider("Specific Gravity",
                           options=["1.005","1.010","1.015","1.020","1.025"], value="1.015")
    rbc_l = st.radio("Red Blood Cells (rbc)", ["Normal","Abnormal"], horizontal=True)
    rbc = "normal" if rbc_l == "Normal" else "abnormal"
with u2:
    rc = dual_input("RBC Count (millions/cmm)", 2.0, 8.0, 4.5, 0.1, "rc")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="stitle">📋 Medical History</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
with m1:
    htn = "yes" if st.radio("Hypertension", ["No","Yes"], horizontal=True, key="htn_r") == "Yes" else "no"
with m2:
    dm = "yes" if st.radio("Diabetes Mellitus", ["No","Yes"], horizontal=True, key="dm_r") == "Yes" else "no"
with m3:
    pe = "yes" if st.radio("Pedal Edema", ["No","Yes"], horizontal=True, key="pe_r") == "Yes" else "no"
st.markdown('</div>', unsafe_allow_html=True)

# ── Static Insights Fallback ────────────────────────────────────────────────
def _show_static_insights(hemo, sc, bu, bgr, sod, pcv, rc, rbc, htn, dm, pe, sg, is_ckd, tc):
    """Fallback static insights when Groq API is unavailable."""
    flags = []
    if hemo < 10:
        flags.append(("#dc2626", f"<b>Hemoglobin</b> critically low at <b>{hemo} g/dL</b> (normal ≥12) — suggests anemia, a common CKD indicator."))
    elif hemo < 12:
        flags.append(("#d97706", f"<b>Hemoglobin</b> below normal at <b>{hemo} g/dL</b> (normal ≥12)."))
    if sc > 3:
        flags.append(("#dc2626", f"<b>Serum Creatinine</b> elevated at <b>{sc} mg/dL</b> (normal 0.6–1.2) — impaired kidney filtration."))
    elif sc > 1.4:
        flags.append(("#d97706", f"<b>Serum Creatinine</b> mildly elevated at <b>{sc} mg/dL</b>."))
    if bu > 80:
        flags.append(("#dc2626", f"<b>Blood Urea</b> high at <b>{bu} mg/dL</b> (normal 7–20)."))
    elif bu > 40:
        flags.append(("#d97706", f"<b>Blood Urea</b> above optimal at <b>{bu} mg/dL</b>."))
    if bgr > 200:
        flags.append(("#dc2626", f"<b>Blood Glucose</b> very high at <b>{bgr} mg/dL</b>."))
    elif bgr > 140:
        flags.append(("#d97706", f"<b>Blood Glucose</b> elevated at <b>{bgr} mg/dL</b>."))
    if sod < 120:
        flags.append(("#dc2626", f"<b>Sodium</b> low at <b>{sod} mEq/L</b> (normal 136–145)."))
    if pcv < 30:
        flags.append(("#dc2626", f"<b>PCV</b> low at <b>{pcv}%</b> (normal 36–44%)."))
    if rc < 3.5:
        flags.append(("#dc2626", f"<b>RBC Count</b> low at <b>{rc} M/cmm</b>."))
    if rbc == "abnormal":
        flags.append(("#d97706", "<b>Red Blood Cells</b> in urine are <b>Abnormal</b>."))
    if htn == "yes":
        flags.append(("#d97706", "History of <b>Hypertension</b> — a CKD risk factor."))
    if dm == "yes":
        flags.append(("#d97706", "History of <b>Diabetes</b> — the #1 cause of CKD."))
    if pe == "yes":
        flags.append(("#d97706", "<b>Pedal Edema</b> reported — may indicate fluid retention."))
    if float(sg) <= 1.010:
        flags.append(("#d97706", f"<b>Specific Gravity</b> low at <b>{sg}</b>."))
    if flags:
        items_html = "".join(
            f'<div class="insight-item"><div class="insight-dot" style="background:{c}"></div>'
            f'<div class="insight-text">{t}</div></div>' for c, t in flags)
        st.markdown(f'<div class="insight-box"><div class="stitle">🧠 Clinical Insights</div>{items_html}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight-box"><div class="stitle">🧠 Clinical Insights</div>'
                    '<p style="color:#374151">All biomarkers within normal ranges. Continue regular check-ups.</p></div>',
                    unsafe_allow_html=True)

# ── Predict ─────────────────────────────────────────────────────────────────
if st.button("🔍  Analyze Patient Data"):
    # 24-column dummy trick
    num_cols = ['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc']
    cat_cols = ['sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
    rfe_cols = ['sg','rbc','bgr','bu','sc','sod','hemo','pcv','rc','htn','dm','pe']

    data = {c: [0.0] for c in num_cols}
    data.update({c: ['missing'] for c in cat_cols})
    row = pd.DataFrame(data)

    row.at[0,'sg'] = sg;     row.at[0,'rbc'] = rbc
    row.at[0,'bgr'] = float(bgr); row.at[0,'bu'] = float(bu)
    row.at[0,'sc'] = float(sc);    row.at[0,'sod'] = float(sod)
    row.at[0,'hemo'] = float(hemo); row.at[0,'pcv'] = float(pcv)
    row.at[0,'rc'] = float(rc);    row.at[0,'htn'] = htn
    row.at[0,'dm'] = dm;          row.at[0,'pe'] = pe

    for c in num_cols:
        row[c] = pd.to_numeric(row[c], errors='coerce')

    row[cat_cols] = encoder.transform(row[cat_cols])
    row[num_cols] = scaler.transform(row[num_cols])

    prediction = model.predict(row[rfe_cols])
    prob = model.predict_proba(row[rfe_cols])[0]
    prob_ckd = prob[1]
    is_ckd = prediction[0] == 1
    confidence = prob_ckd if is_ckd else prob[0]

    # Risk label
    if prob_ckd >= .75:
        rl, rc2 = "High Risk", "#dc2626"
    elif prob_ckd >= .45:
        rl, rc2 = "Moderate Risk", "#d97706"
    else:
        rl, rc2 = "Low Risk", "#059669"

    # Result card
    if is_ckd:
        cls, tc = "r-ckd", "#dc2626"
        ttl, sub = "🚨 CKD Risk Detected", "The AI model indicates a significant likelihood of Chronic Kidney Disease."
    else:
        cls, tc = "r-ok", "#059669"
        ttl, sub = "✅ Low CKD Risk", "Clinical indicators do not strongly suggest CKD at this time."

    st.markdown(f"""<div class="{cls}">
      <div class="r-title" style="color:{tc}">{ttl}</div>
      <p style="color:#4b5563;margin-bottom:14px">{sub}</p>
      <div style="display:flex;align-items:center;gap:12px">
        <span style="font-weight:600;color:#6b7280">Confidence</span>
        <span style="font-weight:800;font-size:1.3rem;color:{tc}">{confidence*100:.1f}%</span>
      </div>
      <div class="meter-track"><div class="meter-fill" style="width:{confidence*100:.1f}%;
        background:{tc}"></div></div>
      <p style="text-align:right;font-size:.82rem;color:#6b7280;margin:4px 0 0">
        Assessment: <b style="color:{rc2}">{rl}</b></p>
    </div>""", unsafe_allow_html=True)

    # ── Groq LLM Explanation ────────────────────────────────────────────────
    input_summary = (f"Hemoglobin={hemo}g/dL, Blood Glucose={bgr}mg/dL, "
                     f"Blood Urea={bu}mg/dL, Serum Creatinine={sc}mg/dL, "
                     f"Sodium={sod}mEq/L, PCV={pcv}%, Specific Gravity={sg}, "
                     f"RBC={rbc}, RBC Count={rc}M/cmm, "
                     f"Hypertension={htn}, Diabetes={dm}, Pedal Edema={pe}")
    pred_text = f"CKD detected (probability {prob_ckd*100:.1f}%)" if is_ckd else f"No CKD (probability {prob[0]*100:.1f}%)"

    prompt = f"""You are a medical AI assistant explaining a CKD prediction to a patient.
The patient's lab values: {input_summary}
The ML model prediction: {pred_text}

In 3-4 sentences, explain WHY the model likely made this prediction by referencing
the patient's specific abnormal values with their numbers. Be empathetic and educational.
End with one actionable recommendation. Do not use markdown formatting."""

    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        if api_key:
            client = Groq(api_key=api_key)
            with st.spinner("🧠 AI Doctor is analyzing your results..."):
                resp = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "system", "content": "You are a concise, empathetic medical AI."},
                              {"role": "user", "content": prompt}],
                    temperature=0.4, max_tokens=300)
                llm_text = resp.choices[0].message.content

            st.markdown(f"""<div class="card" style="border-color:#a7f3d0">
              <div class="stitle">🧠 AI Doctor's Analysis</div>
              <p style="color:#374151;font-size:.95rem;line-height:1.65">{llm_text}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("💡 Add your GROQ_API_KEY to `.streamlit/secrets.toml` to enable AI-powered explanations.")
            # Fallback: static explanation
            _show_static_insights(hemo, sc, bu, bgr, sod, pcv, rc, rbc, htn, dm, pe, sg, is_ckd, tc)
    except Exception as e:
        st.warning(f"Groq API unavailable: {e}")
        _show_static_insights(hemo, sc, bu, bgr, sod, pcv, rc, rbc, htn, dm, pe, sg, is_ckd, tc)

    # Recommendation
    st.markdown("""<div class="card" style="border-color:#bbf7d0;background:#f0fdf4">
      <div class="stitle">💡 What To Do Next</div>
      <p style="color:#374151;font-size:.95rem;line-height:1.6">
        This assessment is generated by a Machine Learning model and is <b>not a medical
        diagnosis</b>. If any risk factors were flagged, consult a <b>nephrologist</b>
        for comprehensive kidney function tests (eGFR, urine albumin-to-creatinine ratio).
        Early detection can slow or prevent CKD progression.</p>
    </div>""", unsafe_allow_html=True)
