import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Health System", layout="wide")

# ---------------- COLORFUL UI ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb, #a6c1ee);
    background-size: 400% 400%;
}
.block-container {
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}
.stButton>button {
    background: linear-gradient(90deg, #ff6a00, #ee0979);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
heart_model = pickle.load(open("heart_model.pkl", "rb"))
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
lung_model = pickle.load(open("lung_model.pkl", "rb"))

# ---------------- TITLE ----------------
st.title("🧠 AI Health Analytics Platform")
st.markdown("### 🚑 Advanced Prediction + AI Assistant + Smart Insights")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧾 User Profile")

name = st.sidebar.text_input("Enter your name")

age = st.sidebar.slider("Age", 1, 100)
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
exercise = st.sidebar.selectbox("Exercise Level", ["Low", "Medium", "High"])
alcohol = st.sidebar.selectbox("Alcohol", ["No", "Yes"])

# BMI Calculation
height = st.sidebar.slider("Height (cm)", 100, 200)
weight = st.sidebar.slider("Weight (kg)", 30, 120)

bmi = weight / ((height/100)**2)
st.sidebar.write(f"📊 BMI: {bmi:.2f}")

# Convert inputs
smoking_val = 1 if smoking == "Yes" else 0
alcohol_val = 1 if alcohol == "Yes" else 0
exercise_val = {"Low":0, "Medium":1, "High":2}[exercise]

input_data = np.array([[age, smoking_val, exercise_val, bmi, alcohol_val]])

# ---------------- CHATBOT ----------------
def smart_chatbot(age, smoking, bmi, exercise, alcohol, risks):
    heart, diabetes, lung = risks
    tips = []

    if heart > 0.7:
        tips.append("❤️ Reduce oily food and do cardio")
    if diabetes > 0.7:
        tips.append("🍬 Control sugar intake")
    if lung > 0.7:
        tips.append("🫁 Avoid smoking immediately")

    if smoking == "Yes":
        tips.append("🚭 Quit smoking gradually")
    if bmi > 30:
        tips.append("⚖️ Reduce weight with diet & exercise")
    if exercise == "Low":
        tips.append("🏃 Exercise daily")
    if alcohol == "Yes":
        tips.append("🍺 Reduce alcohol consumption")

    if not tips:
        tips.append("✅ You are healthy. Maintain lifestyle")

    return tips

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------- PREDICTION ----------------
if st.sidebar.button("🔍 Predict"):

    with st.spinner("Analyzing your health..."):

        heart = heart_model.predict_proba(input_data)[0][1]
        diabetes = diabetes_model.predict_proba(input_data)[0][1]
        lung = lung_model.predict_proba(input_data)[0][1]

        st.session_state["data"] = (heart, diabetes, lung)

        st.session_state["history"].append({
            "heart": heart,
            "diabetes": diabetes,
            "lung": lung
        })

        st.subheader(f"👤 Hello {name}, here is your report")

        col1, col2, col3 = st.columns(3)

        def show(col, title, val):
            col.subheader(title)
            col.metric("Risk %", f"{val*100:.1f}%")

            if val > 0.7:
                col.error("High Risk ⚠️")
            elif val > 0.4:
                col.warning("Medium Risk")
            else:
                col.success("Low Risk")

        show(col1, "❤️ Heart", heart)
        show(col2, "🍬 Diabetes", diabetes)
        show(col3, "🫁 Lung", lung)

        # Chart
        st.subheader("📊 Risk Analysis")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Heart', 'Diabetes', 'Lung'],
            y=[heart, diabetes, lung],
            text=[f"{heart*100:.1f}%", f"{diabetes*100:.1f}%", f"{lung*100:.1f}%"],
            textposition='auto'
        ))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Health Score
        score = int((1 - max([heart, diabetes, lung])) * 100)
        st.subheader("🏆 Health Score")
        st.metric("Score", f"{score}/100")

        # Explanation
        st.subheader("🧠 Why this risk?")
        if smoking == "Yes":
            st.write("• Smoking increases lung & heart risk")
        if bmi > 30:
            st.write("• High BMI increases diabetes risk")
        if exercise == "Low":
            st.write("• Low exercise reduces fitness")

# ---------------- CHATBOT UI ----------------
st.subheader("🤖 AI Assistant")

if st.button("Get Health Advice"):

    if "data" in st.session_state:

        tips = smart_chatbot(age, smoking, bmi, exercise, alcohol, st.session_state["data"])

        for tip in tips:
            st.success(tip)

    else:
        st.warning("Run prediction first")

# ---------------- DAILY PLAN ----------------
st.subheader("📅 Daily Health Plan")

if bmi > 30:
    st.write("🏃 45 mins cardio")
else:
    st.write("🏃 20 mins walking")

st.write("🥗 Balanced diet recommended")

# ---------------- HISTORY ----------------
st.subheader("📊 Prediction History")
st.write(st.session_state["history"])

# ---------------- PDF ----------------
st.subheader("📄 Download Report")

if st.button("Generate PDF"):

    if "data" in st.session_state:

        heart, diabetes, lung = st.session_state["data"]

        doc = SimpleDocTemplate("health_report.pdf")
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("AI Health Report", styles["Title"]))
        content.append(Paragraph(f"Name: {name}", styles["Normal"]))
        content.append(Paragraph(f"Age: {age}", styles["Normal"]))
        content.append(Paragraph(f"BMI: {bmi:.2f}", styles["Normal"]))

        content.append(Paragraph(f"Heart Risk: {heart*100:.2f}%", styles["Normal"]))
        content.append(Paragraph(f"Diabetes Risk: {diabetes*100:.2f}%", styles["Normal"]))
        content.append(Paragraph(f"Lung Risk: {lung*100:.2f}%", styles["Normal"]))

        doc.build(content)

        with open("health_report.pdf", "rb") as f:
            st.download_button("⬇️ Download Report", f)

    else:
        st.warning("Run prediction first")