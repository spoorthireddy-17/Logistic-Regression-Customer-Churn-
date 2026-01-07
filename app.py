import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# -------------------------------------------------
# Load CSS
# -------------------------------------------------
def load_css(file):
    with open(file) as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("style.css")

st.title("📉 Customer Churn Prediction")
st.write("Logistic Regression based churn prediction")

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict Customer"])

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\spoor\\Downloads\\Telco-Customer-Churn.csv")

df = load_data()

# -------------------------------------------------
# Data Preprocessing
# -------------------------------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Train Model
# -------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------------------------
# Predictions (default threshold = 0.5)
# -------------------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred) * 100
total_churn = (y_pred == 1).sum()
total_stay = (y_pred == 0).sum()

# =================================================
# 📊 DASHBOARD PAGE
# =================================================
if page == "Dashboard":

    st.markdown('<div class="section-card">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head())
    st.caption(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # -------------------------------------------------
    # KPI Cards
    # -------------------------------------------------
    st.markdown('<div class="section-card">Model Summary</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div>Model Accuracy</div>
                <h2 style="color:#2563eb;">{accuracy:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="card">
                <div>Likely to Churn</div>
                <h2 style="color:#dc2626;">{total_churn}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="card">
                <div>Likely to Stay</div>
                <h2 style="color:#16a34a;">{total_stay}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------
    # Churn Distribution
    # -------------------------------------------------
    st.markdown('<div class="section-card">Churn Distribution</div>', unsafe_allow_html=True)
    st.bar_chart(df["Churn"].value_counts())

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    st.markdown('<div class="section-card">Confusion Matrix</div>', unsafe_allow_html=True)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="PuBu",
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Churn Prediction")

    st.pyplot(fig)

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------
    st.markdown('<div class="section-card">Classification Report</div>', unsafe_allow_html=True)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.style.format({
            "precision": "{:.2f}",
            "recall": "{:.2f}",
            "f1-score": "{:.2f}",
            "support": "{:.0f}"
        })
    )

# =================================================
# 🧾 PREDICT CUSTOMER PAGE
# =================================================
if page == "Predict Customer":

    st.markdown('<div class="section-card">Predict Customer Churn</div>', unsafe_allow_html=True)

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    if st.button("Predict Churn"):
        prob = model.predict_proba(input_scaled)[0][1]

        if prob >= 0.5:
            st.error(f"🚨 Likely to Churn ({prob:.2%})")
        else:
            st.success(f"✅ Likely to Stay ({prob:.2%})")

st.markdown(
    """
    <div class="footer">
        Built with Streamlit & Logistic Regression • Telco Customer Churn Dataset
    </div>
    """,
    unsafe_allow_html=True
)
