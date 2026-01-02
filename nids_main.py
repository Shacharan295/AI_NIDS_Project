import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------
# STYLING
# -------------------------------------------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.title { font-size: 40px; font-weight: 800; color: #4deeea; }
.card {
    background-color: #161b22;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.signal {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.markdown('<div class="title">🛡️ AI-Based Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.write("### Real-Time Cyber Attack Detection using Machine Learning")
st.divider()

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df.columns = df.columns.str.strip()

    df = df[
        [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Packet Length Mean",
            "Active Mean",
            "Label",
        ]
    ]

    # 🔥 ADD PACKETS PER SECOND (FIX)
    df["Packets Per Second"] = df["Total Fwd Packets"] / (df["Flow Duration"] + 1)

    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    df = df.dropna()

    if len(df) > 75000:
        df = df.sample(75000, random_state=42)

    return df

df = load_data()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("⚙️ Model Controls")
n_estimators = st.sidebar.slider("Number of Trees", 50, 150, 100)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 30)
train_btn = st.sidebar.button("🔥 Train Model Now")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
left, right = st.columns([1.2, 1.8])

# -------------------------------------------------
# OVERVIEW
# -------------------------------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Dashboard Overview")
    st.write("""
    • Detects cyber attacks using Machine Learning  
    • Trained on real network traffic (CIC-IDS dataset)  
    • Uses Random Forest classifier  
    • Includes live traffic simulation  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🚀 Model Training")

    if train_btn:
        with st.spinner("Training model..."):
            X = df.drop("Label", axis=1)
            y = df["Label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42
            )

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state.model = model
            st.session_state.accuracy = accuracy_score(y_test, y_pred)
            st.session_state.cm = confusion_matrix(y_test, y_pred)

        st.success("Model trained successfully!")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Model Performance")

    if st.session_state.model is not None:
        acc = st.session_state.accuracy

        signal = "🟢 EXCELLENT" if acc >= 0.99 else "🟡 GOOD" if acc >= 0.95 else "🔴 NEEDS IMPROVEMENT"

        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.markdown(f"<div class='signal'>{signal}</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4.5, 3))
        sns.heatmap(
            st.session_state.cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Attack"],
            yticklabels=["Benign", "Attack"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.info("Train the model first.")

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# LIVE TRAFFIC SIMULATOR
# -------------------------------------------------
st.subheader("🧪 Live Traffic Simulator")

c1, c2, c3, c4 = st.columns(4)

flow = c1.number_input("Flow Duration", 0, 100000, 500)
packets = c2.number_input("Total Fwd Packets", 0, 5000, 100)
pkt_len = c3.number_input("Packet Length Mean", 0, 2000, 500)
active = c4.number_input("Active Mean", 0, 10000, 50)

if st.button("🔍 Analyze Traffic"):
    if st.session_state.model is None:
        st.warning("Please train the model first.")
    else:
        packets_per_sec = packets / (flow + 1)

        input_data = np.array([[ 
            80,                  # Destination Port
            flow,
            packets,
            pkt_len,
            active,
            packets_per_sec
        ]])

        prediction = st.session_state.model.predict(input_data)[0]

        if prediction == 1:
            st.error("🚨 ATTACK DETECTED")
        else:
            st.success("✅ BENIGN TRAFFIC")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.markdown("<center>🔐 Machine Learning • Streamlit • CIC-IDS Dataset</center>", unsafe_allow_html=True)
