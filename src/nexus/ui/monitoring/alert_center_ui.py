
import streamlit as st
from anomaly_detector import detect_anomalies

def show_alert_center():
    st.title("🚨 Alert Center")

    alerts = detect_anomalies()
    if not alerts:
        st.success("✅ System operating within expected parameters.")
        return

    for level, message in alerts:
        if level == "danger":
            st.error(f"🔴 {message}")
        elif level == "warning":
            st.warning(f"🟠 {message}")
        elif level == "info":
            st.info(f"🔵 {message}")
