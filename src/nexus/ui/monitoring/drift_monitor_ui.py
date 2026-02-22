
import streamlit as st
import pandas as pd
import altair as alt
from feedback_stats import compute_feedback_statistics

def show_drift_monitor():
    st.title("📉 Drift & Feedback Monitoring")

    model_df, feedback_df, trend_df = compute_feedback_statistics()

    if model_df.empty:
        st.warning("No feedback data available yet.")
        return

    st.subheader("🔁 Model Selection Frequency")
    chart1 = alt.Chart(model_df).mark_bar().encode(
        x=alt.X('Model:N', sort='-y'),
        y='Count:Q',
        tooltip=['Model', 'Count']
    ).properties(width=600)
    st.altair_chart(chart1)

    st.subheader("👍 User Feedback Distribution")
    chart2 = alt.Chart(feedback_df).mark_bar().encode(
        x='Feedback:N',
        y='Count:Q',
        color='Feedback:N',
        tooltip=['Feedback', 'Count']
    ).properties(width=600)
    st.altair_chart(chart2)

    st.subheader("📆 Model Drift Over Time")
    drift_trend = trend_df.groupby(['timestamp', 'model']).size().reset_index(name='count')
    chart3 = alt.Chart(drift_trend).mark_line().encode(
        x='timestamp:T',
        y='count:Q',
        color='model:N',
        tooltip=['timestamp', 'model', 'count']
    ).properties(width=800)
    st.altair_chart(chart3)
