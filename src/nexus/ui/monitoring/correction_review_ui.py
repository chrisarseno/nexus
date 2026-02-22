
import streamlit as st
import pandas as pd

def show_correction_review_panel():
    st.title("✅ Correction Review Panel")

    # Example data of flagged corrections
    corrections = [
        {"Item": "Gravity pulls objects downward", "Original": "Factual", "Suggested": "Skill", "Reviewer": "", "Approved": False},
        {"Item": "Write a basic SQL query", "Original": "Factual", "Suggested": "Skill", "Reviewer": "", "Approved": False},
        {"Item": "Sun rises in the East", "Original": "Factual", "Suggested": "Factual", "Reviewer": "", "Approved": False},
    ]
    df = pd.DataFrame(corrections)

    st.dataframe(df)

    st.markdown("### 🧐 Review and Approve a Correction")
    item_to_review = st.selectbox("Select Item", df["Item"])
    reviewer_name = st.text_input("Your Name")
    approve = st.checkbox("Approve suggested correction")

    if st.button("Submit Review"):
        idx = df[df["Item"] == item_to_review].index[0]
        df.at[idx, "Reviewer"] = reviewer_name
        df.at[idx, "Approved"] = approve
        st.success(f"Correction review recorded for: {item_to_review}")
        st.dataframe(df)
