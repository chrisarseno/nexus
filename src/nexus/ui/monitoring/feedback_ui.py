
import streamlit as st
from ensemble_core import run_ensemble
from feedback_logger import log_feedback

def show_feedback_dashboard():
    st.title("🗣️ Feedback Reasoning Panel")

    query = st.text_input("Ask a question to the ensemble:")
    if query:
        result = run_ensemble(query)
        chosen = result['chosen']
        ranked = result['ranked_responses']

        st.subheader("🏆 Selected Response")
        st.success(f"{chosen['model']} — {chosen['response']}")

        st.caption(f"🔍 Weighted Score: {chosen['weighted_score']} (Unweighted: {chosen['score']})")

        st.subheader("📊 Ranked Model Responses")
        for r in ranked:
            st.markdown(f"**{r['model']}** — *Score: {r['score']} | Weighted: {r['weighted_score']}*")
            st.code(r['response'])

        st.subheader("🧠 Why This Response?")
        st.write("The selected response had the highest weighted score, factoring both model confidence and manually assigned model weight.")

        st.subheader("🔁 Provide Feedback")
        feedback = st.radio("Do you agree with the selected response?", ["Yes", "No", "Partially"])
        comment = st.text_area("Optional comment:")
        if st.button("Submit Feedback"):
            log_feedback(query, chosen["model"], ranked, feedback, comment)
            st.success("✅ Feedback logged successfully!")
