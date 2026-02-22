
import streamlit as st
from memory_trace_io import export_memory, export_trace, reingest_memory

# Sample memory/traces for demo/testing purposes
sample_memory = [
    {"id": "k001", "type": "fact", "content": "The Earth orbits the Sun.", "belief_score": 0.98, "source": "NASA"},
    {"id": "k002", "type": "skill", "content": "Python supports list comprehensions.", "belief_score": 0.95, "source": "Docs"}
]
sample_traces = [
    {"question": "Why is the sky blue?", "steps": ["Check atmosphere", "Analyze Rayleigh scattering", "Conclude"]},
    {"question": "What is 2+2?", "steps": ["Simple arithmetic", "Result: 4"]}
]

def show_memory_trace_controls():
    st.subheader("📤 Export Memory & Traces")

    if st.button("Export Sample Memory"):
        msg = export_memory(sample_memory)
        st.success(msg)

    if st.button("Export Sample Traces"):
        msg = export_trace(sample_traces)
        st.success(msg)

    st.divider()
    st.subheader("📥 Re-Ingest Memory (from review_hook.json)")
    if st.button("Re-Ingest Reviewed Memory"):
        result = reingest_memory()
        st.info(result)
