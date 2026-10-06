import streamlit as st

from complaint_analyzer import analyze_complaint
from complaint_analyzer import generate_operations_intelligence


st.set_page_config(
    page_title="Complaint Operations Intelligence",
    page_icon="🏦",
    layout="centered"
)

col1, col2 = st.columns([1, 5])

with col1:
    st.image("logo.png", width=80)

with col2:
    st.title("Complaint Operations Intelligence")
    
st.write(
    "Enter a customer complaint below to analyze its business intent, "
    "historical patterns, and operational risk."
)

complaint_text = st.text_area(
    "Customer complaint",
    placeholder="Paste the customer complaint here...",
    height=250
)

product = st.text_input(
    "Product (optional)",
    placeholder="e.g. Checking or savings account"
)

if st.button("Analyze Complaint"):

    if not complaint_text.strip():
        st.warning("Please enter a complaint first.")

    else:
        with st.spinner("Analyzing complaint..."):

            analysis = analyze_complaint(
                complaint_text=complaint_text,
                product=product if product.strip() else None
            )

            intelligence = generate_operations_intelligence(
                analysis
            )

        st.success("Analysis complete.")

        st.subheader("Business Intent")
        st.write(analysis["business_intent"])

        st.subheader("Classification Confidence")
        st.write(
            f"{analysis['classification_confidence']:.2%}"
        )

        st.subheader("Risk Signals")
        st.write(analysis["risk_signals"])

        st.subheader("Operations Intelligence")
        st.write(
            intelligence["operations_intelligence"]
        )

        st.subheader("Human Review Required")

        if intelligence["human_review_required"]:
            st.error("YES")
        else:
            st.success("NO")