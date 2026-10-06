import os
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

# Load trained models
clf = joblib.load("../models/business_intent_classifier.pkl")

embedding_model = SentenceTransformer("../models/all-MiniLM-L6-v2")

# Load historical complaint data
complaint_embeddings = np.load("../models/all_embeddings.npy")

historical_data = pd.read_pickle("../models/historical_complaints.pkl")

class OperationsIntelligence(BaseModel):
    complaint_summary: str
    historical_pattern: str
    historical_outcomes: str
    recommended_operational_action: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "UNCERTAIN"]

llm = ChatOpenAI(
    model="openai/gpt-4.1", # Replace with the model available in your evaluation environment
    temperature=0,
    api_key=os.environ["LITELLM_API_KEY"],   # Set this environment variable to your API key
    base_url="https://management.llmproxy.ai.orange"    # REPLACE with your LLM provider/proxy API endpoint 
)

# Build similarity search model
similarity_model = NearestNeighbors(
    n_neighbors=5,
    metric="cosine",
    algorithm="brute"
)

similarity_model.fit(complaint_embeddings)


def extract_risk_signals(complaint_text):
    text = complaint_text.lower()

    signals = {
        "unauthorized_transaction": any(
            phrase in text
            for phrase in [
                "don't recognize",
                "do not recognize",
                "didn't make",
                "did not make",
                "not authorized",
                "unauthorized",
                "unrecognized"
            ]
        ),

        "financial_loss_reported": any(
            phrase in text
            for phrase in [
                "lost money",
                "money was taken",
                "money was withdrawn",
                "withdrawal",
                "charged",
                "deducted",
                "stolen"
            ]
        ),

        "refund_requested_or_missing": any(
            phrase in text
            for phrase in [
                "refund",
                "refunded",
                "money back",
                "reimburse",
                "reimbursement"
            ]
        )
    }

    return signals


def analyze_complaint(
    complaint_text,
    product=None,
    issue=None,
    sub_issue=None,
    submitted_via=None,
    state=None
):
    # Build the same text format used during Task 1
    model_text = (
        "Product: " + (product or "") +
        " | Complaint: " + complaint_text
    )

    # Create embedding
    embedding = embedding_model.encode([model_text])

    # Predict business intent
    predicted_intent = clf.predict(embedding)[0]

    # Extract risk signals from complaint text 
    risk_signals = extract_risk_signals(complaint_text)

    # Get probabilities
    probabilities = clf.predict_proba(embedding)[0]
    classes = clf.classes_

    # Get confidence
    confidence = probabilities.max()

    # Get similar complaints 
    similar_complaints = find_similar_complaints(embedding,n=5)

    rag_context = build_rag_context(similar_complaints)

    return {
        "complaint_text": complaint_text,
        "product": product,
        "issue": issue,
        "sub_issue": sub_issue,
        "submitted_via": submitted_via,
        "state": state,
        "business_intent": predicted_intent,
        "classification_confidence": float(confidence),
        "risk_signals": risk_signals,
        "embedding": embedding,
        "similar_complaints": similar_complaints,
        "rag_context": rag_context
    }

def find_similar_complaints(embedding, n=5):
    distances, indices = similarity_model.kneighbors(
        embedding,
        n_neighbors=n
    )

    results = historical_data.iloc[indices[0]].copy()

    results["similarity"] = 1 - distances[0]

    return results


def build_rag_context(similar_complaints):
    context = []

    for i, (_, row) in enumerate(similar_complaints.iterrows(), start=1):
        case = (
            f"Historical Case {i}\n" 
            f"Similarity: {row['similarity']:.4f}\n" 
            f"Complaint ID: {row['complaint_id']}\n" 
            f"Product: {row['product']}\n" 
            f"Business Intent: {row['business_intent']}\n" 
            f"Company Response: {row['company_response']}\n" 
            f"Complaint: {row['complaint_what_happened']}\n"
        )

        context.append(case)

    return "\n" + "\n".join(context)


def build_operations_prompt(analysis):
    prompt = f"""
You are an operations intelligence assistant for a bank.

Your task is to analyze a new customer complaint using ONLY:
1. The information contained in the new complaint.
2. The classification results provided.
3. The historical complaint evidence provided.

IMPORTANT RULES
---------------
- Do not invent facts, policies, procedures, regulations, timelines, or outcomes.
- Do not assume that two complaints have the same root cause simply because they are similar.
- Treat similarity scores as evidence of relevance, not proof of equivalence.
- Treat historical company responses as historical observations, not current bank policy.
- Do not claim that a customer is entitled to a refund, compensation, or monetary relief unless the provided evidence explicitly establishes that.
- Do not make legal or regulatory conclusions unless the provided evidence explicitly supports them.
- Clearly distinguish facts, historical evidence, and recommendations.
- If the available evidence is insufficient to make a determination, explicitly say so.

NEW COMPLAINT
-------------
Complaint:
{analysis["complaint_text"]}

Product:
{analysis["product"]}

Business Intent:
{analysis["business_intent"]}

Classification Confidence:
{analysis["classification_confidence"]:.4f}

Risk Signals:
{analysis["risk_signals"]}

HISTORICAL COMPLAINT EVIDENCE
-----------------------------
{analysis["rag_context"]}

Based ONLY on the information above, produce the following:

1. COMPLAINT SUMMARY
   - Summarize the customer's reported problem.
   - Include only facts explicitly stated in the complaint.

2. HISTORICAL PATTERN
   - Identify the most relevant patterns in the retrieved historical complaints.
   - Explain why those cases are relevant.
   - Do not assume the underlying cause is identical.

3. HISTORICAL OUTCOMES
   - Summarize the company responses shown in the historical cases.
   - Clearly distinguish observed historical outcomes from current recommendations.
   - Do not infer details that are not present in the company response.

4. RECOMMENDED OPERATIONAL ACTION
   - Suggest a reasonable next operational step based on the available evidence.
   - Frame this as a recommendation, not as an established bank policy.
   - If the evidence is insufficient to recommend a specific action, say what information should be reviewed first.

5. RISK / ATTENTION LEVEL
   - Assign one of:
     LOW
     MEDIUM
     HIGH
     UNCERTAIN
   - Use the provided Risk Signals as structured evidence.
   - Consider the classification confidence when assessing how reliable the classification-based evidence is.
   - If the evidence is conflicting, insufficient, or the classification confidence is low, use UNCERTAIN rather than forcing a LOW, MEDIUM, or HIGH decision.
   - Explain the rating using ONLY:
     1. The complaint text.
     2. The Risk Signals.
     3. Classification confidence.
     4. Historical complaint evidence.
   - Do not infer general banking practices or state that a type of complaint "typically" requires a particular level of attention unless that is explicitly supported by the provided evidence.
   - Do not introduce unsupported legal, regulatory, fraud, financial-risk, or urgency claims.
   - The risk level is an operational attention assessment based on the supplied evidence, not a legal or regulatory risk determination.

OUTPUT FORMAT
-------------
Use exactly these sections:

1. Complaint Summary
2. Historical Pattern
3. Historical Outcomes
4. Recommended Operational Action
5. Risk / Attention Level

Within each section, clearly distinguish:
- FACT
- HISTORICAL EVIDENCE
- RECOMMENDATION

Do not add information that is not supported by the provided evidence.
"""
    return prompt


#def determine_human_review(operations_intelligence):
#    text = operations_intelligence.upper()

#    if "RISK LEVEL: HIGH" in text:
#        return True

#    if "RISK LEVEL: UNCERTAIN" in text:
#        return True

#    return False

def generate_operations_intelligence(analysis):
    prompt = build_operations_prompt(analysis)

    structured_llm = llm.with_structured_output(
        OperationsIntelligence
    )

    response = structured_llm.invoke(prompt)

    human_review_required = response.risk_level in {
        "HIGH",
        "UNCERTAIN"
    }

    return {
        "operations_intelligence": response,
        "human_review_required": human_review_required
    }