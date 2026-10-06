# Complaint Analyzer — Technical Documentation

## 1. Overview

`complaint_analyzer.py` is the core analytical component of the **Complaint Operations Intelligence** layer.

The module takes a new customer complaint and combines:

1. Machine-learning classification
2. Sentence embeddings
3. Historical complaint similarity search
4. Retrieval-Augmented Generation (RAG)
5. Risk signal extraction
6. Large Language Model (LLM) reasoning
7. Structured operational output

The objective is to transform an unstructured customer complaint into actionable operational intelligence while grounding the generated output in both the trained classification model and historical complaint data.

### High-level workflow

```text
New Complaint
      │
      ▼
Create Model Input
      │
      ▼
Sentence Embedding
      │
      ├──────────────► Business Intent Classification
      │                         │
      │                         ▼
      │                  Classification Confidence
      │
      ├──────────────► Risk Signal Extraction
      │
      └──────────────► Similarity Search
                                │
                                ▼
                     Historical Complaints
                                │
                                ▼
                         RAG Context
                                │
                                ▼
                       Operations Prompt
                                │
                                ▼
                              LLM
                                │
                                ▼
                Structured Operations Intelligence
                                │
                                ▼
                    Human Review Decision
```

---

## 2. Dependencies

The module uses the following libraries:

| Library | Purpose |
|---|---|
| `os` | Access environment variables such as the LLM API key |
| `joblib` | Load the trained classification model |
| `numpy` | Load and manipulate complaint embeddings |
| `pandas` | Load and manipulate historical complaint data |
| `sentence-transformers` | Generate semantic embeddings |
| `scikit-learn` | Perform nearest-neighbor similarity search |
| `langchain-openai` | Connect to the configured LLM |
| `pydantic` | Define structured LLM output |
| `typing` | Define allowed risk-level values |

---

## 3. Input Artifacts

The analyzer depends on models and historical data generated during Task 1.

The expected project structure is:

```text
project/
│
├── models/
│   ├── business_intent_classifier.pkl
│   ├── all-MiniLM-L6-v2/
│   ├── all_embeddings.npy
│   └── historical_complaints.pkl
│
└── task2/
    └── complaint_analyzer.py
```

### `business_intent_classifier.pkl`

A trained supervised classification model used to predict the business intent of a complaint.

The classifier was trained using complaint embeddings generated from the `all-MiniLM-L6-v2` model.

### `all-MiniLM-L6-v2`

The Sentence Transformer model used to convert complaint text into numerical vector representations.

The embedding dimension is 384.

The same embedding model must be used during inference as was used during model training.

### `all_embeddings.npy`

Contains embeddings for the historical complaints used for similarity search.

These embeddings provide the vector representation of the historical complaint dataset.

### `historical_complaints.pkl`

Contains the historical complaint records associated with the embeddings.

The historical records are used to retrieve contextual information such as:

- Complaint ID
- Product
- Business intent
- Original complaint text
- Company response

The ordering of this dataset must remain aligned with `all_embeddings.npy`.

---

## 4. Model Initialization

### 4.1 Loading the Classifier

```python
clf = joblib.load("../models/business_intent_classifier.pkl")
```

The previously trained business-intent classifier is loaded from disk.

This allows the Task 2 application to reuse the model developed during Task 1 rather than retraining it every time the application starts.

### 4.2 Loading the Embedding Model

```python
embedding_model = SentenceTransformer(
    "../models/all-MiniLM-L6-v2"
)
```

The Sentence Transformer converts the complaint and product information into a semantic vector.

The same model is used during both training and inference so that generated embeddings remain compatible with the trained classifier and historical embeddings.

### 4.3 Loading Historical Data

```python
complaint_embeddings = np.load(
    "../models/all_embeddings.npy"
)

historical_data = pd.read_pickle(
    "../models/historical_complaints.pkl"
)
```

Two components are loaded:

- historical embeddings for similarity search
- historical complaint records for retrieving contextual information

The embeddings and historical records correspond to the same complaint records.

---

## 5. Structured Operations Intelligence Model

The application defines a Pydantic model:

```python
class OperationsIntelligence(BaseModel):
    complaint_summary: str
    historical_pattern: str
    historical_outcomes: str
    recommended_operational_action: str
    risk_level: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
        "UNCERTAIN"
    ]
```

This defines the expected output from the LLM.

### Output fields

#### `complaint_summary`

A concise summary of the customer's reported problem.

#### `historical_pattern`

Describes relevant patterns found among the retrieved historical complaints.

#### `historical_outcomes`

Summarizes the company responses associated with similar historical complaints.

#### `recommended_operational_action`

Provides a suggested operational next step based on the available evidence.

#### `risk_level`

Represents the operational attention level assigned to the complaint.

Only four values are permitted:

```text
LOW
MEDIUM
HIGH
UNCERTAIN
```

Using a constrained set of values makes the LLM output easier to consume programmatically and allows the application to trigger downstream actions such as human review.

---

## 6. LLM Configuration

```python
llm = ChatOpenAI(
    model="openai/gpt-4.1",
    temperature=0,
    api_key=os.environ["LITELLM_API_KEY"],
    base_url="https://management.llmproxy.ai.orange"
)
```

The LLM is used for the operations intelligence layer, rather than for the initial complaint classification.

### Temperature

```python
temperature=0
```

A low temperature is used to make the generated output more deterministic and consistent.

### API Key

The API key is retrieved from an environment variable:

```python
os.environ["LITELLM_API_KEY"]
```

This avoids hard-coding credentials in the source code.

### Proxy

The configured `base_url` routes requests through the organization's LLM proxy.

---

## 7. Similarity Search

The application creates a nearest-neighbor search model:

```python
similarity_model = NearestNeighbors(
    n_neighbors=5,
    metric="cosine",
    algorithm="brute"
)
```

The system uses cosine distance because complaint embeddings represent semantic meaning.

The model is fitted on the historical complaint embeddings:

```python
similarity_model.fit(complaint_embeddings)
```

This creates the retrieval layer used by the RAG pipeline.

---

## 8. Risk Signal Extraction

### Function

```python
extract_risk_signals(complaint_text)
```

This function identifies predefined signals directly from the complaint text.

The text is first normalized:

```python
text = complaint_text.lower()
```

The function currently extracts three categories of signals.

### 8.1 Unauthorized Transaction

The function searches for phrases such as:

```text
don't recognize
do not recognize
didn't make
did not make
not authorized
unauthorized
unrecognized
```

If one of these phrases is present:

```python
"unauthorized_transaction": True
```

Otherwise:

```python
"unauthorized_transaction": False
```

### 8.2 Financial Loss

The function checks for expressions including:

```text
lost money
money was taken
money was withdrawn
withdrawal
charged
deducted
stolen
```

This produces:

```python
"financial_loss_reported": True/False
```

### 8.3 Refund Request or Missing Refund

The function searches for:

```text
refund
refunded
money back
reimburse
reimbursement
```

This produces:

```python
"refund_requested_or_missing": True/False
```

### 8.4 Why Risk Signals Are Separate

Risk signals are intentionally extracted separately from the machine-learning classifier.

The classifier answers:

> What business intent does this complaint most closely represent?

Risk signals answer:

> What explicit indicators of operational attention are present in the complaint text?

This separation provides additional structured evidence to the LLM.

---

## 9. Main Complaint Analysis Function

### Function

```python
analyze_complaint(
    complaint_text,
    product=None,
    issue=None,
    sub_issue=None,
    submitted_via=None,
    state=None
)
```

This is the main entry point for analyzing a new complaint.

Only `complaint_text` is required.

The remaining fields are optional metadata.

---

## 10. Creating the Model Input

The function constructs:

```python
model_text = (
    "Product: " + (product or "") +
    " | Complaint: " + complaint_text
)
```

This is important because the classifier was trained using the same format.

For example:

```text
Product: Checking or savings account |
Complaint: I noticed a cash withdrawal...
```

Including the product provides additional context to the semantic representation.

Maintaining the same input format between training and inference helps ensure consistency between Task 1 and Task 2.

---

## 11. Generating the Complaint Embedding

```python
embedding = embedding_model.encode([model_text])
```

The text is converted into a numerical vector using the Sentence Transformer.

The resulting embedding is then used for two purposes:

1. Business-intent classification
2. Historical similarity search

This means the same semantic representation powers both prediction and retrieval.

---

## 12. Business Intent Classification

```python
predicted_intent = clf.predict(embedding)[0]
```

The trained classifier predicts the business intent.

For example:

```text
Bank Account / Savings
```

The classifier is responsible for the primary categorization of the complaint.

---

## 13. Classification Confidence

The analyzer also retrieves the classifier probabilities:

```python
probabilities = clf.predict_proba(embedding)[0]
```

The highest probability is used as the classification confidence:

```python
confidence = probabilities.max()
```

For example:

```text
0.9404
```

This means the classifier assigned its highest probability of approximately 94% to the predicted class.

The confidence value is passed to the LLM as supporting evidence.

---

## 14. Risk Signal Extraction

The complaint is passed to:

```python
risk_signals = extract_risk_signals(complaint_text)
```

Example output:

```python
{
    "unauthorized_transaction": True,
    "financial_loss_reported": True,
    "refund_requested_or_missing": True
}
```

These signals provide explicit evidence from the complaint itself.

---

## 15. Historical Similarity Search

The complaint embedding is passed to:

```python
find_similar_complaints(
    embedding,
    n=5
)
```

The system retrieves the five most similar historical complaints.

The similarity search uses cosine distance.

The returned distance is converted into a similarity score:

```python
results["similarity"] = 1 - distances[0]
```

Therefore:

```text
Higher similarity → more semantically similar
Lower similarity → less semantically similar
```

---

## 16. Finding Similar Complaints

### Function

```python
find_similar_complaints(embedding, n=5)
```

The function performs:

```python
distances, indices = similarity_model.kneighbors(
    embedding,
    n_neighbors=n
)
```

The corresponding historical records are retrieved using:

```python
historical_data.iloc[indices[0]]
```

The similarity score is then added to the retrieved records.

The resulting dataframe contains both:

- historical complaint information
- similarity score

This creates the retrieval component of the RAG pipeline.

---

## 17. Building the RAG Context

### Function

```python
build_rag_context(similar_complaints)
```

The retrieved complaints are transformed into structured textual context for the LLM.

Each historical case contains:

```text
Historical Case
Similarity
Complaint ID
Product
Business Intent
Company Response
Complaint
```

For example:

```text
Historical Case 1
Similarity: 0.8619
Complaint ID: 3537927
Product: Checking or savings account
Business Intent: Bank Account / Savings
Company Response: Closed with explanation
Complaint: ...
```

The cases are combined into one context string.

---

## 18. RAG Architecture

The retrieval process can be summarized as:

```text
New Complaint
      │
      ▼
Embedding
      │
      ▼
Vector Similarity Search
      │
      ▼
Top 5 Historical Complaints
      │
      ▼
Historical Context
      │
      ▼
LLM
```

The LLM does not directly search the entire historical dataset.

Instead, Python performs the retrieval and provides the relevant historical evidence to the LLM.

This approach helps keep the generated response grounded in the available complaint data.

---

## 19. Building the Operations Prompt

### Function

```python
build_operations_prompt(analysis)
```

This function creates the prompt sent to the LLM.

The prompt contains four major evidence sources:

### 1. New complaint

The original customer complaint.

### 2. Classification

The predicted business intent and classification confidence.

### 3. Risk signals

Explicit signals extracted from the complaint.

### 4. Historical evidence

The five retrieved similar complaints and their company responses.

---

## 20. Grounding and Hallucination Controls

A major component of the prompt is the set of grounding instructions.

The LLM is explicitly instructed:

> Do not invent facts, policies, procedures, regulations, timelines, or outcomes.

This is important because the system is intended to support operational decision-making.

The prompt also states that:

> Similarity scores are evidence of relevance, not proof of equivalence.

This prevents the model from assuming that two semantically similar complaints necessarily have the same underlying cause.

---

## 21. Historical Outcome Handling

Historical company responses are treated as observations rather than current policies.

For example, if a historical complaint says:

```text
Closed with monetary relief
```

the LLM is not allowed to conclude that the new customer is automatically entitled to monetary relief.

Instead, the historical response is presented as evidence of what occurred in that particular historical case.

This distinction helps prevent unsupported operational or policy conclusions.

---

## 22. Operational Recommendation

The LLM is asked to provide a recommended next operational action.

The prompt explicitly requires that this be presented as:

```text
RECOMMENDATION
```

rather than as an established bank policy.

If the available evidence is insufficient, the LLM is instructed to identify what information should be reviewed first.

---

## 23. Risk / Attention Assessment

The LLM assigns one of four values:

```text
LOW
MEDIUM
HIGH
UNCERTAIN
```

The assessment is based only on:

1. Complaint text
2. Risk signals
3. Classification confidence
4. Historical complaint evidence

The prompt explicitly prevents the LLM from introducing unsupported:

- legal conclusions
- regulatory conclusions
- fraud conclusions
- financial-risk claims
- urgency claims

The risk level therefore represents an **operational attention assessment**, not a legal or regulatory risk classification.

---

## 24. Structured LLM Output

The function uses:

```python
structured_llm = llm.with_structured_output(
    OperationsIntelligence
)
```

This forces the response into the predefined Pydantic schema.

The LLM response is therefore returned as:

```python
OperationsIntelligence(
    complaint_summary=...,
    historical_pattern=...,
    historical_outcomes=...,
    recommended_operational_action=...,
    risk_level=...
)
```

This is preferable to parsing free-form text because the application can reliably access individual fields.

---

## 25. Human Review Logic

After receiving the structured response:

```python
human_review_required = response.risk_level in {
    "HIGH",
    "UNCERTAIN"
}
```

Human review is required when:

```text
Risk = HIGH
```

or:

```text
Risk = UNCERTAIN
```

This introduces a **human-in-the-loop** mechanism.

The system therefore does not attempt to fully automate every operational decision.

Instead:

```text
LOW       → No mandatory human review
MEDIUM    → No mandatory human review
HIGH      → Human review required
UNCERTAIN → Human review required
```

---

## 26. Final Output

The `generate_operations_intelligence()` function returns:

```python
{
    "operations_intelligence": response,
    "human_review_required": human_review_required
}
```

The resulting object can be consumed directly by the Streamlit interface.

The UI can display:

- Complaint Summary
- Historical Pattern
- Historical Outcomes
- Recommended Operational Action
- Risk / Attention Level
- Human Review Required

---

## 27. End-to-End Example

Consider a complaint:

```text
I noticed a cash withdrawal on my account that I don't recognize.
I never made this transaction and Chase has not refunded the money.
```

The analyzer performs the following.

### Step 1 — Classification

```text
Business Intent:
Bank Account / Savings
```

### Step 2 — Confidence

```text
Classification Confidence:
0.9404
```

### Step 3 — Risk Signals

```python
{
    "unauthorized_transaction": True,
    "financial_loss_reported": True,
    "refund_requested_or_missing": True
}
```

### Step 4 — Retrieval

Five historically similar complaints are retrieved.

### Step 5 — RAG

The retrieved complaints and historical company responses are provided to the LLM.

### Step 6 — Operations Intelligence

The LLM generates a structured assessment containing:

```text
Complaint Summary
Historical Pattern
Historical Outcomes
Recommended Operational Action
Risk / Attention Level
```

### Step 7 — Human Review

If the resulting risk level is:

```text
HIGH
```

the system sets:

```python
human_review_required = True
```

---

## 28. Design Principles

The implementation follows several important design principles.

### Separation of responsibilities

The machine-learning model is responsible for **classification**.

The similarity model is responsible for **retrieval**.

The LLM is responsible for **reasoning over the supplied evidence and generating operational intelligence**.

### Evidence-grounded generation

The LLM is not given unrestricted responsibility for determining facts.

Instead, it receives:

```text
Complaint
+
Classification
+
Confidence
+
Risk Signals
+
Historical Evidence
```

and is explicitly instructed to operate within those boundaries.

### Human-in-the-loop

Cases assessed as:

```text
HIGH
```

or:

```text
UNCERTAIN
```

are flagged for human review.

This reduces the risk of treating an uncertain model-generated assessment as an autonomous operational decision.

### Reuse of Task 1 components

Task 2 reuses the artifacts produced during Task 1:

```text
Sentence Transformer
        ↓
Complaint Embeddings
        ↓
Business Intent Classifier
        ↓
Historical Complaint Dataset
```

This creates a direct connection between the data-science layer and the operations-intelligence layer.

---

## 29. Function Summary

| Function | Responsibility |
|---|---|
| `extract_risk_signals()` | Extract explicit risk indicators from complaint text |
| `analyze_complaint()` | Main analysis pipeline for a new complaint |
| `find_similar_complaints()` | Retrieve semantically similar historical complaints |
| `build_rag_context()` | Convert retrieved complaints into LLM context |
| `build_operations_prompt()` | Construct the grounded operations-intelligence prompt |
| `generate_operations_intelligence()` | Generate structured LLM output and determine human-review requirement |

---

## 30. Overall Architecture

```text
                         ┌─────────────────────┐
                         │   New Complaint     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Text Preparation   │
                         │ Product + Complaint │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Sentence Transformer│
                         │   Embedding Model   │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   ▼                ▼                ▼
          ┌────────────────┐ ┌──────────────┐ ┌───────────────┐
          │ Classification │ │ Risk Signals │ │ Similarity    │
          │     Model      │ │  Extraction  │ │    Search     │
          └───────┬────────┘ └──────┬───────┘ └───────┬───────┘
                  │                 │                 │
                  ▼                 ▼                 ▼
          Business Intent     Risk Evidence    Historical Cases
                  │                 │                 │
                  └─────────────────┼─────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │     RAG Context     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Operations LLM    │
                         │  Grounded Reasoning │
                         └──────────┬──────────┘
                                    │
                                    ▼
                     ┌───────────────────────────┐
                     │ Operations Intelligence   │
                     ├───────────────────────────┤
                     │ Complaint Summary         │
                     │ Historical Pattern        │
                     │ Historical Outcomes       │
                     │ Recommended Action        │
                     │ Risk / Attention Level    │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Human Review Flag   │
                         └─────────────────────┘
```

---

## 31. Conclusion

`complaint_analyzer.py` implements the operational intelligence layer of the complaint-processing solution.

It combines traditional machine-learning techniques with semantic retrieval and an LLM-based reasoning layer. The architecture deliberately separates **prediction, retrieval, evidence extraction, and language-model reasoning**, allowing the generated operational output to remain grounded in the available data.

The resulting pipeline transforms a raw customer complaint into a structured operational assessment that can be consumed by the Streamlit application and used to support human decision-making.
