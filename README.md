# Complaint Operations Intelligence

## Overview

This project implements an AI-powered **Complaint Operations Intelligence** application for analyzing customer complaints.

The application combines:

- A supervised machine-learning classifier for business-intent classification
- SentenceTransformer embeddings
- Historical complaint similarity search
- Retrieval-Augmented Generation (RAG)
- An LLM for operational analysis and recommendations
- Structured risk/attention assessment
- Human-review flagging for `HIGH` and `UNCERTAIN` cases
- A Streamlit user interface

The project is divided into two conceptual tasks:

### Task 1 — Data Preparation and Machine Learning

The complaint data was cleaned, embedded, clustered/topic-modeled, and used to train a supervised business-intent classifier.

The trained artifacts required by Task 2 are stored in the `models/` directory.

### Task 2 — Complaint Operations Intelligence

For a new complaint, the application:

1. Accepts complaint text and optional metadata.
2. Classifies the complaint into a business intent.
3. Calculates classification confidence.
4. Extracts structured risk signals.
5. Finds similar historical complaints.
6. Builds historical evidence for RAG.
7. Sends the evidence to an LLM.
8. Produces structured operational intelligence.
9. Determines whether human review is required.

---

# Project Structure

Expected structure:

```text
your-project/
│
├── task1/
│   └── DataCleaning.ipynb
│   └── DataEmbeddings.ipynb
│   └── complaints_flat.csv
├── task2/
│   ├── complaint_analyzer.py
│   ├── test_complaint_analyzer.py
│   └── app.py
│
├── models/
│   ├── business_intent_classifier.pkl
│   ├── all-MiniLM-L6-v2/
│   ├── all_embeddings.npy
│   └── historical_complaints.pkl
│
├── screenshots/
│
└── README.md
└── requirements.txt
```
---

# Requirements

The application requires:

- Python 3.10 or later
- Internet/network access to the configured LLM endpoint
- Access to an OpenAI-compatible LLM API or proxy
- A valid API key

Main Python dependencies:

- `numpy`
- `pandas`
- `scikit-learn`
- `sentence-transformers`
- `joblib`
- `langchain-openai`
- `pydantic`
- `streamlit`
- `torch`
- `torchvision`

---

# 1. Create a Python Environment

From the project root:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 2. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Otherwise:

```bash
pip install numpy pandas scikit-learn sentence-transformers joblib langchain-openai pydantic streamlit torch torchvision
```

---

# 3. Verify the Model Files

The following files are required:

```text
models/
├── business_intent_classifier.pkl
├── all-MiniLM-L6-v2/
├── all_embeddings.npy
└── historical_complaints.pkl
```

| File | Purpose |
|---|---|
| `business_intent_classifier.pkl` | Trained business-intent classifier |
| `all-MiniLM-L6-v2/` | Saved SentenceTransformer embedding model |
| `all_embeddings.npy` | Historical complaint embeddings |
| `historical_complaints.pkl` | Historical complaint data used for similarity search |

**Important:** `all_embeddings.npy` and `historical_complaints.pkl` must remain aligned by row order. Do not independently reorder the historical dataframe.

---

# 4. Configure the API Key

The application does **not** store the API key in the source code.

The code expects:

```python
api_key=os.environ["LITELLM_API_KEY"]
```

The evaluator must provide the API key through an environment variable.

### Windows PowerShell

```powershell
$env:LITELLM_API_KEY="YOUR_API_KEY"
```

### Windows Command Prompt

```cmd
set LITELLM_API_KEY=YOUR_API_KEY
```

### macOS / Linux

```bash
export LITELLM_API_KEY="YOUR_API_KEY"
```

**Do not commit or submit a real API key.**

---

# 5. Configure the LLM Endpoint

The current code contains:

```python
base_url="https://management.llmproxy.ai.orange"
```

> **IMPORTANT FOR THE TESTER:** This endpoint is environment-specific. If the evaluation environment does not have access to it, replace it with the LLM API/proxy endpoint available to you.

For example:

```python
base_url="YOUR_LLM_PROVIDER_OR_PROXY_ENDPOINT"
```

The configured model is currently:

```python
model="openai/gpt-4.1"
```

If this model is not available in the evaluation environment, replace it with the model provided by the evaluator's LLM service.

### Recommended portable configuration

For a more portable deployment, the LLM configuration can be changed to:

```python
llm = ChatOpenAI(
    model=os.environ["LLM_MODEL"],
    temperature=0,
    api_key=os.environ["LITELLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"]
)
```

Then configure:

```powershell
$env:LITELLM_API_KEY="YOUR_API_KEY"
$env:LLM_BASE_URL="YOUR_LLM_ENDPOINT"
$env:LLM_MODEL="YOUR_MODEL"
```

---

# 6. Run the Command-Line Test

Before starting Streamlit, verify the analysis pipeline:

```bash
python task2/test_complaint_analyzer.py
```

Alternatively:

```bash
cd task2
python test_complaint_analyzer.py
```

A successful test should produce output containing information such as:

```text
Business intent: Bank Account / Savings
Confidence: 0.94
Risk signals: ...
Historical complaints: ...
Operations Intelligence: ...
Risk level: HIGH
Human review required: True
```

Exact values depend on the test complaint and model configuration.

---

# 7. Run the Streamlit Application

From the project root:

```bash
streamlit run task2/app.py
```

Alternatively:

```bash
cd task2
streamlit run app.py
```

Streamlit will display a local URL, normally similar to:

```text
Local URL: http://localhost:8501
```

Open the displayed URL in a browser.

---

# 8. Using the Application

The application accepts a new customer complaint and product information.

### Required

- Complaint text

### Optional metadata

Depending on the UI:

- Product
- Issue
- Sub-issue
- Submission channel
- State

After submission, the complaint is processed through the complete Task 2 pipeline.

---

# 9. Processing Pipeline

```text
Customer Complaint
        |
        v
Input Processing
        |
        +----------------------> Business Intent Classifier
        |                                  |
        |                                  v
        |                           Intent + Confidence
        |
        +----------------------> Risk Signal Extraction
        |
        +----------------------> Similarity Search
                                           |
                                           v
                                  Historical Complaints
                                           |
                                           v
                                       RAG Context
                                           |
                                           v
                                          LLM
                                           |
                                           v
                              Operations Intelligence
                                           |
                       +-------------------+-------------------+
                       |                   |                   |
                       v                   v                   v
                    Summary          Historical          Risk / Attention
                                     Evidence
                       |
                       v
                Recommended Action
                       |
                       v
                Human Review Flag
```

---

# 10. Business Intent Classification

The classifier was trained during Task 1.

The application constructs the same model input format used during training:

```text
Product: <product> | Complaint: <complaint text>
```

The classifier returns:

- Predicted business intent
- Classification confidence

Example:

```text
Business Intent:
Bank Account / Savings

Classification Confidence:
0.9404
```

---

# 11. Risk Signal Extraction

The application extracts structured signals from the complaint text.

### Unauthorized transaction

Examples of detected phrases:

```text
don't recognize
do not recognize
didn't make
did not make
not authorized
unauthorized
unrecognized
```

### Financial loss reported

Examples:

```text
lost money
money was taken
money was withdrawn
withdrawal
charged
deducted
stolen
```

### Refund requested or missing

Examples:

```text
refund
refunded
money back
reimburse
reimbursement
```

These signals are supplied to the LLM as structured evidence.

---

# 12. Historical Similarity Search

The application uses the complaint embedding to retrieve similar historical complaints.

The current implementation retrieves:

```python
n_neighbors=5
```

using cosine distance.

Similarity is calculated as:

```python
similarity = 1 - cosine_distance
```

Historical records provide evidence such as:

- Complaint ID
- Product
- Business intent
- Company response
- Original complaint text
- Similarity score

---

# 13. RAG Context

The retrieved historical complaints are converted into structured context and provided to the LLM.

The LLM is instructed to treat historical company responses as **historical observations**, not as current bank policy.

The prompt explicitly prevents unsupported claims about:

- Policies
- Regulations
- Timelines
- Refund eligibility
- Compensation eligibility
- Legal conclusions
- Outcomes not present in the historical evidence

---

# 14. Operations Intelligence Output

The LLM produces structured output containing:

```text
1. Complaint Summary
2. Historical Pattern
3. Historical Outcomes
4. Recommended Operational Action
5. Risk / Attention Level
```

The allowed risk levels are:

```text
LOW
MEDIUM
HIGH
UNCERTAIN
```

---

# 15. Human Review

Human review is automatically required when the LLM returns:

```text
HIGH
```

or:

```text
UNCERTAIN
```

Therefore:

```python
human_review_required = True
```

for those cases.

This provides a human-in-the-loop mechanism instead of treating the LLM output as an autonomous final decision.

---

# 16. Example Test Complaint

Use the following example to test the application:

```text
I noticed a cash withdrawal on my account that I don't recognize.
I never made this transaction and Chase has not refunded the money.
```

Product:

```text
Checking or savings account
```

This complaint should trigger signals related to:

- Unauthorized transaction
- Financial loss
- Missing/refund request

The exact classification, retrieved cases, LLM response, and risk level may vary depending on the configured model and historical evidence.

---

# 17. Troubleshooting

## `KeyError: 'LITELLM_API_KEY'`

The API key environment variable has not been configured.

Set it before running the application:

```powershell
$env:LITELLM_API_KEY="YOUR_API_KEY"
```

---

## Unable to connect to the LLM endpoint

Check:

1. The configured `base_url`
2. Network/internet access
3. API key validity
4. Whether the evaluator has access to the configured endpoint
5. Whether the configured model is available through the endpoint

If the Orange proxy is unavailable, replace:

```python
base_url="https://management.llmproxy.ai.orange"
```

with the endpoint provided for the evaluation environment.

---

## Model file not found

Verify:

```text
models/business_intent_classifier.pkl
models/all-MiniLM-L6-v2/
models/all_embeddings.npy
models/historical_complaints.pkl
```

exist in the expected location.

---

## `torchvision` import error

Install it with:

```bash
python -m pip install torchvision
```

Then verify:

```bash
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__)"
```

---

## Streamlit does not start

Install Streamlit:

```bash
python -m pip install streamlit
```

Then run:

```bash
streamlit run task2/app.py
```

---

# 18. Security Notes

### API credentials

Do not submit a real API key.

The project uses:

```python
os.environ["LITELLM_API_KEY"]
```

so credentials can be supplied by the evaluation environment.

### LLM endpoint

The Orange proxy URL is environment-specific. The evaluator should replace it if a different provider or proxy is required.

### Historical data

The historical complaint files are used locally for similarity search and RAG. Include them in the submission only if permitted by the assessment and data-sharing requirements.

---

# 19. Quick Start for the Evaluator

```bash
# Create environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
$env:LITELLM_API_KEY="YOUR_API_KEY"

# Configure the LLM endpoint/model if required
# Update complaint_analyzer.py or use environment variables
# as described above.

# Run pipeline test
python task2/test_complaint_analyzer.py

# Start application
streamlit run task2/app.py
```

Then open the Streamlit URL shown in the terminal.

---
