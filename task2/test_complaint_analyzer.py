from complaint_analyzer import analyze_complaint, build_operations_prompt, generate_operations_intelligence


new_complaint = analyze_complaint(
    complaint_text=(
        "I noticed a cash withdrawal on my account that I don't recognize. "
        "I never made this transaction and Chase has not refunded the money."
    ),
    product="Checking or savings account"
)

print("\nBusiness intent:")
print(new_complaint["business_intent"])

print("\nConfidence:")
print(new_complaint["classification_confidence"])

print("\nRetrieved complaint text:")

for _, row in new_complaint["similar_complaints"].iterrows():
    print("\n" + "=" * 80)
    print("Complaint ID:", row["complaint_id"])
    print("Similarity:", round(row["similarity"], 4))
    print("Business intent:", row["business_intent"])
    print("Company response:", row["company_response"])
    print("\nComplaint:")
    print(row["clean_text"])

print("\nRAG context:")
print(new_complaint["rag_context"])

print("\nOperations Intelligence Prompt:") 
print(build_operations_prompt(new_complaint))


print("\nOperations Intelligence:")
print(generate_operations_intelligence(new_complaint))

print("\nRisk signals:")
print(new_complaint["risk_signals"])