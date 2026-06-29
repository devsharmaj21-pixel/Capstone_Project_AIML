import os
import re
import json
import logging
import requests
import joblib
import pandas as pd
import numpy as np
import jsonschema

# Initialize local logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# STEP 1: INITIALIZE LLM CONFIGURATION & API ENGINE
# =====================================================================
# Fetch key safely from environment variables (Strictly avoiding hardcoding)
API_KEY = os.environ.get('LLM_API_KEY', '')
# Using a generic provider endpoint like OpenRouter or direct OpenAi routes
LLM_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-flash"  # Flexible target configuration model

def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    """
    Executes an HTTP POST request to the LLM API endpoint with JSON payloads.
    """
    if not API_KEY:
        print("[ERROR] API Key is missing! Set your 'LLM_API_KEY' environment variable.")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            print(f"[API ERROR] HTTP Status: {response.status_code} | Text: {response.text}")
            return None
        
        result_json = response.json()
        return result_json['choices'][0]['message']['content']
    except Exception as e:
        print(f"[CONNECTION ERROR] Exception caught during API post: {str(e)}")
        return None

# Verification test run
print("--- VERIFICATION: TESTING BASIC LLM CONNECTIVITY ---")
test_res = call_llm("You are a literal echo.", "Reply with only the word: hello", temperature=0.0)
print(f"LLM API Handshake Response: {test_res}")


# =====================================================================
# STEP 2: DEFINE SYSTEM PROMPTS AND VALIDATION SCHEMAS
# =====================================================================
TRACK_C_SYSTEM_PROMPT = """You are an AI machine learning explainability assistant. 
Your task is to provide structural explanations for tabular model predictions. 
You MUST respond with only a single raw, valid JSON object that exactly satisfies the required schema. Do not wrap code in markdown formatting like ```json.

Required JSON Structure:
{
  "prediction_label": "string (High Housing Value / Low Housing Value)",
  "confidence_level": "string (low, medium, high)",
  "top_reason": "string (Explanation of the single most influential feature)",
  "second_reason": "string (Explanation of the second most influential feature)",
  "next_step": "string (Actionable advisory step for real estate analysts)"
}"""

USER_PROMPT_TEMPLATE = """Model Prediction Report Summary:
- Input Feature Vector: {feature_dict}
- Predicted Target Classification: {pred_class}
- Predicted Positive Class Probability: {pred_proba:.4f}

Provide your structured explanation JSON following the schema requirements."""

# Define strict target validation schema containing 5 required scalar fields
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string"},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
}

FALLBACK_JSON = {
    "prediction_label": None,
    "confidence_level": None,
    "top_reason": None,
    "second_reason": None,
    "next_step": None
}


# =====================================================================
# STEP 3: SECURITY GUARDRAILS (PII SCANNER)
# =====================================================================
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))

print("\n--- TESTING SECURITY GUARDRAIL MODULE ---")
test_dirty_input = "Contact user at analyst_test@housingcorp.com for property profiles."
test_clean_input = "MedInc: 8.5, HouseAge: 32.0, AveRooms: 6.2"

for idx, inp in enumerate([test_dirty_input, test_clean_input], 1):
    if has_pii(inp):
        print(f"Sample {idx} -> Input blocked: PII detected.")
    else:
        print(f"Sample {idx} -> Clean verification check passed. Proceeding...")


# =====================================================================
# STEP 4: LOAD SAVED ML ARTIFACTS & RECONSTRUCT PREDICTION INPUTS
# =====================================================================
logger.info("Loading best pipeline artifact saved in Part 3...")
try:
    best_pipeline = joblib.load('best_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Missing 'best_model.pkl'. Please run Part 3 compilation steps first!")

# Construct 3 distinctive hand-crafted real-estate vector dictionaries 
# Match columns precisely with raw training features schema layout
handcrafted_inputs = [
    {
        "MedInc": 8.32, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02,
        "Population": 322.0, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23,
        "Account_Segment_Str": 2.0, "Inferred_Numeric_Object": 5.23
    },
    {
        "MedInc": 2.15, "HouseAge": 15.0, "AveRooms": 3.80, "AveBedrms": 1.15,
        "Population": 1400.0, "AveOccup": 3.8, "Latitude": 34.05, "Longitude": -118.24,
        "Account_Segment_Str": 0.0, "Inferred_Numeric_Object": 3.45
    },
    {
        "MedInc": 4.50, "HouseAge": 28.0, "AveRooms": 5.20, "AveBedrms": 1.00,
        "Population": 850.0, "AveOccup": 2.9, "Latitude": 36.77, "Longitude": -119.41,
        "Account_Segment_Str": 1.0, "Inferred_Numeric_Object": 4.62
    }
]


# =====================================================================
# STEP 5: RUN END-TO-END REPRODUCIBLE EXPLANATION PIPELINE
# =====================================================================
print("\n--- PIPELINE EXECUTION: MODEL PREDICTION EXPLANATION SUITE (temp=0.0) ---")

pipeline_history = []

for i, feature_dict in enumerate(handcrafted_inputs, 1):
    # Convert data directly into a matching single-row Pandas DataFrame for the pipeline
    df_row = pd.DataFrame([feature_dict])
    
    # Calculate predictions (Pipeline securely executes its internal scaling + imputer)
    pred_class = int(best_pipeline.predict(df_row)[0])
    pred_proba = float(best_pipeline.predict_proba(df_row)[0][1])
    
    # Build context structures for user prompts
    formatted_user_prompt = USER_PROMPT_TEMPLATE.format(
        feature_dict=feature_dict,
        pred_class="High Value (1)" if pred_class == 1 else "Low Value (0)",
        pred_proba=pred_proba
    )
    
    # Security Interception layer
    if has_pii(formatted_user_prompt):
        print(f"\n[BLOCKED] Input {i} rejected by security agent.")
        continue
        
    # Call the API
    raw_llm_output = call_llm(TRACK_C_SYSTEM_PROMPT, formatted_user_prompt, temperature=0.0)
    
    # Validate and Parse response JSON blocks securely
    parsed_json = None
    validation_status = "Pass"
    
    if raw_llm_output:
        cleaned_text = raw_llm_output.strip()
        try:
            parsed_json = json.loads(cleaned_text)
            jsonschema.validate(instance=parsed_json, schema=EXPLANATION_SCHEMA)
        except json.JSONDecodeError:
            validation_status = "Fail (JSON Parsing Error)"
            parsed_json = FALLBACK_JSON
        except jsonschema.ValidationError as ve:
            validation_status = f"Fail (Schema Validation Error: {ve.message})"
            parsed_json = FALLBACK_JSON
    else:
        validation_status = "Fail (Null Response Received)"
        parsed_json = FALLBACK_JSON
        
    print(f"\n--- RECORD {i} PIPELINE METRICS ---")
    print(f"Input Features: {feature_dict}")
    print(f"Prediction    : Class={pred_class}, Probability={pred_proba:.4f}")
    print(f"Raw Output    : {raw_llm_output}")
    print(f"Validation    : {validation_status}")
    
    pipeline_history.append({
        "input": feature_dict,
        "pred_class": pred_class,
        "pred_proba": pred_proba,
        "output": parsed_json,
        "status": validation_status
    })


# =====================================================================
# STEP 6: TEMPERATURE A/B STABILITY COMPARISON EXPERIMENT
# =====================================================================
print("\n--- STEP 6: RUNNING TEMPERATURE A/B COMPARISON METRICS ---")
# Evaluate stability effects using Record 1 inputs
sample_features = handcrafted_inputs[0]
sample_df = pd.DataFrame([sample_features])
s_class = int(best_pipeline.predict(sample_df)[0])
s_proba = float(best_pipeline.predict_proba(sample_df)[0][1])

comp_user_prompt = USER_PROMPT_TEMPLATE.format(feature_dict=sample_features, pred_class=s_class, pred_proba=s_proba)

out_temp_0 = call_llm(TRACK_C_SYSTEM_PROMPT, comp_user_prompt, temperature=0.0)
out_temp_7 = call_llm(TRACK_C_SYSTEM_PROMPT, comp_user_prompt, temperature=0.7)

print(f"\n[OUTPUT AT TEMP = 0.0]:\n{out_temp_0}")
print(f"\n[OUTPUT AT TEMP = 0.7]:\n{out_temp_7}")
