# /// script
# dependencies = [
#     "pandas",
#     "tqdm",
#     "openai",
#     "requests",
#     "python-dotenv",
# ]
# ///

import pandas as pd
import re
from tqdm import tqdm
import openai
import os
from typing import Tuple, Optional
import time
import requests
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================
load_dotenv()
FANAR_API_KEY = os.getenv("FANAR_API_KEY")
FANAR_BASE_URL = "https://api.fanar.qa/v1"
MODEL_NAME = "Fanar-C-2-27B"
INPUT_FILE = "commonsenseDataset.csv"
DELIMITER = ";"
MAX_RETRIES = 3
RETRY_DELAY = 2

# =========================
# FANAR API SETUP
# =========================
def setup_fanar_api():
    """Initialize Fanar API client"""
    try:
        client = openai.OpenAI(
            base_url=FANAR_BASE_URL,
            api_key=FANAR_API_KEY,
        )
        # Test connection
        test_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "مرحبا"}],
            max_tokens=5,
            temperature=0.0
        )
        print(f"✓ Successfully connected to Fanar API")
        print(f"  Model: {MODEL_NAME}")
        print(f"  Test response: {test_response.choices[0].message.content}")
        return client
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

# =========================
# DIRECT API CALL (fallback)
# =========================
def call_fanar_directly(prompt: str):
    headers = {
        "Authorization": f"Bearer {FANAR_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 10,
        "stop": [" ", "\n", ".", "،"]
    }
    try:
        response = requests.post(f"{FANAR_BASE_URL}/chat/completions",
                                 headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Direct API call failed: {e}")
        return None

# =========================
# PROMPT CREATION
# =========================
def create_prompt(text: str) -> str:
    if "[MASK]" not in text:
        return None
    return f"أكمل الجملة التالية بكلمة واحدة فقط: '{text}'"

# =========================
# PREDICTION FUNCTION
# =========================
def predict_mask(text: str, client=None) -> Tuple[Optional[str], str]:
    if "[MASK]" not in text:
        return None, text
    prompt = create_prompt(text)
    if not prompt:
        return None, text

    for attempt in range(MAX_RETRIES):
        try:
            if client:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                    stop=[" ", "\n", ".", "،"]
                )
                predicted_word = response.choices[0].message.content.strip()
            else:
                predicted_word = call_fanar_directly(prompt)

            if predicted_word:
                predicted_word = re.sub(r'[^\u0600-\u06FF\s]', '', predicted_word).strip()
                if predicted_word:
                    completed = text.replace("[MASK]", predicted_word, 1)
                    return predicted_word, completed

            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return None, text

# =========================
# LOAD DATASET
# =========================
def load_dataset():
    try:
        df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, encoding="utf-8-sig")
        context_col = None
        for col in df.columns:
            if df[col].astype(str).str.contains('\[MASK\]').any():
                context_col = col
                break
        if not context_col:
            context_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        return df, context_col
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        # sample dataset fallback
        data = {
            'id': [1,2,3],
            'context': [
                'الطالب يذاكر [MASK] للامتحان.',
                'أشرب [MASK] في الصباح.',
                'القطة تصطاد [MASK].'
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(INPUT_FILE, sep=DELIMITER, index=False, encoding="utf-8-sig")
        return df, 'context'

# =========================
# MAIN PROCESSING
# =========================
def process_dataset():
    output_file = "fanar_predictions.csv"
    df, context_col = load_dataset()
    client = setup_fanar_api()
    if not client:
        client = None

    predictions = [None] * len(df)
    completed_texts = [None] * len(df)

    for idx in tqdm(range(len(df)), desc="Predicting"):
        text = str(df.iloc[idx][context_col])
        predicted_token, completed_text = predict_mask(text, client)
        predictions[idx] = predicted_token
        completed_texts[idx] = completed_text

        time.sleep(1.5) 

        if (idx+1) % 50 == 0:
            df_temp = df.copy()
            df_temp['predicted_token'] = predictions
            df_temp['completed_text'] = completed_texts
            df_temp.to_csv(output_file, sep=DELIMITER, index=False, encoding="utf-8-sig")

    df['predicted_token'] = predictions
    df['completed_text'] = completed_texts
    df.to_csv(output_file, sep=DELIMITER, index=False, encoding="utf-8-sig")
    print(f"✅ Predictions saved to {output_file}")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    process_dataset()
