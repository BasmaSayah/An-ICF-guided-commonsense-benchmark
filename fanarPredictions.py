import pandas as pd
import re
from tqdm import tqdm
import openai
import os
from typing import List, Tuple, Optional
import time
import requests
# =========================
# CONFIG - USING FANAR API
# =========================
FANAR_API_KEY = "CQzvy8bjQJh3iVWXQYRPIyasS4GZc1Wq"
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
        print(f"\nTroubleshooting steps:")
        print(f"1. Check API key: {FANAR_API_KEY[:10]}...")
        print(f"2. Check base URL: {FANAR_BASE_URL}")
        print(f"3. Check model name: {MODEL_NAME}")
        return None

# =========================
# SIMPLE DIRECT API CALL (BACKUP METHOD)
# =========================
def call_fanar_directly(prompt: str):
    """Alternative: Call Fanar API directly using requests"""
    headers = {
        "Authorization": f"Bearer {FANAR_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 10,
        "stop": [" ", "\n", ".", "،"]  # Multiple stop tokens
    }
    
    try:
        response = requests.post(
            f"{FANAR_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
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
    """Create prompt for mask prediction"""
    if "[MASK]" not in text:
        return None
    
    # Simple prompt based on working code
    prompt = f"أكمل الجملة التالية بكلمة واحدة فقط: '{text}'"
    return prompt

# =========================
# PREDICTION FUNCTION
# =========================
def predict_mask(text: str, client=None) -> Tuple[Optional[str], str]:
    """Predict mask token in text"""
    if "[MASK]" not in text:
        return None, text
    
    prompt = create_prompt(text)
    if not prompt:
        return None, text
    
    for attempt in range(MAX_RETRIES):
        try:
            # Method 1: Use OpenAI client
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
                # Method 2: Direct API call
                predicted_word = call_fanar_directly(prompt)
            
            if predicted_word:
                # Clean the predicted word
                predicted_word = predicted_word.strip()
                
                # Remove any punctuation
                predicted_word = re.sub(r'[^\u0600-\u06FF\s]', '', predicted_word).strip()
                
                if predicted_word:
                    # Replace the mask with predicted word
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
    """Load and validate dataset"""
    try:
        df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, encoding="utf-8-sig")
        print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show column names
        print(f"Columns: {list(df.columns)}")
        
        # Find context column (looking for column with [MASK])
        context_col = None
        for col in df.columns:
            if df[col].astype(str).str.contains('\[MASK\]').any():
                context_col = col
                break
        
        if not context_col:
            # Use second column if no mask found
            context_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        print(f"Using context column: '{context_col}'")
        
        # Count masks
        mask_count = df[context_col].astype(str).str.contains('\[MASK\]').sum()
        print(f"Rows with [MASK]: {mask_count}")
        
        if mask_count == 0:
            print("⚠️ Warning: No [MASK] tokens found in dataset!")
            print("Please ensure your CSV contains text with [MASK] placeholder")
        
        return df, context_col
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        
        # Create sample dataset for testing
        print("Creating sample dataset for testing...")
        data = {
            'id': [1, 2, 3, 4, 5],
            'context': [
                'الطالب يذاكر [MASK] للامتحان.',
                'السماء لونها [MASK] في النهار.',
                'أشرب [MASK] في الصباح.',
                'القطة تصطاد [MASK].',
                'الكتاب يحتوي على [MASK] كثيرة.'
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(INPUT_FILE, sep=DELIMITER, index=False, encoding="utf-8-sig")
        print(f"Created sample file: {INPUT_FILE}")
        
        return df, 'context'

# =========================
# MAIN PROCESSING
# =========================
def process_dataset():
    """Main processing function"""
    print("\n" + "="*80)
    print("ARABIC MASK PREDICTION SYSTEM")
    print("="*80)
    
    # Set default output file
    output_file = "fanar_predictions.csv"
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    df, context_col = load_dataset()
    

    print("\n[2/4] Setting up API connection...")
    client = setup_fanar_api()
    if not client:
        print("❌ Failed to set up API client!")
        print("Trying direct API calls...")
        client = None  # Will use direct API calls instead
    
    # Check if output file already exists
    start_idx = 0
    if os.path.exists(output_file):
        print(f"\nOutput file '{output_file}' already exists.")
        print("Options:")
        print("1. Overwrite existing file")
        print("2. Resume from last processed row")
        print("3. Create new file with timestamp")
        
        resume_choice = input("\nEnter choice (1-3): ").strip()
        
        if resume_choice == "2":
            try:
                existing_df = pd.read_csv(output_file, delimiter=DELIMITER, encoding="utf-8-sig")
                start_idx = len(existing_df)
                print(f"Resuming from row {start_idx}...")
                
                # If we have existing predictions, load them
                if 'predicted_token' in existing_df.columns and 'completed_text' in existing_df.columns:
                    predictions = existing_df['predicted_token'].tolist()
                    completed_texts = existing_df['completed_text'].tolist()
                    
                    # Extend lists with None for remaining rows
                    remaining_rows = len(df) - start_idx
                    predictions.extend([None] * remaining_rows)
                    completed_texts.extend([None] * remaining_rows)
                    
                else:
                    predictions = [None] * len(df)
                    completed_texts = [None] * len(df)
                    
            except Exception as e:
                print(f"Error loading existing file: {e}")
                print("Starting from beginning...")
                predictions = [None] * len(df)
                completed_texts = [None] * len(df)
                
        elif resume_choice == "3":
            # Create new file with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"fanar_predictions_{timestamp}.csv"
            print(f"Creating new output file: {output_file}")
            predictions = [None] * len(df)
            completed_texts = [None] * len(df)
            
        else:
            # Overwrite
            print("Overwriting existing file...")
            predictions = [None] * len(df)
            completed_texts = [None] * len(df)
    else:
        predictions = [None] * len(df)
        completed_texts = [None] * len(df)
    
    # Process predictions
    print("\n[3/4] Processing predictions...")
    total_rows = len(df)
    print(f"Processing {total_rows} rows...")
    
    # Process only from start_idx onward
    for idx in tqdm(range(start_idx, total_rows), desc="Predicting", initial=start_idx, total=total_rows):
        text = df.iloc[idx][context_col]
        if pd.isna(text):
            predictions[idx] = None
            completed_texts[idx] = None
            continue
        
        predicted_token, completed_text = predict_mask(str(text), client)
        predictions[idx] = predicted_token
        completed_texts[idx] = completed_text
        
        # Show progress for first few
        if idx < 5 and predicted_token:
            print(f"\nSample {idx + 1}:")
            print(f"  Original:  {text}")
            print(f"  Predicted: '{predicted_token}'")
            print(f"  Completed: {completed_text}")
        
        # Save progress every 50 rows
        if (idx + 1) % 50 == 0:
            print(f"\n[Progress] Saving intermediate results after {idx + 1} rows...")
            df_temp = df.copy()
            df_temp['predicted_token'] = predictions
            df_temp['completed_text'] = completed_texts
            df_temp.to_csv(output_file, sep=DELIMITER, index=False, encoding="utf-8-sig")
    
    # Add predictions to dataframe
    df['predicted_token'] = predictions
    df['completed_text'] = completed_texts
    
    # Save final results
    print("\n[4/4] Saving final results...")
    df.to_csv(output_file, sep=DELIMITER, index=False, encoding="utf-8-sig")
    
    # Statistics
    successful = sum(1 for p in predictions if p)
    total = len(predictions)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total processed: {total}")
    print(f"Successful predictions: {successful}")
    print(f"Success rate: {(successful/total*100):.1f}%" if total > 0 else "0%")
    print(f"\nOutput saved to: {output_file}")
    
    # Show first 5 results
    print("\nFirst 5 predictions:")
    for idx in range(min(5, len(df))):
        if pd.notna(df.iloc[idx]['predicted_token']):
            print(f"\nRow {idx + 1}:")
            print(f"  Context: {df.iloc[idx][context_col]}")
            print(f"  Predicted: '{df.iloc[idx]['predicted_token']}'")
            print(f"  Completed: {df.iloc[idx]['completed_text']}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Process the entire dataset
    process_dataset()