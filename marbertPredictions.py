import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# =========================
# CONFIG - USING MARBERTv2
# =========================
# CHANGED: Using MARBERTv2 which has 512 sequence length
MODEL_NAME = "UBC-NLP/MARBERTv2"  # 162M parameters, 512 sequence length
INPUT_FILE = "commonsenseDataset.csv"
OUTPUT_FILE = "marbertv2_predictions.csv"
DELIMITER = ";"
TOP_K = 10

# =========================
# LOAD MODEL
# =========================
print(f"Loading {MODEL_NAME} (162M parameters)...")
print(f"This model supports up to 512 tokens sequence length")

try:
    # MARBERTv2 uses standard tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Check model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# =========================
# MARBERT-SPECIFIC ADJUSTMENTS
# =========================
def adjust_for_marbert(text: str) -> str:
    """
    MARBERT sometimes uses different mask token handling
    """
    # Ensure mask token is properly formatted
    if "[MASK]" in text:
        # MARBERT typically uses [MASK] token
        return text
    elif "<mask>" in text:
        return text.replace("<mask>", "[MASK]")
    return text

# =========================
# VALIDATION (Same as before)
# =========================
ARABIC_MORPHEME_SUFFIXES = {
    "ات", "ان", "ين", "ون", "ة", "ه", "ها", "هم", "هن",
    "ك", "ي", "نا", "كم", "كن", "ا", "و", "ى"
}

def clean_token(token: str) -> str:
    """Clean token but don't validate"""
    if not token:
        return ""
    
    token = str(token).strip()
    
    # Remove special tokens
    token = re.sub(r'\[(UNK|PAD|CLS|SEP|MASK)\]', '', token)
    token = re.sub(r'^\+', '', token)
    token = re.sub(r'^\.', '', token)
    token = token.replace("##", "")
    
    # Remove non-Arabic characters
    token = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s\-\.،؛؟]', '', token)
    
    return token.strip()

def is_valid_arabic(token: str) -> bool:
    """Validate cleaned token"""
    if not token:
        return False
    
    if not re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', token):
        return False
    
    if len(token) < 3:
        return False
    
    arabic_chars = sum(1 for c in token if '\u0600' <= c <= '\u06FF')
    if arabic_chars < 2:
        return False
    
    if token in ARABIC_MORPHEME_SUFFIXES:
        return False
    
    common_suffixes = ["ات", "ان", "ين", "ون", "ة", "ى"]
    if len(token) <= 4:
        for suffix in common_suffixes:
            if token.endswith(suffix) and len(token[:-len(suffix)]) < 2:
                return False
    
    return True

# =========================
# PREDICTION LOGIC FOR MARBERT
# =========================
def predict_with_marbert(text: str, top_k: int = TOP_K):
    """
    Prediction function optimized for MARBERT
    """
    text = adjust_for_marbert(text)
    
    if "[MASK]" not in text:
        return None, text
    
    try:
        # MARBERTv2 supports 512 tokens, so we can use longer sequences
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        
        if len(mask_positions) == 0:
            return None, text
        
        mask_pos = mask_positions[0]
        logits = outputs.logits[0, mask_pos]
        
        # Get top predictions
        top_k_results = torch.topk(logits, top_k)
        top_token_ids = top_k_results.indices.tolist()
        top_scores = top_k_results.values.tolist()
        
        valid_candidates = []
        
        for token_id, score in zip(top_token_ids, top_scores):
            raw_token = tokenizer.decode([token_id])
            cleaned_token = clean_token(raw_token)
            
            if not cleaned_token:
                continue
            
            if is_valid_arabic(cleaned_token):
                valid_candidates.append((cleaned_token, score))
        
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            chosen_token = valid_candidates[0][0]
            
            completed = text.replace("[MASK]", chosen_token, 1)
            return chosen_token, completed
        else:
            return None, text
            
    except Exception as e:
        print(f"Error: {e}")
        return None, text

# =========================
# PROCESS DATASET
# =========================
def process_full_dataset():
    """Process entire dataset with MARBERT"""
    print("\n" + "="*80)
    print(f"PROCESSING DATASET WITH {MODEL_NAME}")
    print("="*80)
    
    print("Reading dataset...")
    df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, encoding="utf-8-sig")
    
    context_col = df.columns[2] if len(df.columns) >= 3 else df.columns[0]
    print(f"Using column: '{context_col}'")
    
    mask_count = df[context_col].astype(str).str.contains('\[MASK\]').sum()
    print(f"Rows with [MASK]: {mask_count}/{len(df)}")
    
    print(f"\nProcessing {len(df)} rows with MARBERTv2...")
    predicted_tokens = []
    
    for idx, text in tqdm(enumerate(df[context_col].astype(str)), total=len(df), desc="Processing"):
        token, completed = predict_with_marbert(text)
        predicted_tokens.append(token)
    
    # Save results
    df["predicted_token"] = predicted_tokens
    df["completed_text"] = df.apply(
        lambda row: str(row[context_col]).replace("[MASK]", row["predicted_token"], 1) 
        if row["predicted_token"] else str(row[context_col]),
        axis=1
    )
    
    df.to_csv(OUTPUT_FILE, sep=DELIMITER, index=False, encoding="utf-8-sig")
    
    # Statistics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    total = len(df)
    successful = sum(1 for t in predicted_tokens if t is not None)
    
    print(f"Model: {MODEL_NAME}")
    print(f"Parameters: ~162M")
    print(f"Total rows: {total}")
    print(f"Rows with [MASK]: {mask_count}")
    print(f"Successful predictions: {successful}")
    
    if mask_count > 0:
        success_rate = successful / mask_count * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    # Show predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    success_indices = [i for i, t in enumerate(predicted_tokens) if t is not None]
    if success_indices:
        print("First 3 successful predictions:")
        for i, idx in enumerate(success_indices[:3]):
            original = df.iloc[idx][context_col]
            predicted = predicted_tokens[idx]
            
            orig_display = str(original)[:80] + "..." if len(str(original)) > 80 else str(original)
            
            print(f"\nRow {idx}:")
            print(f"  Original:  {orig_display}")
            print(f"  Predicted: '{predicted}'")
    
    print(f"\nSaved to {OUTPUT_FILE}")
    print("DONE ✅")

# =========================
# COMPARISON FUNCTION
# =========================
def compare_marbert_versions():
    """Compare different MARBERT versions"""
    print("="*80)
    print("MARBERT VERSION COMPARISON")
    print("="*80)
    
    versions = [
        {"name": "MARBERT", "model": "UBC-NLP/MARBERT", "params": "162M", "seq_len": 128},
        {"name": "MARBERTv1", "model": "UBC-NLP/MARBERTv1", "params": "162M", "seq_len": 512},
        {"name": "MARBERTv2", "model": "UBC-NLP/MARBERTv2", "params": "162M", "seq_len": 512},
    ]
    
    print("\nAvailable MARBERT versions:")
    for v in versions:
        print(f"  • {v['name']}: {v['params']} parameters, {v['seq_len']} token sequence length")
    
    print("\nRecommendation:")
    print("  • Use MARBERTv2 for best overall performance (512 seq length)")
    print("  • Use MARBERT if you have shorter texts and want faster processing")
    print("  • All versions have the same 162M parameter count")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Show comparison
    compare_marbert_versions()
    
    # Process dataset with MARBERTv2
    process_full_dataset()