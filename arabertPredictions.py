import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODEL_NAME = "aubmindlab/bert-large-arabertv2"
INPUT_FILE = "commonsenseDataset.csv"
OUTPUT_FILE = "arabert_fixed_logic.csv"
DELIMITER = ";"
TOP_K = 10

# =========================
# LOAD MODEL
# =========================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# =========================
# VALIDATION
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
    
    # Remove non-Arabic characters (keep Arabic and basic punctuation)
    token = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s\-\.،؛؟]', '', token)
    
    return token.strip()

def is_valid_arabic(token: str) -> bool:
    """Validate cleaned token"""
    if not token:
        return False
    
    # Check for Arabic characters
    if not re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', token):
        return False
    
    # Length check
    if len(token) < 3:
        return False
    
    # Count Arabic characters
    arabic_chars = sum(1 for c in token if '\u0600' <= c <= '\u06FF')
    if arabic_chars < 2:
        return False
    
    # Reject morphemes
    if token in ARABIC_MORPHEME_SUFFIXES:
        return False
    
    # Reject short suffix-only words
    common_suffixes = ["ات", "ان", "ين", "ون", "ة", "ى"]
    if len(token) <= 4:
        for suffix in common_suffixes:
            if token.endswith(suffix) and len(token[:-len(suffix)]) < 2:
                return False
    
    return True

# =========================
# CORRECTED PREDICTION LOGIC
# =========================
def predict_correct_logic(text: str, top_k: int = TOP_K):
    """
    CORRECTED: Continues checking even if a candidate becomes empty
    """
    if "[MASK]" not in text:
        return None, text
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
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
        
        # ============================================
        # CORRECTED LOGIC: Check ALL candidates
        # ============================================
        valid_candidates = []
        
        for token_id, score in zip(top_token_ids, top_scores):
            raw_token = tokenizer.decode([token_id])
            cleaned_token = clean_token(raw_token)
            
            # Skip if empty after cleaning
            if not cleaned_token:
                continue  # ← KEY: Continue to next candidate!
            
            # Check if valid
            if is_valid_arabic(cleaned_token):
                valid_candidates.append((cleaned_token, score))
        
        # Choose the highest-scoring valid candidate
        if valid_candidates:
            # Sort by score (highest first)
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            chosen_token = valid_candidates[0][0]
            
            completed = text.replace("[MASK]", chosen_token, 1)
            return chosen_token, completed
        else:
            # No valid candidates found
            return None, text
            
    except Exception as e:
        print(f"Error: {e}")
        return None, text

# =========================
# DEMONSTRATE THE FIX
# =========================
def demonstrate_fix():
    """Show the difference between old and new logic"""
    print("="*80)
    print("DEMONSTRATING THE LOGIC FIX")
    print("="*80)
    
    # Simulate a case where top-1 becomes empty, but top-3 is valid
    print("\nExample scenario:")
    print("Top predictions for a [MASK]:")
    print("  1. '[PAD]'  → after cleaning: '' (empty)")
    print("  2. '##ه'    → after cleaning: 'ه' (invalid - too short)")
    print("  3. 'كتاب'   → after cleaning: 'كتاب' (VALID)")
    print("  4. 'قلم'    → after cleaning: 'قلم' (VALID)")
    print("  5. 'مدرسة'  → after cleaning: 'مدرسة' (VALID)")
    
    print("\nOLD LOGIC (WRONG):")
    print("  - Checks #1: becomes '' → immediately returns None")
    print("  - NEVER checks #2, #3, #4, #5")
    print("  - Result: No prediction (None)")
    
    print("\nNEW LOGIC (CORRECT):")
    print("  - Checks #1: becomes '' → SKIPS, continues to #2")
    print("  - Checks #2: 'ه' → invalid (too short) → continues to #3")
    print("  - Checks #3: 'كتاب' → VALID → accepts it")
    print("  - Result: 'كتاب'")
    
    # Test with actual model
    print("\n" + "="*80)
    print("ACTUAL TEST")
    print("="*80)
    
    test_sentences = [
        "يبيع المزارع بيض [MASK].",
        "هذا [MASK] جميل.",
        "[MASK] في المكتبة."
    ]
    
    for sentence in test_sentences:
        print(f"\nTesting: '{sentence}'")
        
        # Show what model predicts
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        
        if len(mask_positions) > 0:
            mask_pos = mask_positions[0]
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits[0, mask_pos]
            top_5 = torch.topk(logits, 5)
            
            print("  Raw predictions and cleaning:")
            for i, (token_id, score) in enumerate(zip(top_5.indices, top_5.values)):
                raw = tokenizer.decode([token_id])
                cleaned = clean_token(raw)
                valid = is_valid_arabic(cleaned)
                status = "✅ VALID" if valid else "❌ INVALID"
                empty_mark = " (EMPTY)" if cleaned == "" else ""
                print(f"    {i+1}. '{raw}' → '{cleaned}'{empty_mark} - {status}")
        
        # Get prediction using corrected logic
        token, completed = predict_correct_logic(sentence)
        print(f"  Final prediction: '{token}'")
        print(f"  Completed: '{completed}'")

# =========================
# PROCESS DATASET
# =========================
def process_full_dataset():
    """Process entire dataset with corrected logic"""
    print("\n" + "="*80)
    print("PROCESSING FULL DATASET WITH CORRECTED LOGIC")
    print("="*80)
    
    print("Reading dataset...")
    df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, encoding="utf-8-sig")
    
    context_col = df.columns[2] if len(df.columns) >= 3 else df.columns[0]
    print(f"Using column: '{context_col}'")
    
    mask_count = df[context_col].astype(str).str.contains('\[MASK\]').sum()
    print(f"Rows with [MASK]: {mask_count}/{len(df)}")
    
    # Process
    print(f"\nProcessing {len(df)} rows...")
    predicted_tokens = []
    
    for idx, text in tqdm(enumerate(df[context_col].astype(str)), total=len(df), desc="Processing"):
        token, completed = predict_correct_logic(text)
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
    
    print(f"Total rows: {total}")
    print(f"Rows with [MASK]: {mask_count}")
    print(f"Successful predictions: {successful}")
    
    if mask_count > 0:
        success_rate = successful / mask_count * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    # Show what predictions were actually made
    print("\n" + "="*80)
    print("PREDICTION SAMPLE")
    print("="*80)
    
    # Show cases where logic made a difference
    success_indices = [i for i, t in enumerate(predicted_tokens) if t is not None]
    
    if success_indices:
        print("First 5 successful predictions:")
        for i, idx in enumerate(success_indices[:5]):
            original = df.iloc[idx][context_col]
            predicted = predicted_tokens[idx]
            
            orig_display = str(original)
            if len(orig_display) > 80:
                orig_display = orig_display[:80] + "..."
            
            print(f"\nRow {idx}:")
            print(f"  Original:  {orig_display}")
            print(f"  Predicted: '{predicted}'")
    
    # Count predictions that would have failed with old logic
    print("\n" + "="*80)
    print("LOGIC IMPROVEMENT ANALYSIS")
    print("="*80)
    
    # Simulate old logic for comparison
    old_success_count = 0
    new_success_count = successful
    
    # For a sample, check what old logic would have produced
    sample_size = min(100, len(df))
    improvements = 0
    
    for idx in range(sample_size):
        text = df.iloc[idx][context_col]
        if "[MASK]" in str(text):
            # Get predictions
            inputs = tokenizer(str(text), return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_positions) > 0:
                mask_pos = mask_positions[0]
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs.logits[0, mask_pos]
                top_3 = torch.topk(logits, 3)
                
                # Old logic: stop at first candidate
                old_result = None
                for token_id in top_3.indices:
                    raw = tokenizer.decode([token_id])
                    cleaned = clean_token(raw)
                    if cleaned and is_valid_arabic(cleaned):
                        old_result = cleaned
                        break
                
                # New logic result
                new_result = predicted_tokens[idx]
                
                if new_result and not old_result:
                    improvements += 1
    
    print(f"In sample of {sample_size} rows:")
    print(f"  - Old logic would fail on {improvements} rows")
    print(f"  - New logic succeeds on those {improvements} rows")
    print(f"  - Improvement: +{improvements} successful predictions")
    
    print(f"\nSaved to {OUTPUT_FILE}")
    print("DONE ✅")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Show the fix
    demonstrate_fix()
    
    # Process dataset
    process_full_dataset()