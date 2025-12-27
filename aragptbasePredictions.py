# =========================
# COMPLETE CORRECTED ARAGPT-BASE PREDICTION CODE
# =========================
import torch
import pandas as pd
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
MODEL_NAME = "aubmindlab/aragpt2-base"  # Changed to base model
INPUT_FILE = "commonsenseDataset.csv"
OUTPUT_FILE = "aragpt2_base_corrected.csv"  # Updated output filename
DELIMITER = ";"

# =========================
# LOAD MODEL (COMPATIBLE WITH TRANSFORMERS 3.5.1)
# =========================
print(f"Loading {MODEL_NAME}...")
print(f"PyTorch version: {torch.__version__}")

try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading: {e}")
    try:
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, from_tf=False)
    except Exception as e2:
        print(f"Alternative loading failed: {e2}")
        exit(1)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# CORRECTED TEXT PREPARATION
# =========================
def prepare_text_for_aragpt2(text: str) -> str:
    """
    CORRECTED: Removes [MASK] and everything after it in the sentence.
    Example: '... يُعرف منذ القدم باسم [MASK].' → '... يُعرف منذ القدم باسم'
    """
    if "[MASK]" not in text:
        return text.strip()
    
    # Method 1: Split at [MASK] and take what's before it
    parts = text.split("[MASK]")
    if parts and parts[0]:
        before_mask = parts[0].strip()
        
        # Remove any trailing punctuation/whitespace
        before_mask = re.sub(r'[\s\.\,\;\:\!\?]+$', '', before_mask)
        
        # Remove trailing Arabic prepositions that make no sense alone
        # e.g., remove "بـ" from "باسم بـ"
        if before_mask.endswith(" بـ") or before_mask.endswith(" ب"):
            before_mask = before_mask[:-2].strip()
        elif before_mask.endswith(" لـ") or before_mask.endswith(" ل"):
            before_mask = before_mask[:-2].strip()
        elif before_mask.endswith(" في") or before_mask.endswith(" على"):
            before_mask = before_mask[:-3].strip()
        
        return before_mask
    
    return text.strip()

# =========================
# ARABIC VALIDATION
# =========================
def is_valid_arabic_word(word: str) -> bool:
    """
    Validate if it's a proper Arabic word (not garbage like 'حولها', 'أن')
    """
    if not word or len(word) < 2:
        return False
    
    # Must contain Arabic letters
    if not re.search(r'[\u0600-\u06FF]', word):
        return False
    
    # Reject common garbage tokens
    garbage = {
        "حولها", "حوله", "أن", "إن", "ما", "هو", "هي", 
        "كان", "يكون", "بأن", "كأن", "لأن", "ولكن", "أو", "و"
    }
    if word in garbage:
        return False
    
    # Check reasonable length for Arabic word
    if len(word) > 25:  # Too long for single word
        return False
    
    # Count Arabic characters
    arabic_chars = sum(1 for c in word if '\u0600' <= c <= '\u06FF')
    if arabic_chars < 2:
        return False
    
    # Check it's not just a suffix
    common_suffixes = ["ات", "ين", "ون", "ان", "ة", "ى"]
    if len(word) <= 4:
        for suffix in common_suffixes:
            if word.endswith(suffix) and len(word[:-len(suffix)]) < 2:
                return False
    
    return True

# =========================
# IMPROVED PREDICTION WITH BEAM SEARCH (ADJUSTED FOR BASE MODEL)
# =========================
def predict_next_word_beam(text: str, num_beams=3, num_return_sequences=3, max_new_tokens=6):
    """
    Use beam search for better quality predictions.
    Adjusted parameters for base model.
    """
    if "[MASK]" not in text:
        return None, text
    
    # CORRECT: Remove [MASK] properly
    context = prepare_text_for_aragpt2(text)
    
    # Debug: Show what's being sent to the model
    print(f"\nDebug: Context for prediction (length: {len(context)})")
    print(f"  '{context[-50:]}...'" if len(context) > 50 else f"  '{context}'")
    
    try:
        # Encode context
        input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
        
        # Generate with beam search
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.9,  # Added temperature for better diversity
                do_sample=False,  # Beam search doesn't use sampling
            )
        
        # Try each generated sequence
        for i in range(min(num_return_sequences, len(outputs))):
            # Get only the NEW tokens (after input)
            new_tokens = outputs[i][input_ids.shape[1]:]
            if len(new_tokens) == 0:
                continue
            
            # Decode new tokens
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean and extract first Arabic word
            generated_text = generated_text.strip()
            
            # Method 1: Split by common separators
            words = re.split(r'[\s\.\,\;\:\!\?\"\'\(\)\[\]\{\}\-]+', generated_text)
            for word in words:
                if word and is_valid_arabic_word(word):
                    # Replace [MASK] with prediction
                    completed = text.replace("[MASK]", word, 1)
                    return word, completed
            
            # Method 2: Extract Arabic words directly
            arabic_words = re.findall(r'[\u0600-\u06FF]{2,}', generated_text)
            for word in arabic_words:
                if is_valid_arabic_word(word):
                    completed = text.replace("[MASK]", word, 1)
                    return word, completed
        
        # If no valid word found
        return None, text
        
    except Exception as e:
        print(f"Beam search error: {e}")
        return None, text

# =========================
# SIMPLE GREEDY PREDICTION (FALLBACK - OPTIMIZED FOR BASE MODEL)
# =========================
def predict_next_word_greedy(text: str):
    """
    Simple greedy prediction as fallback.
    Optimized for base model's vocabulary.
    """
    if "[MASK]" not in text:
        return None, text
    
    context = prepare_text_for_aragpt2(text)
    
    try:
        # Encode and get next token probabilities
        input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[0][:, -1, :]
            
            # Get top 30 predictions for base model (more options needed)
            top_k = torch.topk(next_token_logits, 30)
            token_ids = top_k.indices[0].tolist()
            scores = top_k.values[0].tolist()
            
            # Try each candidate
            best_score = -1000
            best_word = None
            
            for token_id, score in zip(token_ids, scores):
                token_text = tokenizer.decode([token_id])
                token_text = token_text.strip()
                
                # Basic cleaning
                token_text = re.sub(r'[^\u0600-\u06FF]', '', token_text)
                
                if token_text and is_valid_arabic_word(token_text):
                    if score > best_score:
                        best_score = score
                        best_word = token_text
            
            if best_word:
                completed = text.replace("[MASK]", best_word, 1)
                return best_word, completed
            else:
                return None, text
                
    except Exception as e:
        print(f"Greedy prediction error: {e}")
        return None, text

# =========================
# MAIN PREDICTION FUNCTION
# =========================
def predict_with_aragpt2(text: str):
    """
    Main function: tries beam search first, then greedy fallback.
    """
    if "[MASK]" not in text:
        return None, text
    
    # Try beam search first
    predicted, completed = predict_next_word_beam(text)
    
    # If beam search fails or returns garbage, try greedy
    if not predicted or not is_valid_arabic_word(predicted):
        predicted, completed = predict_next_word_greedy(text)
    
    return predicted, completed

# =========================
# TEST WITH YOUR EXAMPLES
# =========================
def test_with_your_dataset():
    """
    Test with examples from your dataset to verify predictions make sense.
    """
    print("\n" + "="*80)
    print("TESTING WITH DATASET EXAMPLES")
    print("="*80)
    
    # Your problematic examples
    test_cases = [
        ("تم اختيار بنيامين بدلاً من بريت ليكون فنان الماكياج للمسرحية. بريت كان أقل [MASK].", "خبرة/مهارة"),
        ("الكوميدي سرد بعض النكات و قدم عرضا مميزا.طيلة السهرة تسمع الجمهور [MASK].", "يضحك"),
        ("رفع الدف بيديه.اراد ان يضيف [MASK].", "إيقاعًا/لمسة/نغمة"),
        ("قام الممثلون بالتحضير للعرض المسرحي لمدة شهر.كان اهل المدينة يحبون ارتياد [MASK].", "المسرح"),
        ("أعضاء الموكب يسيرون في الشارع وهم يحملون آلاتهم. يصفق لهم الجمهور بينما هم يقرعون [MASK].", "الطبول"),
    ]
    
    print(f"\n{'#':<3} {'Expected':<20} {'Predicted':<15} {'Status':<10} {'Context Preview':<40}")
    print("-" * 100)
    
    for i, (sentence, expected) in enumerate(test_cases):
        predicted, completed = predict_with_aragpt2(sentence)
        
        # Status indicator
        if predicted and is_valid_arabic_word(predicted) and predicted not in ["حولها", "أن"]:
            status = "✓ GOOD"
        elif predicted in ["حولها", "أن", "إن", "ما"]:
            status = "✗ GARBAGE"
        elif not predicted:
            status = "✗ NONE"
        else:
            status = "? OK"
        
        # Short context preview
        context_preview = sentence[:40] + "..." if len(sentence) > 40 else sentence
        
        print(f"{i+1:<3} {expected:<20} {str(predicted):<15} {status:<10} {context_preview:<40}")
    
    return True

# =========================
# PROCESS FULL DATASET
# =========================
def process_full_dataset():
    """
    Process the entire CSV file.
    """
    print("\n" + "="*80)
    print("PROCESSING FULL DATASET")
    print("="*80)
    
    # Read CSV
    try:
        df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, encoding="utf-8-sig")
    except:
        df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, encoding="utf-8")
    
    # Find context column (usually column 2)
    if len(df.columns) >= 3:
        context_col = df.columns[2]
    else:
        context_col = df.columns[0]
    
    print(f"Dataset: {INPUT_FILE}")
    print(f"Context column: '{context_col}'")
    print(f"Total rows: {len(df)}")
    
    # Count rows with [MASK]
    mask_count = 0
    for text in df[context_col].astype(str):
        if "[MASK]" in text:
            mask_count += 1
    
    print(f"Rows with [MASK]: {mask_count}")
    
    # Process all rows
    predictions = []
    completed_texts = []
    
    print("\nProcessing rows...")
    for idx, text in enumerate(df[context_col].astype(str)):
        if idx % 50 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df)} rows...")
        
        predicted, completed = predict_with_aragpt2(text)
        predictions.append(predicted)
        completed_texts.append(completed)
    
    # Save results
    df["predicted_token"] = predictions
    df["completed_text"] = completed_texts
    
    df.to_csv(OUTPUT_FILE, sep=DELIMITER, index=False, encoding="utf-8-sig")
    
    # Statistics
    successful = sum(1 for t in predictions if t is not None)
    garbage = sum(1 for t in predictions if t in ["حولها", "أن", "إن", "ما"])
    good = successful - garbage
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total rows: {len(df)}")
    print(f"Rows with [MASK]: {mask_count}")
    print(f"Predictions made: {successful}")
    print(f"Garbage predictions (حولها/أن): {garbage}")
    print(f"Good predictions: {good}")
    print(f"Good prediction rate: {good/len(df)*100:.1f}%")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    
    # Show sample predictions
    print("\nSample of good predictions (first 5):")
    good_samples = [(i, p) for i, p in enumerate(predictions) 
                   if p and p not in ["حولها", "أن", "إن", "ما"]]
    
    for i, (idx, prediction) in enumerate(good_samples[:5]):
        original = df.iloc[idx][context_col]
        preview = str(original)[:60] + "..." if len(str(original)) > 60 else str(original)
        print(f"\nRow {idx}:")
        print(f"  Context: {preview}")
        print(f"  Predicted: '{prediction}'")

# =========================
# VALIDATE MASK REMOVAL
# =========================
def validate_mask_removal():
    """
    Verify that [MASK] is being removed correctly.
    """
    print("\n" + "="*80)
    print("VALIDATING [MASK] REMOVAL")
    print("="*80)
    
    test_sentence = "لإحياء تراث الأجداد في مهرجان التراث، قرروا بناء مأوى بدوي على الطراز الفرعوني الذي كان شائعًا في صحراء مصر العليا. لم يستخدموا الطوب أو الحجر، بل أقاموا بناءً متنقلاً من القماش والأعمدة يُعرف منذ القدم باسم [MASK]."
    
    print(f"Original sentence length: {len(test_sentence)} chars")
    print(f"Contains [MASK]? {'YES' if '[MASK]' in test_sentence else 'NO'}")
    
    # Process with our function
    processed = prepare_text_for_aragpt2(test_sentence)
    
    print(f"\nProcessed sentence length: {len(processed)} chars")
    print(f"Contains [MASK]? {'YES' if '[MASK]' in processed else 'NO'}")
    
    # Show the end of processed text (where [MASK] was)
    print(f"\nLast 30 chars of processed text:")
    print(f"  '...{processed[-30:]}'")
    
    # Show what was removed
    mask_index = test_sentence.find("[MASK]")
    if mask_index > 0:
        print(f"\nText before [MASK]: '...{test_sentence[max(0, mask_index-30):mask_index]}'")
        print(f"Text after [MASK]: '{test_sentence[mask_index:mask_index+50]}...'")
    
    # Final validation
    if "[MASK]" in processed:
        print("\n❌ FAIL: [MASK] still present in processed text!")
        return False
    else:
        print("\n✅ PASS: [MASK] correctly removed")
        return True

# =========================
# BASE MODEL SPECIFIC ADJUSTMENTS
# =========================
def check_model_capabilities():
    """
    Check base model capabilities and adjust parameters accordingly.
    """
    print("\n" + "="*80)
    print("BASE MODEL CONFIGURATION")
    print("="*80)
    
    # Model info
    print(f"Model name: {MODEL_NAME}")
    print(f"Model parameters: ~137M (base) vs ~835M (large)")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Memory requirements
    memory_estimate = model.get_memory_footprint() / 1024**3
    print(f"Estimated memory footprint: {memory_estimate:.2f} GB")
    
    # Generation parameters for base model
    print("\nRecommended parameters for base model:")
    print("  - Smaller context window (512 tokens)")
    print("  - More beam candidates (5-7 beams)")
    print("  - Lower temperature for more focused predictions")
    print("  - More top-k candidates in greedy search")
    
    return True

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("ARAGPT2-BASE PREDICTION PIPELINE")
    print(f"Model: {MODEL_NAME}")
    print(f"Transformers compatibility: 3.5.1")
    
    # Step 0: Check model capabilities
    check_model_capabilities()
    
    # Step 1: Validate mask removal
    if not validate_mask_removal():
        print("\n⚠️  WARNING: [MASK] removal not working. Predictions will fail.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(1)
    
    # Step 2: Test with sample sentences
    test_with_your_dataset()
    
    # Step 3: Process full dataset
    response = input("\nProcess full dataset? This may take a while. (y/n): ")
    if response.lower() == 'y':
        process_full_dataset()
    else:
        print("\nExiting. Run again to process dataset.")