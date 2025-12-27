# =========================
# JAIS OPTIMIZED SINGLE-WORD PREDICTION PIPELINE (GPU-OPTIMIZED)
# =========================
import torch
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIGURATION
# =========================
MODEL_NAME = "./models/jais-13b-chat"  # Keep the 13B model - your GPU can handle it now!
INPUT_FILE = "commonsenseDataset.csv"
OUTPUT_FILE = "jais_single_word_predictions.csv"
DELIMITER = ";"

# =========================
# STOPPING CRITERIA FOR SINGLE WORDS
# =========================
class SingleWordStoppingCriteria(StoppingCriteria):
    """
    Stop generation at the first whitespace, punctuation, or newline.
    Ensures we get exactly one word.
    """
    def __init__(self, tokenizer, max_tokens=3):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.generated_tokens = 0
        self.whitespace_tokens = [
            tokenizer.encode(" ", add_special_tokens=False)[0],
            tokenizer.encode("\n", add_special_tokens=False)[0],
            tokenizer.encode("\t", add_special_tokens=False)[0],
        ]
        
        # Arabic punctuation marks
        self.punctuation_marks = ["،", ".", "؛", ":", "!", "؟", "(", ")", "[", "]", "{" "}"]
        self.punctuation_tokens = []
        for punc in self.punctuation_marks:
            tokens = tokenizer.encode(punc, add_special_tokens=False)
            if tokens:
                self.punctuation_tokens.extend(tokens)
    
    def __call__(self, input_ids, scores, **kwargs):
        self.generated_tokens += 1
        
        # Stop if we've generated enough tokens for a word
        if self.generated_tokens >= self.max_tokens:
            return True
        
        # Check if last token is whitespace or punctuation
        last_token = input_ids[0][-1].item()
        if last_token in self.whitespace_tokens or last_token in self.punctuation_tokens:
            return True
        
        # Also check if the decoded text contains whitespace
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if re.search(r'\s', current_text[-10:]):  # Check last 10 chars
            return True
            
        return False

# =========================
# LOAD MODEL WITH 4-BIT QUANTIZATION (GPU OPTIMIZED)
# =========================
print(f"Loading JAIS model: {MODEL_NAME}")
print(f"PyTorch version: {torch.__version__}")

# Enable TF32 for faster matrix operations (RTX 4060 supports this)
torch.backends.cuda.matmul.allow_tf32 = True

# Check if CUDA is available
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ CUDA is available! Device: {device_name}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ CUDA is not available. Falling back to CPU (slow!).")

try:
    # 4-bit quantization configuration for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,      # Double quantization for better memory efficiency
        bnb_4bit_quant_type="nf4",           # Normal Float 4 quantization
        bnb_4bit_compute_dtype=torch.float16 # Compute in float16 for speed
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True
    )
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,      # This is the key change for GPU compatibility
        device_map="auto",                   # Automatically distribute layers across GPU/CPU
        trust_remote_code=True,
        offload_folder='offload'            # Folder for offloading if needed
    )
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying fallback loading without quantization...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            offload_folder='offload'
        )
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        print("\nPlease ensure you have installed:")
        print("1. CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("2. Required packages: pip install accelerate bitsandbytes")
        exit(1)

print(f"✅ Model loaded successfully!")
print(f"   Model device: {model.device}")
print(f"   Model dtype: {model.dtype}")

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# OPTIMIZED TEXT PREPARATION
# =========================
def prepare_optimized_context(text: str, max_chars_before_mask: int = 300) -> str:
    """
    Extract context before [MASK] and optimize for JAIS.
    Truncates long contexts to focus on relevant part.
    """
    if "[MASK]" not in text:
        return text
    
    # Extract everything before [MASK]
    parts = text.split("[MASK]")
    context_before = parts[0].strip()
    
    # Truncate if too long (focus on most relevant context)
    if len(context_before) > max_chars_before_mask:
        # Keep the end of context (most relevant part)
        context_before = "..." + context_before[-max_chars_before_mask:]
    
    # Clean trailing punctuation that might confuse the model
    context_before = re.sub(r'[\s\.\,\;\:\!\?]+$', '', context_before)
    
    # Remove trailing prepositions
    trailing_patterns = [
        r'\s+بـ$', r'\s+ب$', r'\s+لـ$', r'\s+ل$',
        r'\s+في$', r'\s+على$', r'\s+من$', r'\s+إلى$'
    ]
    for pattern in trailing_patterns:
        context_before = re.sub(pattern, '', context_before)
    
    return context_before

def create_single_word_prompt(text: str) -> str:
    """
    Create optimized prompt for single-word prediction.
    Uses truncated context for better focus.
    """
    context = prepare_optimized_context(text)
    
    # Ultra-short, focused prompt
    prompt = f"""<s>[INST] <<SYS>>
أكمل الجملة بكلمة واحدة فقط.
لا تكرر السؤال ولا تقدم أي شرح.
<</SYS>>

{context} [MASK].

الكلمة المناسبة هي: [/INST]"""
    
    return prompt

# =========================
# SINGLE-WORD GENERATION (GPU OPTIMIZED)
# =========================
def predict_single_word(text: str):
    """
    Predict exactly one word for the [MASK] position.
    Stops at first whitespace or punctuation.
    """
    if "[MASK]" not in text:
        return None, text, ""
    
    try:
        # Create optimized prompt
        prompt = create_single_word_prompt(text)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=400  # Conservative limit
        ).to(model.device)
        
        # Create stopping criteria
        stopping_criteria = StoppingCriteriaList([
            SingleWordStoppingCriteria(tokenizer, max_tokens=4)
        ])
        
        # Generate with STRICT constraints
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,          # Very short - just enough for a word
                temperature=0.01,          # Almost deterministic
                top_p=0.95,                # Focus on high-probability tokens
                do_sample=False,           # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                stopping_criteria=stopping_criteria,
                num_beams=1,               # Greedy search for speed
                no_repeat_ngram_size=2
            )
        
        # Extract only the generated part (after prompt)
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][prompt_length:]
        
        if len(generated_ids) == 0:
            return None, text, ""
        
        # Decode generated tokens
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract the FIRST word only (stop at any whitespace/punctuation)
        generated_text = generated_text.strip()
        
        # Use regex to get exactly one Arabic word
        arabic_words = re.findall(r'([\u0600-\u06FF]+)', generated_text)
        
        if arabic_words:
            predicted_word = arabic_words[0]  # Take first complete Arabic word
        else:
            # Fallback: split by any non-Arabic character
            words = re.split(r'[^\u0600-\u06FF]+', generated_text)
            predicted_word = words[0] if words else ""
        
        # Validate the word
        if predicted_word and is_valid_single_word(predicted_word):
            completed = text.replace("[MASK]", predicted_word, 1)
            return predicted_word, completed, generated_text
        else:
            return None, text, generated_text
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, text, ""

# =========================
# STRICT VALIDATION FOR SINGLE WORDS
# =========================
def is_valid_single_word(word: str) -> bool:
    """
    Strict validation for single-word predictions.
    """
    if not word or len(word) < 2:
        return False
    
    # Must be entirely Arabic characters
    if not re.fullmatch(r'[\u0600-\u06FF]+', word):
        return False
    
    # Common garbage to reject
    garbage_words = {
        "أن", "إن", "ما", "هو", "هي", "كان", "يكون", 
        "بأن", "كأن", "لأن", "ولكن", "أو", "و",
        "في", "على", "من", "إلى", "عن", "مع",
        "هذا", "هذه", "ذلك", "تلك", "هناك",
        "قد", "سوف", "س", "لا", "لم", "لن", "ليس"
    }
    
    if word in garbage_words:
        return False
    
    # Check for reasonable length
    if len(word) < 2 or len(word) > 15:
        return False
    
    # Check it's not just repeated characters
    if len(set(word)) == 1:
        return False
    
    # Check for common invalid patterns
    invalid_patterns = [
        r'^وو+$',  # Repeated waw
        r'^فف+$',  # Repeated fa
        r'^بب+$',  # Repeated ba
        r'^لل+$',  # Repeated lam
    ]
    
    for pattern in invalid_patterns:
        if re.match(pattern, word):
            return False
    
    return True

# =========================
# TEST WITH IMPROVED VISUALIZATION
# =========================
def test_single_word_predictions():
    """
    Test single-word prediction with detailed analysis
    """
    print("\n" + "="*80)
    print("TESTING SINGLE-WORD PREDICTIONS")
    print("="*80)
    
    test_cases = [
        ("تم اختيار بنيامين بدلاً من بريت ليكون فنان الماكياج للمسرحية. بريت كان أقل [MASK].", "خبرة"),
        ("الكوميدي سرد بعض النكات و قدم عرضا مميزا.طيلة السهرة تسمع الجمهور [MASK].", "يضحك"),
        ("رفع الدف بيديه.اراد ان يضيف [MASK].", "إيقاع"),
        ("قام الممثلون بالتحضير للعرض المسرحي لمدة شهر.كان اهل المدينة يحبون ارتياد [MASK].", "المسرح"),
        ("أعضاء الموكب يسيرون في الشارع وهم يحملون آلاتهم. يصفق لهم الجمهور بينما هم يقرعون [MASK].", "الطبول"),
        ("يجب أن يكون الطالب أكثر [MASK] في دراسته.", "جدية"),
        ("الشجرة تحمل الكثير من [MASK].", "الثمار"),
    ]
    
    print(f"\n{'#':<2} {'Expected':<15} {'Predicted':<15} {'Length':<8} {'Valid':<8} {'Raw Output'}")
    print("-" * 90)
    
    for i, (sentence, expected) in enumerate(test_cases):
        predicted, completed, raw_output = predict_single_word(sentence)
        
        length = len(predicted) if predicted else 0
        is_valid = is_valid_single_word(predicted) if predicted else False
        valid_str = "✓" if is_valid else "✗"
        
        # Clean raw output for display
        clean_raw = raw_output.replace('\n', '\\n').replace('\t', '\\t')
        clean_raw = clean_raw[:30] + "..." if len(clean_raw) > 30 else clean_raw
        
        print(f"{i+1:<2} {expected:<15} {str(predicted):<15} {length:<8} {valid_str:<8} '{clean_raw}'")
    
    return True

# =========================
# BATCH PROCESSING WITH STATISTICS
# =========================
def process_dataset_single_word():
    """
    Process entire dataset with single-word constraint
    """
    print("\n" + "="*80)
    print("PROCESSING DATASET - SINGLE WORD ONLY")
    print("="*80)
    
    # Read dataset
    try:
        df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, encoding="utf-8-sig")
    except:
        df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER, encoding="utf-8")
    
    # Identify context column
    if len(df.columns) >= 3:
        context_col = df.columns[2]
    else:
        context_col = df.columns[0]
    
    print(f"Dataset: {INPUT_FILE}")
    print(f"Rows: {len(df)}")
    
    # Initialize results
    results = []
    
    print("\nProcessing...")
    for idx, row in df.iterrows():
        text = str(row[context_col])
        
        if idx % 5 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df)} rows")
        
        if "[MASK]" in text:
            predicted, completed, raw_output = predict_single_word(text)
            is_valid = is_valid_single_word(predicted) if predicted else False
            
            results.append({
                'original_text': text,
                'predicted_word': predicted,
                'completed_text': completed,
                'raw_output': raw_output,
                'is_valid': is_valid,
                'word_length': len(predicted) if predicted else 0
            })
        else:
            results.append({
                'original_text': text,
                'predicted_word': None,
                'completed_text': text,
                'raw_output': "",
                'is_valid': False,
                'word_length': 0
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Add to original DataFrame
    for col in ['predicted_word', 'completed_text', 'is_valid', 'word_length']:
        df[col] = results_df[col]
    
    # Save results
    df.to_csv(OUTPUT_FILE, sep=DELIMITER, index=False, encoding="utf-8-sig")
    
    # Statistics
    total_mask = sum(1 for r in results if "[MASK]" in r['original_text'])
    valid_predictions = sum(1 for r in results if r['is_valid'])
    avg_length = results_df[results_df['is_valid']]['word_length'].mean()
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total rows: {len(df)}")
    print(f"Rows with [MASK]: {total_mask}")
    print(f"Valid single-word predictions: {valid_predictions}")
    print(f"Success rate: {valid_predictions/total_mask*100:.1f}%" if total_mask > 0 else "N/A")
    print(f"Average word length: {avg_length:.1f}" if not pd.isna(avg_length) else "N/A")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    
    # Show examples
    print("\nTOP PREDICTIONS:")
    valid_results = [r for r in results if r['is_valid']]
    
    for i, result in enumerate(valid_results[:5]):
        text_preview = result['original_text'][:50] + "..." if len(result['original_text']) > 50 else result['original_text']
        print(f"\n{i+1}. '{result['predicted_word']}' ({result['word_length']} chars)")
        print(f"   Context: {text_preview}")
    
    return df

# =========================
# COMPARISON ANALYSIS
# =========================
def analyze_prediction_quality():
    """
    Analyze the quality of single-word predictions
    """
    print("\n" + "="*80)
    print("PREDICTION QUALITY ANALYSIS")
    print("="*80)
    
    print("\n✅ OPTIMIZATIONS IMPLEMENTED:")
    print("1. Single-word stopping criteria")
    print("   - Stops at first whitespace")
    print("   - Stops at punctuation")
    print("   - Maximum 3-4 tokens")
    
    print("\n2. Context optimization")
    print("   - Truncates long contexts")
    print("   - Focuses on relevant part before [MASK]")
    
    print("\n3. Generation constraints")
    print("   - max_new_tokens=4")
    print("   - temperature=0.01 (almost deterministic)")
    print("   - Greedy decoding (no sampling)")
    
    print("\n4. GPU OPTIMIZATIONS:")
    print("   - 4-bit quantization for 13B model")
    print("   - TF32 enabled for faster matrix operations")
    print("   - Auto device mapping")
    
    print("\n⚠️ POTENTIAL LIMITATIONS:")
    print("- May miss multi-word answers (e.g., 'رجل عجوز')")
    print("- Requires clear context before [MASK]")
    print("- Arabic morphology can create ambiguity")

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("=" * 80)
    print("JAIS SINGLE-WORD PREDICTION PIPELINE (GPU OPTIMIZED)")
    print("=" * 80)
    
    # Step 1: Test predictions
    test_single_word_predictions()
    
    # Step 2: Show optimizations
    analyze_prediction_quality()
    
    # Step 3: Process dataset
    response = input("\nProcess full dataset with single-word constraint? (y/n): ")
    
    if response.lower() == 'y':
        print("\nStarting single-word prediction pipeline...")
        print("This will ensure ALL predictions are single words only.")
        
        df = process_dataset_single_word()
        
        # Additional statistics
        print("\n" + "=" * 80)
        print("WORD LENGTH DISTRIBUTION")
        print("=" * 80)
        
        valid_words = df[df['is_valid']]['predicted_word'].dropna()
        if not valid_words.empty:
            from collections import Counter
            
            lengths = valid_words.str.len()
            print(f"Min length: {lengths.min()}")
            print(f"Max length: {lengths.max()}")
            print(f"Mean length: {lengths.mean():.1f}")
            print(f"Median length: {lengths.median()}")
            
            print("\nMost common word lengths:")
            length_counts = Counter(lengths)
            for length, count in length_counts.most_common(5):
                print(f"  {length} chars: {count} words")
    
    print("\n✅ Pipeline completed!")