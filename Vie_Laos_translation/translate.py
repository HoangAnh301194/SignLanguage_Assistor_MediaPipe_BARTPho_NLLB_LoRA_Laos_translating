import io
import os
import sys


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
if hasattr(sys.stdin, "buffer"):
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")


base_model_name = "facebook/nllb-200-distilled-600M"
lora_path = os.path.join(
    os.path.dirname(__file__),
    "model",
    "vi_to_lao_nllb_lora",
    "content",
    "drive",
    "MyDrive",
    "vi_to_lao_nllb_lora",
)

tokenizer = None
model = None
device = None


def _get_device():
    """Auto-detect GPU or fallback to CPU"""
    global device
    if device is not None:
        return device
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    except:
        device = "cpu"
    
    return device


def load_translation_model():
    global tokenizer, model

    if tokenizer is not None and model is not None:
        print("[Model] Using cached model and tokenizer")
        return tokenizer, model

    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import traceback

    try:
        # Verify LoRA path exists
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA model path not found: {lora_path}")
        
        print(f"[Model] LoRA path: {lora_path}")
        print(f"[Model] Path exists: {os.path.exists(lora_path)}")
        
        device = _get_device()
        
        print("[Model] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        print(f"[Model] ✅ Tokenizer loaded. Vocab size: {len(tokenizer)}")

        print("[Model] Loading base model...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, device_map=device)
        print(f"[Model] ✅ Base model loaded")

        print("[Model] Loading LoRA weights...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        print(f"[Model] ✅ LoRA weights loaded")
        
        model = model.to(device)
        model.eval()
        print(f"[Model] ✅ Model set to eval mode on {device}")

        print("=" * 60)
        print("✅ ALL MODELS LOADED SUCCESSFULLY")
        print("=" * 60)
        
        return tokenizer, model
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ CRITICAL MODEL LOADING ERROR: {type(e).__name__}")
        print("=" * 60)
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 60)
        raise

def translate_vi_to_lao(text):
    """Translate Vietnamese text to Lao using LoRA-finetuned NLLB model"""
    import traceback
    
    if not text or not text.strip():
        print("[Translation] ⚠️ Empty input text")
        return ""
    
    try:
        print(f"[Translation] Input text: '{text}'")
        
        print("[Translation] Loading models...")
        tokenizer, model = load_translation_model()
        device = _get_device()
        print(f"[Translation] ✅ Models loaded, device: {device}")
        
        print("[Translation] Setting source language: vie_Latn")
        tokenizer.src_lang = "vie_Latn"

        print("[Translation] Tokenizing input...")
        encoded = tokenizer(
            text.strip(),
            return_tensors="pt"
        )
        print(f"[Translation] ✅ Input tokenized, shape: {encoded['input_ids'].shape}")
        
        # Move inputs to device
        print(f"[Translation] Moving tensors to {device}...")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        print("[Translation] ✅ Tensors moved")

        # Get target language token ID
        print("[Translation] Getting Lao language token ID...")
        try:
            lao_token_id = tokenizer.convert_tokens_to_ids("lao_Laoo")
            print(f"[Translation] Lao token ID: {lao_token_id}")
            if isinstance(lao_token_id, list):
                lao_token_id = lao_token_id[0]
        except Exception as e:
            print(f"[Translation] ⚠️ Could not get Lao token via convert_tokens_to_ids: {e}")
            # Fallback method
            if hasattr(tokenizer, 'lang_code_to_id'):
                lao_token_id = tokenizer.lang_code_to_id.get("lao_Laoo", None)
                print(f"[Translation] Fallback Lao token ID: {lao_token_id}")
            else:
                lao_token_id = tokenizer.eos_token_id
                print(f"[Translation] Using EOS token ID: {lao_token_id}")

        print("[Translation] Starting generation...")
        print("[Translation] ⏳ This may take a moment...")
        
        import torch
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=lao_token_id if lao_token_id else tokenizer.eos_token_id,
                max_length=128,
                num_beams=5
            )

        print(f"[Translation] ✅ Generation complete, shape: {generated_tokens.shape}")
        
        print("[Translation] Decoding output...")
        result = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        print(f"[Translation] Decode result: {result}")
        translation = result[0] if result else ""
        
        print("=" * 60)
        print(f"[Translation] ✅ TRANSLATION SUCCESS")
        print(f"[Translation] Output: '{translation}'")
        print("=" * 60)
        
        return translation
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TRANSLATION ERROR: {type(e).__name__}")
        print("=" * 60)
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 60)
        return ""


def run_cli():
    print("=" * 60)
    print("Vietnamese → Lao Translation (with LoRA)")
    print("=" * 60)
    
    while True:
        try:
            text = input("\nNhập tiếng Việt (hoặc 'exit' để thoát): ").strip()

            if text.lower() == "exit":
                print("Tạm biệt!")
                break
            
            if not text:
                continue

            print("Đang dịch...", end=" ", flush=True)
            result = translate_vi_to_lao(text)
            print("✅")
            print(f"Tiếng Lào: {result}")
        except KeyboardInterrupt:
            print("\n\nBị hủy bởi người dùng.")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    run_cli()

