import argparse
import importlib.util
import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
NLP_ROOT = ROOT / "VSL_Vietnamese_NLP"
TRANSLATION_ROOT = ROOT / "dich-vi_lo"
ASL_LABELS_PATH = ROOT / "external" / "google-asl-250" / "sign_to_prediction_index_map.json"


ASL_TO_VI_TOKEN = {
    "can": "can",
    "drink": "uong",
    "fine": "khoe",
    "food": "an",
    "go": "di",
    "have": "co",
    "hear": "nghe",
    "hello": "xin_chao",
    "home": "nha",
    "hungry": "doi",
    "milk": "sua",
    "no": "khong",
    "please": "lam_on",
    "police": "canh_sat",
    "potty": "nha_ve_sinh",
    "sick": "benh",
    "sleep": "ngu",
    "sleepy": "buon_ngu",
    "thankyou": "cam_on",
    "thirsty": "khat",
    "water": "nuoc",
    "where": "o_dau",
    "who": "ai",
    "yes": "dong_y",
}


def _prepend_sys_path(path: Path):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_translation_module():
    module_path = TRANSLATION_ROOT / "translate.py"
    spec = importlib.util.spec_from_file_location("vi_lo_translate", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SignLanguageTranslationPipeline:
    def __init__(self, use_bartpho=False, enable_translation=True):
        _prepend_sys_path(NLP_ROOT)

        from src.nlp.sentence_buffer import SentenceBuffer
        from src.nlp.sentence_builder import SentenceBuilder

        self.buffer = SentenceBuffer()
        self.builder = SentenceBuilder(
            rules_path=str(NLP_ROOT / "configs" / "sentence_rules.json"),
            use_bartpho=use_bartpho,
        )
        self.enable_translation = enable_translation
        if enable_translation:
            self.translator = _load_translation_module()
            print("\n" + "="*50)
            print("⏳ Đang tải trước mô hình dịch NLLB, vui lòng đợi...")
            self.translator.load_translation_model()
            print("✅ Đã tải xong mô hình dịch!")
            print("="*50 + "\n")
        else:
            self.translator = None

    def map_sign(self, sign: str) -> str:
        sign = sign.strip()
        return ASL_TO_VI_TOKEN.get(sign, sign.lower().replace(" ", "_"))

    def add_sign(self, sign: str) -> str:
        token = self.map_sign(sign)
        self.buffer.add_word(token)
        return token

    def build(self):
        raw = self.buffer.raw_sentence()
        vietnamese = self.builder.build(raw)
        lao = ""

        if self.enable_translation and vietnamese:
            lao = self.translator.translate_vi_to_lao(vietnamese)

        return {
            "raw": raw,
            "vietnamese": vietnamese,
            "lao": lao,
        }

    def clear(self):
        self.buffer.clear()


def available_asl_labels():
    with open(ASL_LABELS_PATH, "r", encoding="utf-8") as file:
        sign_to_id = json.load(file)
    return sorted(sign_to_id)


def run_cli():
    parser = argparse.ArgumentParser(
        description="Integrate ASL recognition labels with Vietnamese NLP and Vietnamese-Lao translation."
    )
    parser.add_argument(
        "--no-translation",
        action="store_true",
        help="Only build Vietnamese sentence; do not load NLLB/LoRA translator.",
    )
    parser.add_argument(
        "--use-bartpho",
        action="store_true",
        help="Use fine-tuned BARTpho if VSL_Vietnamese_NLP/models/bartpho_sentence exists.",
    )
    parser.add_argument(
        "--list-labels",
        action="store_true",
        help="Print labels supported by external/asl_baseline.",
    )
    args = parser.parse_args()

    if args.list_labels:
        print("\n".join(available_asl_labels()))
        return

    pipeline = SignLanguageTranslationPipeline(
        use_bartpho=args.use_bartpho,
        enable_translation=not args.no_translation,
    )

    print("Integrated Sign Language Pipeline")
    print("Nhập ASL label, ví dụ: hello, thankyou, water")
    print("Lệnh: /build, /clear, /exit")

    while True:
        text = input("sign> ").strip()

        if text == "/exit":
            break
        if text == "/clear":
            pipeline.clear()
            print("Đã xóa buffer.")
            continue
        if text == "/build":
            result = pipeline.build()
            print("Chuỗi token:", result["raw"])
            print("Tiếng Việt:", result["vietnamese"])
            if not args.no_translation:
                print("Tiếng Lào:", result["lao"])
            continue

        token = pipeline.add_sign(text)
        print(f"{text} -> {token}")


if __name__ == "__main__":
    run_cli()

