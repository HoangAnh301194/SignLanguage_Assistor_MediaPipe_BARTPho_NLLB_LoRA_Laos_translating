import json
from pathlib import Path

from src.nlp.fallback_corrector import normalize_raw_text, to_sentence


class SentenceBuilder:
    """
    Chuyển chuỗi ký hiệu thô thành câu tiếng Việt.

    Thứ tự xử lý:
    1. Rule/template trong configs/sentence_rules.json
    2. BARTpho đã fine-tune
    3. Fallback corrector nếu chưa có model BARTpho
    """

    def __init__(
        self,
        rules_path: str = None,
        use_rules: bool = True,
        use_bartpho: bool = True
    ):
        if rules_path is None:
            self.rules_path = Path(__file__).resolve().parent.parent.parent / "configs" / "sentence_rules.json"
        else:
            self.rules_path = Path(rules_path)
            
        self.use_rules = use_rules
        self.use_bartpho = use_bartpho
        self.rules = self._load_rules()

        self.bartpho = None

        if self.use_bartpho:
            try:
                from src.nlp.bartpho_predictor import BartphoSentencePredictor

                self.bartpho = BartphoSentencePredictor()
                print("BARTpho sentence model đã sẵn sàng.")
            except FileNotFoundError as exc:
                print(exc)
                print("Tạm thời dùng rule/fallback. Hãy train BARTpho sau.")
            except ModuleNotFoundError as exc:
                print(f"Không thể nạp BARTpho vì thiếu thư viện: {exc}")
                print("Tạm thời dùng rule/fallback.")

    def _load_rules(self):
        if not self.rules_path.exists():
            return {}

        with open(self.rules_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def build(self, raw: str) -> str:
        raw = " ".join(raw.strip().split())

        if not raw:
            return ""

        if self.use_rules and raw in self.rules:
            return self.rules[raw]

        if self.bartpho is not None:
            return self.bartpho.predict(raw)

        normalized = normalize_raw_text(raw)
        return to_sentence(normalized)

