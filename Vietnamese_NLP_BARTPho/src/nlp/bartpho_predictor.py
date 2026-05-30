from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PREFIX = "gloss: "


class BARTphoPredictor:
    """
    Dùng model BARTpho đã fine-tune để chuyển chuỗi gloss
    thành câu tiếng Việt hoàn chỉnh.

    Ví dụ:
        toi muon uong_nuoc -> Tôi muốn uống nước.
    """

    def __init__(
        self,
        model_dir: str = None,
        max_source_length: int = 96,
        max_target_length: int = 128,
        num_beams: int = 4,
        device: str | None = None,
    ):
        if model_dir is None:
            # Default to VSL_Vietnamese_NLP/models/bartpho_sentence
            self.model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "bartpho_sentence"
        else:
            self.model_dir = Path(model_dir)

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams

        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            raise FileNotFoundError(
                f"Không tìm thấy model BARTpho tại: {self.model_dir}\n"
                "Hãy train model trước hoặc giải nén model vào thư mục models/bartpho_sentence."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            use_fast=False,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_dir),
        )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def normalize_gloss_text(text: str) -> str:
        text = str(text).strip()
        text = " ".join(text.split())
        return text

    @staticmethod
    def normalize_output_sentence(text: str) -> str:
        text = str(text).strip()
        text = " ".join(text.split())

        if text and text[-1] not in [".", "?", "!"]:
            text += "."

        return text

    def predict(self, gloss_sequence: str) -> str:
        gloss_sequence = self.normalize_gloss_text(gloss_sequence)

        if not gloss_sequence:
            return ""

        source_text = PREFIX + gloss_sequence

        inputs = self.tokenizer(
            source_text,
            return_tensors="pt",
            max_length=self.max_source_length,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self.max_target_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )

        sentence = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return self.normalize_output_sentence(sentence)

    def __call__(self, gloss_sequence: str) -> str:
        return self.predict(gloss_sequence)


# Giữ alias cũ để các file khác nếu đang import BartphoSentencePredictor vẫn không lỗi.
BartphoSentencePredictor = BARTphoPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Dự đoán câu tiếng Việt từ chuỗi gloss bằng BARTpho."
    )

    parser.add_argument(
        "--model-dir",
        default="models/bartpho_sentence",
        help="Thư mục chứa model BARTpho đã fine-tune.",
    )

    parser.add_argument(
        "--text",
        required=True,
        help='Chuỗi gloss, ví dụ: "toi muon uong_nuoc"',
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Số beam search khi sinh câu.",
    )

    args = parser.parse_args()

    predictor = BARTphoPredictor(
        model_dir=args.model_dir,
        num_beams=args.num_beams,
    )

    result = predictor.predict(args.text)

    print("Gloss:", args.text)
    print("Sentence:", result)


if __name__ == "__main__":
    main()
