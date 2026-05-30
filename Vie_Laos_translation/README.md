# Vietnamese → Lao Translation Model

This project provides a Vietnamese → Lao translation model fine-tuned with LoRA based on `facebook/nllb-200-distilled-600M`.

Developed by: **Phạm Tuấn Anh**  
GitHub: **Phamanh12345**

---

## Project Structure

```text
dich-vi_lo/
│
├── model/              # LoRA adapter and tokenizer files
├── dataset/            # Processed Vietnamese-Lao dataset
├── translate.py        # Script for testing translation
├── requirements.txt    # Required Python libraries
└── README.md           # Project documentation
```

---

## Model Details

- Task: Vietnamese → Lao machine translation
- Base model: `facebook/nllb-200-distilled-600M`
- Fine-tuning method: LoRA
- Framework: PyTorch, Transformers, PEFT

Note: This repository stores a LoRA adapter. The base model will be downloaded automatically when running the script.

---

## Installation

```bash
git clone https://github.com/Phamanh12345/dich-vi_lo.git
cd dich-vi_lo
pip install -r requirements.txt
```

---

## Model Folder Explanation

The `model/` folder contains the LoRA adapter and tokenizer files used for Vietnamese → Lao translation.

```text
model/
│
├── adapter_config.json            Configuration file for the LoRA adapter. It tells PEFT how the adapter was trained and which base model it uses.
├── adapter_model.safetensors      The trained LoRA weights. This is the most important model file in the folder.
├── added_tokens.json              Stores extra special tokens added to the tokenizer, if any.
├── sentencepiece.bpe.model        SentencePiece/BPE tokenizer model used to split text into tokens.
├── tokenizer_config.json          Configuration file for the tokenizer, including language and tokenization settings.
└── vocab.json                     Vocabulary file that maps tokens to token IDs.
```

---

## Usage

```bash
python translate.py
```

Example:

```text
Input:
Tôi muốn uống nước

Output:
ຂ້ອຍຕ້ອງການດື່ມນ້ໍາ
```

---

## Dataset

The original dataset contains Vietnamese-Lao parallel sentences.  
Only Vietnamese → Lao sentence pairs were used for fine-tuning.

The dataset was split into:

```text
train 90% / validation 5% / test 5%
```

---

## Author

**Phạm Tuấn Anh**  
GitHub: https://github.com/Phamanh12345

---

