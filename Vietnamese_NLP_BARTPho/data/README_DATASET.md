# VSL Vietnamese NLP Gloss-to-Sentence Dataset v4 Mega

## Mục đích

Dataset này dùng cho module NLP của hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam.
Nhiệm vụ chính là chuyển chuỗi gloss đã nhận diện thành câu tiếng Việt tự nhiên.

Ví dụ:

```text
toi muon uong_nuoc -> Tôi muốn uống nước.
em cam_thay lanh -> Em cảm thấy lạnh.
```

## Thống kê

- Số mẫu: 32,755
- Số gloss token: 262
- Số nhóm ngữ cảnh: 32
- Train / Val / Test: 26,204 / 3,275 / 3,276

## Nguồn dữ liệu

Đây là silver dataset/tập dữ liệu bán thủ công - sinh mẫu theo template.
Dữ liệu được xây dựng dựa trên:
1. Các gloss thường dùng trong hệ thống nhận diện VSL.
2. Các tình huống giao tiếp phổ biến trong đời sống.
3. Các mẫu câu tiếng Việt được chuẩn hóa thủ công theo ngữ cảnh.
4. Các mẫu câu mở rộng sinh tự động bằng template có kiểm soát.

Hiện chưa có public dataset hoàn chỉnh cho bài toán:
"chuỗi gloss ngôn ngữ ký hiệu Việt Nam -> câu tiếng Việt tự nhiên".
Vì vậy dataset này được dùng làm dữ liệu khởi tạo cho rule-based correction,
kiểm thử module SentenceBuilder và fine-tune BARTpho ban đầu.

## File

- `sentence_pairs_full_v4.csv`: dataset đầy đủ có metadata.
- `sentence_pairs_simple.csv`: bản 2 cột dùng fine-tune BARTpho.
- `sentence_pairs_train_simple.csv`: train set bản 2 cột.
- `sentence_pairs_val_simple.csv`: validation set bản 2 cột.
- `sentence_pairs_test_simple.csv`: test set bản 2 cột.
- `sentence_rules.json`: luật chuyển đổi gloss -> câu cho các mẫu phổ biến.
- `gloss_vocabulary_full_v4.csv`: danh sách gloss token.
- `dataset_summary_v4.json`: thống kê dataset.

## Cột dữ liệu

- `id`: mã mẫu.
- `domain`: nhóm ngữ cảnh.
- `intent`: ý định câu.
- `gloss_sequence`: chuỗi gloss đầu vào.
- `vietnamese_sentence`: câu tiếng Việt đầu ra.
- `complexity`: mức độ câu.
- `source`: cách tạo dữ liệu.
- `split`: train/val/test.

## Khuyến nghị sử dụng

Dùng các file `*_simple.csv` cho fine-tune BARTpho:

```powershell
python -m src.training.train_bartpho `
  --train data\sentence_pairs_train_simple.csv `
  --val data\sentence_pairs_val_simple.csv `
  --test data\sentence_pairs_test_simple.csv `
  --output models\bartpho_sentence
```

## Lưu ý

Dataset này chưa phải ground-truth được thu từ cộng đồng người dùng ký hiệu thật.
Khi triển khai sản phẩm thực tế, nên tiếp tục hiệu chỉnh bằng dữ liệu thật từ người dùng
và map lại toàn bộ gloss theo `class_map.csv` của mô hình nhận diện VSL400.

