# Dataset Card Report - VSL NLP Dataset v4 Mega

## 1. Tổng quan

Dataset v4 Mega được xây dựng cho bài toán chuyển chuỗi gloss ngôn ngữ ký hiệu
sang câu tiếng Việt hoàn chỉnh. Dataset gồm 32,755 mẫu câu, 262
gloss token và 32 nhóm ngữ cảnh giao tiếp.

## 2. Phân chia dữ liệu

| Split | Số mẫu |
|---|---:|
| Train | 26,204 |
| Validation | 3,275 |
| Test | 3,276 |

## 3. Các nhóm ngữ cảnh

- command: 80 mẫu
- complex_health: 800 mẫu
- complex_need: 768 mẫu
- complex_schedule: 800 mẫu
- confirmation: 200 mẫu
- daily_activity: 552 mẫu
- demo_fixed: 6 mẫu
- description: 580 mẫu
- education: 40 mẫu
- emergency: 7 mẫu
- family: 204 mẫu
- food_drink: 600 mẫu
- generated_command: 86 mẫu
- generated_emergency: 209 mẫu
- generated_family: 201 mẫu
- generated_health: 6,065 mẫu
- generated_mix: 10,181 mẫu
- generated_question: 1,029 mẫu
- greeting: 6 mẫu
- health: 532 mẫu
- location: 792 mẫu
- need: 720 mẫu
- negation: 200 mẫu
- object_action: 680 mẫu
- quantity: 3,328 mẫu
- question: 175 mẫu
- shopping: 324 mẫu
- social: 54 mẫu
- time_activity: 3,024 mẫu
- transport: 152 mẫu
- weather: 270 mẫu
- work: 90 mẫu


## 4. Bản chất dữ liệu

Đây là silver dataset được tạo bằng phương pháp bán thủ công và template có kiểm soát.
Mục đích là tạo dữ liệu đủ lớn để kiểm thử rule-based sentence builder và fine-tune
mô hình BARTpho ở giai đoạn đầu.

## 5. Hạn chế

- Chưa phải dữ liệu ground-truth thu từ người dùng ký hiệu thật.
- Một số chuỗi gloss có thể chứa token ngữ pháp hỗ trợ như `dang`, `se`, `can`, `muon`.
- Khi tích hợp với mô hình VSL400 thực tế, cần đối chiếu lại với `class_map.csv`.

## 6. Hướng mở rộng

- Thu thập gloss thực tế từ mô hình nhận diện realtime.
- Cho người dùng xác nhận câu đúng/sai.
- Bổ sung dữ liệu từ người khiếm thính/người dùng ngôn ngữ ký hiệu.
- Fine-tune lại BARTpho bằng dữ liệu đã xác thực.

