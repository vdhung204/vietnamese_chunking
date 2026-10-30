# Phân tích hậu nghiệm phục vụ rà soát báo cáo

Ngày báo cáo: 30/10/2026. Phân tích đọc artifact có sẵn; không chạy lại chunking,
không sửa test/nhãn, không chọn tham số và không gọi LLM.

## Tái lập

Từ thư mục gốc dự án:

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/audit_report_evidence.py
```

Yêu cầu tokenizer đã có trong Hugging Face cache ở snapshot
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Script dùng offline và lưu hash
tokenizer/config, phiên bản thư viện, checksum đầu vào và mã nguồn trong `audit.json`.
Snapshot này không được pin trong artifact retrieval cũ, nên không thể khẳng định
revision của lần chạy cũ chỉ từ metadata. Không có thành phần ngẫu nhiên, seed không
áp dụng. Không thay seed/bootstrap 2026 của kết quả gốc.

## Nội dung artifact

- `audit.json`: provenance, metric gốc đã đối chiếu, tokenizer, budget và dependency.
- `original_metrics_verified.csv`: tính lại từ từng câu hỏi và đối chiếu JSON/CSV gốc.
- `tokenizer_summary.csv`: độ dài token theo bốn cấu hình.
- `chunk_token_lengths.csv`: độ dài từng chunk, gồm special tokens, không truncation.
- `budget_summary.csv`: chỉ số theo trần 512 và 1024 token.
- `budget_questions.csv`: số chunk/token, hit và reciprocal rank cho từng câu hỏi/budget.
- `report_revision.json`: đầu ra báo cáo và thông tin bản sao trước chỉnh sửa.

## Quy tắc budget cần giữ khi diễn giải

Đây là **top-5 có trần token**, không phải full-ranking fixed-budget retrieval.
Duyệt top-5 đã lưu theo thứ hạng, nối nguyên chunk bằng hai ký tự xuống dòng và
dừng trước chunk đầu tiên làm vượt ngân sách. Không cắt, không bỏ qua ứng viên,
không lấy thêm ngoài top-5. Đếm tokenizer tokens của context nối, không có special
tokens, prompt hoặc câu hỏi. Đây không phải tokenizer của một LLM được đánh giá.
Cùng trần nhưng token thực dùng có thể khác; có thể còn dư ngân sách do giới hạn
top-5. Các số liệu này là hậu nghiệm trên test đã xem, không dùng để chọn budget
thắng/thua và chưa có khoảng tin cậy mới.

## Truncation và dependency

Độ dài chunk có tính special tokens được đối chiếu với `max_seq_length=128` trong
`sentence_bert_config.json`, không nhầm với `model_max_length` của tokenizer.
Không đo vị trí evidence theo offset, không báo tỷ lệ evidence bị truncation.

Dependency dùng nhãn có sẵn tại `data/benchmark/annotations.json` và chunk pilot
Max–Min/Ours trong `evaluation/results/rag_comparison.json`. Có 3 ranh giới dương
tính trong 3 tài liệu và 1 tài liệu đối chứng ngoại lệ. Vi phạm tính theo điểm cắt
(cuối chunk trước), không phải khả năng overlap giữ nguyên quan hệ. Max–Min: 2/3;
Ours: 0/3. Cả 4 tài liệu dưới 240 đơn vị khoảng trắng, không kích hoạt dependency
penalty. Đây không phải chứng minh hiệu quả riêng của penalty hoặc benchmark độc lập.

## Tính toàn vẹn

Script xác nhận hash trước/sau của `src/**/*.py` và `baselines/*.py` giống nhau.
Trong lần rà soát, các hash source lưu trong cả hai artifact XQuAD gốc cũng khớp
mã hiện tại. Số liệu gốc được đối chiếu từ ranking, corpus và nhãn trước khi ghi
artifact mới. Các kết quả chính và test set gốc được giữ nguyên.
