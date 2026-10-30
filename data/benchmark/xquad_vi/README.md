# XQuAD tiếng Việt cho thử nghiệm RAG

**1.190 câu hỏi–đáp án có sẵn, 240 đoạn văn thuộc 48 bài nguồn.** Bộ này độc lập
với 13 case pilot đã dùng phát triển thuật toán trong repo. Không phải dữ liệu QA
do trợ lý tự sinh. Đây là bản chuyển đổi cho RAG, **không phải giao thức đánh giá
XQuAD chính thức** và không nên so điểm retrieval với F1/EM công bố của XQuAD.

Kết quả đã chạy trên cả hai góc nhìn: [RESULTS.md](RESULTS.md).

## Nguồn và giấy phép

- XQuAD, Google DeepMind: https://github.com/google-deepmind/xquad
- Artetxe, Ruder & Yogatama, *On the Cross-lingual Transferability of Monolingual
  Representations* (2020): https://aclanthology.org/2020.acl-main.421/
- Phiên bản cố định: `7d30520c717524000f0d9d2f9c10a069acd9d285`.
- `source/xquad.vi.json`: nguyên bản tiếng Việt; SHA-256 trong `manifest.json`.
- Văn bản và QA được dịch chuyên nghiệp từ một phần SQuAD v1.1 theo mô tả nguồn.
- Dữ liệu gốc và dữ liệu chuyển đổi trong thư mục này: **CC BY-SA 4.0**. Giữ ghi
  công, thông tin thay đổi và giấy phép khi chia sẻ. Toàn văn tại `source/CC-BY-SA4.0.txt`.
- Thay đổi ở repo này: schema, metadata, gán nhóm lĩnh vực, chia dev/test,
  trích evidence tự động, thêm góc nhìn văn bản dài. Không sửa nội dung QA nguồn.

## Quy mô và hai cách tổ chức cùng dữ liệu

| Góc nhìn | Tập | Văn bản | Câu hỏi | Bài nguồn |
|---|---|---:|---:|---:|
| paragraph | full | 240 | 1.190 | 48 |
| paragraph | dev | 50 | 233 | 10 |
| paragraph | test | 190 | 957 | 38 |
| article | full | 48 | 1.190 | 48 |
| article | dev | 10 | 233 | 10 |
| article | test | 38 | 957 | 38 |

`paragraph`: mỗi context XQuAD là một văn bản, 39–763 đơn vị khoảng trắng,
trung vị 155. Đây là cách giữ cấu trúc context gốc, nhưng nhiều context ngắn nên
không kích hoạt đầy đủ việc cắt văn bản dài của Ours.

`article`: ghép 5 context cùng title theo thứ tự file nguồn, dài 496–2.169 đơn vị,
trung vị 804,5. Đây là **văn bản ghép từ các đoạn trích**, không phải bài Wikipedia
đầy đủ hay bảo đảm các đoạn vốn liền kề. Dùng để kiểm tra chunking dài trong cùng
chủ đề. Metadata giữ ID/offset của từng context gốc.

Hai cách tổ chức dùng cùng câu hỏi và nội dung. **Không cộng chúng thành 2.380
câu hỏi độc lập.** Full = dev + test; cũng không cộng full vào hai tập con.

7 lĩnh vực (nhãn rộng do project gán theo title): khoa học–công nghệ, tự nhiên–
sức khỏe, lịch sử–tôn giáo, địa lý, giáo dục, xã hội–kinh tế–luật, văn hóa–thể thao–
truyền thông. Phân bố cụ thể trong `manifest.json`.

## Dev/test và quy tắc sử dụng

Hash title với seed `utc-rag-xquad-vi-v1`, lấy khoảng 20% title mỗi lĩnh vực làm
dev, phần còn lại làm test. Hai tập không trùng title, context hay ID câu hỏi;
mọi đoạn/câu hỏi cùng bài luôn đi cùng một tập. Đây là **split cục bộ của repo**,
không phải train/dev/test chính thức của XQuAD. Không dùng test để tune threshold.
Các chunker hiện có được giữ nguyên khi chạy bộ này.

Câu hỏi cùng bài có tương quan; 957 câu hỏi test không có nghĩa 957 mẫu hoàn toàn
độc lập. Đơn vị nhóm nguồn của test là 38 bài. Khi báo khoảng tin cậy, bootstrap
theo bài nguồn, không mặc nhiên coi mỗi câu hỏi là độc lập.

## Gold và hạn chế nhãn

- `gold_answer`, `gold_answers`, `source_answers`: giữ đáp án/offset của nguồn.
- `gold_evidence`: bao câu đầy đủ nhỏ nhất chứa **đáp án đầu tiên của nguồn**,
  dùng bộ tách câu hiện tại; có offset chính xác trong document.
- Evidence là **nhãn suy ra tự động**, chưa được người gán supporting facts.
  Một câu chứa đáp án có thể chưa đủ thông tin suy luận hay xác định thực thể.
- Giữ cách chấm hiện tại: một chunk phải chứa toàn bộ evidence ở đúng document.
  Hai chunk cộng lại đủ evidence vẫn miss. Đây không phải đánh giá câu trả lời LLM.
- Bộ này không có câu hỏi unanswerable, ít kiểm tra suy luận nhiều bước và không
  có nhãn boundary/dependency đầy đủ. Không dùng nó để tuyên bố độ chính xác
  boundary hay khả năng chống hallucination.
- Model có thể từng thấy dữ liệu công khai này trong pretraining; không khẳng
  định đây là dữ liệu hoàn toàn chưa thấy đối với MiniLM hoặc Qwen.

Đã kiểm tra tất cả offset, answer nằm trong evidence, evidence nằm trong nguồn,
ID duy nhất, split không rò rỉ và context không trùng. Có **9 nhóm câu hỏi trùng**
(1.181 chuỗi câu hỏi chuẩn hóa duy nhất), một số có đáp án khác nhau. Audit tại
`duplicate_question_audit.json`; không âm thầm xóa hoặc sửa nhãn.

`review.csv` có toàn bộ câu hỏi/evidence để người rà soát. Ba nhãn nguồn đáng
nghi đã được đánh dấu từ lúc đọc dữ liệu trước khi chạy chunker. Đây không phải
danh sách lỗi đầy đủ. Số lượng lớn hơn không thay thế việc rà soát chất lượng.
Nếu sửa nhãn, tạo phiên bản dataset mới và áp dụng cùng nhãn cho cả hai phương pháp.

## Chạy

Ví dụ trên văn bản dài, tập test, retrieval thật, không gọi LLM:

```powershell
.\venv\Scripts\python.exe evaluation/compare_rag.py --documents data/benchmark/xquad_vi/article/test/documents.json --questions data/benchmark/xquad_vi/article/test/questions.json --output-dir evaluation/results/xquad_vi_article_test
```

Đổi `article` thành `paragraph` để chạy trên context gốc; đổi `test` thành `dev`
để thử trên tập phát triển. Luôn dùng output-dir riêng để không ghi đè pilot.
CLI không tham số vẫn dùng bộ pilot cũ.

Khi bạn sẵn sàng tự chạy Qwen local, thêm `--ollama` và dùng output-dir mới. Tập
test có 957 câu hỏi, nên hai chiến lược sẽ gọi LLM **1.914 lần**. Task xây dữ liệu
này không chạy Ollama. Có gold answer nhưng chưa tự tính QA Accuracy cho câu trả
lời sinh tự do; cần thống nhất rubric/EM/F1 và cách xử lý trích dẫn trước khi chấm.

Tóm tắt retrieval theo lĩnh vực và khoảng tin cậy chênh lệch:

```powershell
.\venv\Scripts\python.exe evaluation/summarize_rag.py --report evaluation/results/xquad_vi_article_test/rag_comparison.json --questions data/benchmark/xquad_vi/article/test/questions.json
```

Script resample theo bài nguồn, 5.000 lượt, seed cố định; báo 95% percentile CI
cho chênh lệch paired giữa hai phương pháp và điểm sau loại câu hỏi trùng theo
quy tắc cố định. Đây là phân tích khám phá, không phải chứng minh ưu thế chung.

Tái tạo artifact từ nguồn đã lưu (không cần mạng):

```powershell
.\venv\Scripts\python.exe scripts/build_xquad_benchmark.py --offline
.\venv\Scripts\python.exe -m unittest discover -s tests
```

Builder tái tạo cả `review.csv`: lưu bản review của bạn ra file riêng trước khi
chạy lại. `manifest.json` giữ quy trình, checksum và thống kê để tái lập.
