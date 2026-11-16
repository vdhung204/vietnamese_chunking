# Thực nghiệm và tái lập

Tài liệu này mô tả pipeline đánh giá hiện tại: **Max–Min so với phương pháp đề xuất**, với trọng số semantic/size/risk **0,5/0,3/0,6**. Chạy mọi lệnh từ thư mục gốc dự án. Ví dụ sử dụng môi trường `.venv` được tạo theo [README](../README.md); có thể thay bằng đường dẫn Python của môi trường đang dùng.

## Dữ liệu

| Bộ dữ liệu | Vai trò | Quy mô |
|---|---|---|
| Pilot | Phát triển và chẩn đoán hành vi | 13 văn bản, 16 câu hỏi retrieval |
| XQuAD tiếng Việt dev | Khảo sát tham số | 233 câu hỏi, 10 bài nguồn |
| XQuAD tiếng Việt test | Báo cáo kết quả | 957 câu hỏi, 38 bài nguồn |

XQuAD có hai cách tổ chức: `paragraph` giữ context gốc; `article` ghép các context cùng title. Hai góc nhìn dùng cùng nội dung và câu hỏi. Split cục bộ tách theo bài nguồn; đây là bản chuyển đổi phục vụ retrieval, không phải giao thức EM/F1 chính thức của XQuAD. Xem [nguồn, giấy phép và nhãn](../data/benchmark/xquad_vi/README.md).

## Pipeline dùng chung

```text
Document → chunker được truyền vào → chunk có document_id/chunk_id
         → MiniLM dùng chung → chỉ mục cosine trong bộ nhớ
Question → embedding dùng chung → top-k → chấm evidence
                                       → prompt → LLM tùy chọn
```

- `src/rag/embedder.py` dùng lại instance MiniLM của chunker.
- `src/rag/retriever.py` xếp hạng trên toàn corpus; khi bằng điểm, giữ thứ tự corpus.
- `src/rag/rag_pipeline.py` điều phối index, retrieve và answer.
- `evaluation/compare_rag.py` chạy hai phương pháp với cùng các thành phần phía sau.
- `evaluation/evaluate_retrieval.py` kiểm tra dữ liệu và chấm evidence.

Gold evidence không tham gia truy xuất hoặc prompt. Không dùng reranker. Query embeddings được tính một lần để chia sẻ giữa hai phương pháp.

## Chạy retrieval

Cài `requirements.txt` cho retrieval hoặc `requirements-gui.txt` nếu cần giao diện. Các lệnh hiện tại không cần LlamaIndex hay Ollama.

Ví dụ đánh giá context gốc, ghi ra thư mục riêng:

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluation/compare_rag.py --documents data/benchmark/xquad_vi/paragraph/test/documents.json --questions data/benchmark/xquad_vi/paragraph/test/questions.json --output-dir evaluation/results/local_paragraph_test
```

Với văn bản ghép dài:

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluation/compare_rag.py --documents data/benchmark/xquad_vi/article/test/documents.json --questions data/benchmark/xquad_vi/article/test/questions.json --output-dir evaluation/results/local_article_test
```

Đổi `test` thành `dev` khi khảo sát tham số. CLI không truyền đường dẫn sử dụng pilot: `data/test_case.json` và `data/benchmark/questions.json`. Luôn chọn output riêng cho mỗi thí nghiệm để giữ được kết quả đối chiếu.

Đầu ra gồm:

| Artifact | Nội dung |
|---|---|
| `rag_comparison.csv` | Metric tổng hợp, số chunk, lượng context và thời gian |
| `rag_comparison.json` | Chunk, ranking, score, prompt, chẩn đoán miss, cấu hình và provenance |
| `rag_statistical_summary.json` | Bootstrap ghép cặp và phân tích nhóm, tạo bằng lệnh dưới đây |

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluation/summarize_rag.py --report evaluation/results/local_paragraph_test/rag_comparison.json --questions data/benchmark/xquad_vi/paragraph/test/questions.json
```

Phép thống kê dùng 5.000 lượt bootstrap theo **bài nguồn**, seed 2026, và khoảng percentile 95%. Các câu hỏi cùng bài có tương quan nên không coi 957 câu là 957 mẫu độc lập.

## Đọc metric

Evidence và chunk được chuẩn hóa Unicode NFC, chữ hoa/thường và khoảng trắng. Một hit yêu cầu **một chunk** trong top-k chứa toàn bộ evidence từ đúng document; hai chunk cộng lại đủ evidence vẫn không được tính hit.

- **Evidence Recall@3/@5:** tỷ lệ câu hỏi có hit trong top-3/top-5.
- **MRR@5:** nghịch đảo hạng hit đầu tiên trong top-5, bằng 0 nếu không hit.
- **Evidence preservation:** tỷ lệ câu hỏi có ít nhất một chunk đủ evidence ở bất kỳ vị trí nào trong corpus.
- **`evidence_split`:** không có chunk nào giữ đủ evidence.
- **`ranking_miss`:** chunk đủ evidence tồn tại nhưng không vào top-5.

`--top-k` phải từ 5 trở lên. Metric vẫn chấm tại mốc 3/5; prompt dùng toàn bộ top-k yêu cầu. Cùng top-k không đồng nghĩa cùng lượng context. Các trường độ dài `words` đếm đơn vị phân cách bằng khoảng trắng, không phải token của mô hình.

## Kết quả của cấu hình triển khai hiện tại

Nguồn chính là [thư mục lần chạy ngày 16/11/2026](../evaluation/results/scientific_paper_current), sử dụng cấu hình đã đóng băng trong [protocol](../evaluation/results/scientific_paper_current/protocol.json).

| Góc nhìn | Phương pháp | Số chunk | Evidence Recall@5 | MRR@5 | Context top-5 TB |
|---|---|---:|---:|---:|---:|
| Context gốc | Max–Min | 583 | 86,94% | 0,7421 | 361,20 |
| Context gốc | Đề xuất | 226 | 91,85% | 0,8057 | 784,05 |
| Ghép dài | Max–Min | 556 | 85,27% | 0,7259 | 386,55 |
| Ghép dài | Đề xuất | 191 | 80,67% | 0,6553 | 975,71 |

Context tính theo đơn vị khoảng trắng, gồm cả phần lặp. Chênh lệch Recall@5 của phương pháp đề xuất so với Max–Min là **+4,91** điểm phần trăm ở context gốc và **−4,60** điểm ở ghép dài. Bài báo trình bày cả hai thiết lập, cùng khoảng bootstrap và phân tích ngân sách context.

| Artifact hiện tại | Công dụng |
|---|---|
| [metrics_verified.csv](../evaluation/results/scientific_paper_current/metrics_verified.csv) | Đối chiếu metric trực tiếp từ ranking đã lưu |
| [Context gốc](../evaluation/results/scientific_paper_current/xquad_vi_paragraph_test/rag_comparison.json) | Ranking và metric chi tiết |
| [Ghép dài](../evaluation/results/scientific_paper_current/xquad_vi_article_test/rag_comparison.json) | Ranking và metric chi tiết |
| [tokenizer_summary.csv](../evaluation/results/scientific_paper_current/tokenizer_summary.csv) | Độ dài token và tỷ lệ vượt giới hạn encoder |
| [budget_summary.csv](../evaluation/results/scientific_paper_current/budget_summary.csv) | Khảo sát trần context trên tiền tố top-5 đã lưu |
| [manuscript_evidence.json](../evaluation/results/scientific_paper_current/manuscript_evidence.json) | Liên kết bản thảo với cấu hình và checksum artifact |

Encoder trong lần chạy có giới hạn 128 model token. Pipeline không chia lại chunk để tránh truncation; tỷ lệ chunk vượt giới hạn không tự cho biết evidence nào bị mất. Phép đo budget chỉ giữ tiền tố gồm các chunk nguyên vẹn trong top-5 đến trước lần vượt trần đầu tiên; không backfill và chưa phải so sánh fixed-budget trên toàn ranking.

Nhãn evidence được suy ra tự động từ đáp án nguồn và có các hạn chế nêu trong tài liệu dữ liệu. Nghiên cứu tham số được thiết kế sau khi kết quả test cũ đã được quan sát; lần chạy mới không thay thế xác nhận trên một corpus độc lập. Các kết quả này không đo chất lượng đáp án LLM hay chứng minh lợi ích nhân quả riêng của luật liên kết câu.

## Khảo sát tham số và tái lập bài báo

Nghiên cứu dev gồm 180 cấu hình, trong đó có 54 bộ trọng số. Tỷ lệ semantic/size 0,5/0,3 có Recall@5 trung bình cao nhất trong nhóm trọng số. Các mức penalty ở tỷ lệ này đồng hạng; giữ 0,6 là lựa chọn giữa các mức đồng hạng. Cấu hình được triển khai không phải cấu hình tốt nhất trong toàn bộ 180 lựa chọn.

[Phụ lục tham số](../THAM_SO_VA_MINH_CHUNG.md) mô tả đầy đủ lưới thử, ablation, các ngưỡng và hằng số. [active_configuration.json](../evaluation/results/parameter_study/active_configuration.json) ghi kết quả đối chiếu cấu hình CLI/GUI với artifact đã chọn.

Sau khi model đã có trong cache, có thể chạy lại nghiên cứu dev vào thư mục riêng:

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluation/parameter_sensitivity.py --output-dir evaluation/results/local_parameter_study --threads 4
```

Các tiện ích tái lập hiện có:

| Script | Vai trò và đầu ra |
|---|---|
| `scripts/build_xquad_benchmark.py --offline` | Tạo lại bản chuyển đổi dữ liệu từ nguồn đã lưu; tạo lại cả `review.csv` |
| `scripts/verify_selected_weights.py` | Kiểm tra cấu hình đang chạy với kết quả dev đã lưu; cập nhật `active_configuration.json` |
| `scripts/run_paper_experiments.py` | Tái tạo kết quả test và diagnostics trong `evaluation/results/scientific_paper_current/` |
| `scripts/write_scientific_paper.py` | Sinh lại `BAO_CAO.md` từ artifact thực nghiệm |

Các script ghi vào đường dẫn kết quả đã lưu sẽ cập nhật artifact ở đó. Giữ bản review thủ công riêng với `review.csv` do builder tạo tự động. Tiện ích xuất Word `scripts/build_paper_docx.py` cần môi trường dựng tài liệu và bộ chuyển đổi Office riêng, không thuộc bộ dependency để chạy ứng dụng.

## Sinh câu trả lời tùy chọn

`OllamaClient` mặc định gọi `http://localhost:11434/api/generate`, model `qwen2.5:3b`, temperature 0 và tối đa 512 output token. Cần tự chạy dịch vụ và chuẩn bị model. Thêm `--ollama` vào lệnh benchmark để bật generation; đổi URL bằng `--ollama-url`.

CLI retrieval mặc định không gọi LLM. Một lần chạy test đủ hai phương pháp với `--ollama` phát sinh 1.914 request. Đáp án thật được lưu nhưng chưa tự chấm QA Accuracy, EM/F1 hay hallucination. Interface `LLMClient.generate(prompt)` cho phép thay adapter; khi so sánh, dùng cùng generator và cấu hình cho cả hai phương pháp.

## Kiểm tra và pilot lịch sử

Cài `requirements-benchmark.txt` khi chạy toàn bộ unit test vì suite có adapter LlamaIndex lịch sử:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

`evaluation/evaluate_chunking.py` kiểm tra ràng buộc pilot cũ. Nhãn chỉ mô tả một phần vị trí cắt và liên kết, không phải tập segmentation/dependency đầy đủ. Case `010` có kỳ vọng overlap lịch sử khác cấu hình trọng số hiện tại; nhãn gốc được giữ để truy vết. Khác biệt này được ghi trong phụ lục tham số và `active_configuration.json`, không được đổi nhãn để làm kết quả đạt.

[Thư mục lưu trữ](archive/README.md) giữ tài liệu LlamaIndex, pipeline pilot và kết quả ngày 30/10/2026. Các artifact cũ ở `evaluation/results/xquad_vi_*_test/` dùng cấu hình tham chiếu 0,4/0,4/0,6; chúng không thay thế các bảng test mới trong bài báo.
