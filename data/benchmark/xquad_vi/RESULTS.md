# Kết quả retrieval ngày 30/10/2026

Hai chunker giữ nguyên code và tham số, dùng cùng MiniLM/cosine/top-5. Test có
**957 câu hỏi, 38 nhóm bài nguồn**, không trùng bài với 233 câu dev. Không gọi
Ollama. Các con số dưới đây được tính từ output thực tế, không phải số liệu giả.

| Đầu vào | Phương pháp | Chunk | Recall@3 | Recall@5 | MRR@5 |
|---|---|---:|---:|---:|---:|
| Context gốc (190 văn bản) | Max–Min | 583 | 81,61% | 86,94% (832/957) | 0,7421 |
| Context gốc (190 văn bản) | Ours | 226 | 87,98% | 91,85% (879/957) | 0,8062 |
| Ghép cùng bài (38 văn bản dài) | Max–Min | 556 | 80,56% | 85,27% (816/957) | 0,7259 |
| Ghép cùng bài (38 văn bản dài) | Ours | 191 | 73,67% | 80,56% (771/957) | 0,6543 |

Chênh lệch Recall@5 **Ours trừ Max–Min**, bootstrap paired theo bài nguồn,
5.000 lần, khoảng percentile 95%:

- Context gốc: **+4,91 điểm phần trăm**, CI [ +2,49; +7,45 ].
- Văn bản dài ghép cùng bài: **−4,70 điểm phần trăm**, CI [ −8,24; −1,32 ].

Hai thiết lập dùng cùng câu hỏi, không phải hai tập bằng chứng độc lập. Kết quả
cho thấy phương pháp tốt hơn phụ thuộc cách tổ chức đầu vào. Chưa có thí nghiệm
loại từng thành phần nên không quy chênh lệch này riêng cho overlap hay dependency.
Không chọn chỉ bảng Ours thắng và bỏ bảng Ours thua khi báo cáo.

Chi phí context top-5 trung bình (đơn vị khoảng trắng, gồm overlap):

| Đầu vào | Max–Min | Ours |
|---|---:|---:|
| Context gốc | 361,20 | 783,85 |
| Ghép cùng bài | 386,55 | 977,40 |

Cùng top-k nhưng Ours lấy lượng văn bản lớn hơn. Mọi lần encode vẫn chịu giới
hạn độ dài của MiniLM; pipeline không thay đổi giới hạn hay cắt lại chunk. Đây
là một yếu tố cần kiểm tra trong nghiên cứu tiếp theo, chưa phải nguyên nhân đã
được chứng minh của chênh lệch kết quả.

## Artifact và giới hạn

- [Context gốc: CSV](../../../evaluation/results/xquad_vi_paragraph_test/rag_comparison.csv)
- [Context gốc: JSON chi tiết](../../../evaluation/results/xquad_vi_paragraph_test/rag_comparison.json)
- [Context gốc: thống kê](../../../evaluation/results/xquad_vi_paragraph_test/rag_statistical_summary.json)
- [Văn bản dài: CSV](../../../evaluation/results/xquad_vi_article_test/rag_comparison.csv)
- [Văn bản dài: JSON chi tiết](../../../evaluation/results/xquad_vi_article_test/rag_comparison.json)
- [Văn bản dài: thống kê](../../../evaluation/results/xquad_vi_article_test/rag_statistical_summary.json)

JSON thống kê gồm điểm từng lĩnh vực, số câu chỉ một bên hit, lý do miss, điểm
macro theo bài và kiểm tra độ nhạy sau loại câu hỏi trùng theo quy tắc cố định.

Gold evidence được suy ra từ câu chứa answer nguồn, chưa được người rà soát.
Có nhãn nguồn đáng nghi và câu hỏi lặp như mô tả trong README/review.csv.
Bootstrap không khắc phục lỗi nhãn hay làm 38 bài nguồn đại diện cho mọi miền.
Kết quả này đo retrieval, **không phải QA Accuracy, chất lượng câu trả lời Qwen,
Boundary F1 hoặc bằng chứng rằng Ours/Max–Min luôn tốt hơn**.

Đã chạy 43 unit test và 10/10 regression case cũ; hash 16 file chunking/pipeline
không thay đổi trong task mở rộng dữ liệu.
