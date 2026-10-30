# Max–Min baseline (2025)

Baseline mới cho Vietnamese semantic chunking là Max–Min của Kiss, Nagy &
Szilágyi. Module: `baselines/maxmin_semantic_chunker.py`.

- Paper: https://doi.org/10.1007/s10791-025-09638-7
- Code tác giả, phiên bản đối chiếu cố định:
  https://github.com/hsdslab/MaxMinChunking/blob/0caba41d2ef30114729f95e5c788173e4f40920e/maxmin_chunker.py

## Phạm vi tích hợp

```python
from baselines.maxmin_semantic_chunker import maxmin_semantic_chunk

chunks = maxmin_semantic_chunk(text)  # list[str]
```

Dùng chung `src/sentence_splitter.py::split_sentences` và chính instance
`semantic_similarity.model` (paraphrase-multilingual-MiniLM-L12-v2).
Đây là thích nghi đầu vào cho tiếng Việt, không phải tái lập embedding tiếng Anh
trong thí nghiệm gốc. Không thêm luật dependency, overlap hay giới hạn độ dài.
Không tune tham số theo dữ liệu test.

Module dùng độc lập hoặc qua `evaluation/compare_rag.py`, pipeline RAG chung
so sánh Max–Min với Ours (xem `RAG.md`). `compare_chunkers.py`
và kết quả RAG cũ vẫn so sánh LlamaIndex với Ours.
`BASELINE.md` mô tả baseline LlamaIndex cũ, không mô tả Max–Min.

## Quyết định JOIN / NEW CLUSTER

Đọc các câu theo thứ tự. Với cluster có ít nhất hai câu, gọi `m` là cosine
nhỏ nhất giữa các cặp câu đã thuộc cluster; gọi `M` là cosine lớn nhất giữa
câu mới và các câu trong cluster. Ngưỡng quyết định là:

```text
threshold = max(fixed_threshold, m * c * sigmoid(cluster_size - 1))
JOIN khi M > threshold; ngược lại mở cluster mới.
```

Sau khi JOIN, cập nhật minimum với các cặp mới. Khi mở cluster mới, reset trạng
thái. Chính việc đối chiếu maximum của câu mới với minimum nội bộ là Max–Min.

Khi cluster mới chỉ có một câu, JOIN khi:

```text
init_constant * cosine(câu mới, câu duy nhất) > max(0, fixed_threshold)
```

- `fixed_threshold=0.6`: sàn ngưỡng, tránh cluster nhận câu quá ít liên quan.
- `c=0.9`: nhân vào thành phần ngưỡng thích nghi; không phải trọng số embedding.
- `init_constant=1.5`: nhân similarity khi khởi tạo cặp câu đầu tiên, khiến
  cặp đầu dễ được ghép hơn (với mặc định, cosine phải lớn hơn 0.4).

Điều kiện là `>` nghiêm ngặt. Ngưỡng thích nghi dùng minimum **trước** khi xét
thêm câu mới. Code công bố dùng `sigmoid(size - 1)`; công thức (4) trong paper
ghi `sigmoid(size)`. Module bám code công bố để tái lập quyết định. Khác biệt
kỹ thuật: xử lý input rỗng, kiểm tra shape và dùng scalar thay mảng một phần tử.
Mỗi lần xuất hiện của một câu thuộc đúng một cluster, theo thứ tự gốc.

## Kiểm thử

```powershell
.\venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
.\venv\Scripts\python.exe -X utf8 evaluation/evaluate_chunking.py
```

Các test Max–Min kiểm tra ngưỡng khởi tạo, dấu `>`, ngưỡng thích nghi và thứ tự
cập nhật minimum, đối chiếu với câu không liền kề trong cluster, reset trạng
thái, giữ nguyên các câu lặp và adapter dùng model chung.
