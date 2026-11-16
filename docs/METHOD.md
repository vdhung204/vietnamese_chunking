# Phương pháp phân đoạn và đối chứng

Tài liệu này mô tả thuật toán triển khai và đối chứng Max–Min. [Bài báo](../BAO_CAO.md) trình bày phương pháp theo cấu trúc khoa học; [hướng dẫn thực nghiệm](EXPERIMENTS.md) mô tả cách chạy và chấm kết quả.

## Cấu hình đang triển khai

| Thành phần | Giá trị | Ý nghĩa |
|---|---|---|
| Semantic weight | 0,5 | Trọng số semantic drop đã chuẩn hóa |
| Size weight | 0,3 | Trọng số ưu tiên độ dài |
| Linking-risk penalty | 0,6 | Trừ điểm khi khớp dấu hiệu liên kết câu |
| MIN / TARGET / MAX | 120 / 180 / 240 | Đếm đơn vị phân cách bằng khoảng trắng của phần lõi |
| Raw semantic-drop threshold | 0,30 | Ngưỡng cắt tùy chọn khi phần còn lại nằm trong MIN–MAX |
| Overlap thấp / cao | 0,30 / 0,55 | Cosine quyết định giữ 0, 1 hoặc tối đa 2 câu |
| Sentence embedding cache | 512 mục | Tái sử dụng vector câu trong luồng chạy thông thường |

Trong nhánh văn bản dài, các ứng viên có độ dài phần trái trong MIN–MAX được chấm:

```text
F = 0.5 × normalized_semantic_drop + 0.3 × size_score − 0.6 × linking_risk
```

Semantic drop thô là trung bình cosine ở hai phía trừ cosine qua ranh giới. Chỉ các điểm có đủ bốn câu lân cận mới có drop hợp lệ. Điểm ngữ nghĩa dùng để xếp hạng được chuẩn hóa min–max trong tập ứng viên hợp lệ; trường hợp thiếu ngữ cảnh, không có drop hoặc không có biến thiên được gán 0. Size score có dạng tam giác, đạt đỉnh ở TARGET và được làm tròn hai chữ số thập phân. `linking_risk` nhận 0 hoặc 1 từ luật khớp dấu hiệu liên kết, không phải xác suất hay bộ phân tích cú pháp.

Các mức penalty được thử đồng hạng ở tỷ lệ semantic/size đã chọn. Các độ dài và ngưỡng còn lại là cấu hình tham chiếu được giữ lại; không có bằng chứng rằng mọi giá trị đang dùng đều tối ưu. [Phụ lục tham số](../THAM_SO_VA_MINH_CHUNG.md) giải thích lựa chọn và giới hạn của từng nhóm số.

## Pipeline đề xuất

### 1. Đầu vào

Một văn bản tiếng Việt gồm nhiều câu, có thể chứa một hoặc nhiều chủ đề.

**Mục tiêu xử lý:** chia văn bản thành các đoạn nhỏ (chunk), lựa chọn điểm cắt dựa trên độ dài và sự thay đổi ngữ nghĩa, đồng thời giữ lại một phần ngữ cảnh giữa các đoạn liên tiếp.

**Hàm điều phối:** `semantic_chunk(text)` gọi `create_first_chunk(remaining_text)` để tạo từng đoạn, ghép overlap và lặp cho đến khi hết văn bản.

```text
Văn bản đầu vào → semantic_chunk(text)
      ↓
create_first_chunk(remaining_text)
      ↓
Tách câu — split_sentences(text)
      ↓
Tạo điểm cắt — create_candidate_boundaries(sentences)
      ↓
Tạo/lấy embedding câu — _encode_sentence(sentence) → model.encode(...)
      ↓
Đo tương đồng — calculate_semantic_similarity(left, right) → cos_sim(...)
      ↓
Đo suy giảm liên kết — calculate_boundary_semantic_drop(boundary)
                    → calculate_semantic_similarity_drop(s1, s2, s3, s4)
      ↓
Chọn điểm cắt theo độ dài phần còn lại:
  120–240: select_strong_semantic_boundary(boundaries)
  > 240:   select_best_boundary(boundaries) → score_boundaries(boundaries)
      ↓
Cắt đoạn — split_at_boundary(sentences, boundary)
      ↓
Ghép overlap từ lượt trước và lưu chunk — semantic_chunk(text)
      ↓
Chuẩn bị overlap mới — get_overlap_sentences(chunk_sentences, remaining_sentences)
                    → determine_overlap_count(left_sentence, right_sentence)
                    → calculate_semantic_similarity(left_sentence, right_sentence)
      ↓
Còn văn bản chưa xử lý?
      ├── Có → Tiếp tục xử lý phần còn lại
      └── Không → Trả về danh sách chunk
```

Sơ đồ thể hiện dòng dữ liệu khi cần xét điểm cắt. Các bước tính ngữ nghĩa được gọi bên trong hàm chọn/chấm điểm, không phải một lượt encode sẵn toàn bộ tài liệu. Phần còn lại dưới 120 được giữ nguyên; trường hợp không chọn được điểm cắt cũng đi thẳng đến ghép và lưu chunk.

### 2. Tiến trình xử lý

#### Bước 1. Tách văn bản thành câu

**Hàm:** `split_sentences(text)`, được gọi trong `create_first_chunk(text)`.

**Đầu vào:** văn bản cần phân đoạn.

**Xử lý:** loại khoảng trắng ở hai đầu và tách câu dựa trên dấu kết thúc câu `.`, `?`, `!` có khoảng trắng theo sau. Giữ thứ tự và dấu câu.

**Đầu ra:** danh sách câu có thứ tự.

```text
Văn bản → [S0, S1, S2, ..., Sn]
```

Nếu văn bản rỗng, pipeline kết thúc với danh sách chunk rỗng.

#### Bước 2. Xác định các điểm cắt ứng viên

**Hàm:** `create_candidate_boundaries(sentences)`.

**Đầu vào:** danh sách câu của phần văn bản chưa xử lý.

**Xử lý:** xem mỗi ranh giới giữa hai câu liên tiếp là một vị trí có thể cắt. Với mỗi vị trí, xác định độ dài phần bên trái và các câu lân cận để phân tích ngữ cảnh.

**Đầu ra:** tập điểm cắt ứng viên kèm độ dài tích luỹ và ngữ cảnh xung quanh.

```text
S0 | S1 | S2 | S3 | ... | Sn
   ↑    ↑    ↑    ↑
       Các điểm cắt ứng viên
```

Nếu toàn bộ phần còn lại ngắn hơn 120 đơn vị độ dài, giữ nguyên thành một đoạn và chuyển đến bước 6, không cần tạo hay xét điểm cắt.

#### Bước 3. Tạo embedding và tính độ tương đồng giữa các câu

**Hàm:** `_encode_sentence(sentence)` → `model.encode(sentence, convert_to_tensor=True)`; `calculate_semantic_similarity(left_sentence, right_sentence)` → `cos_sim(left_embedding, right_embedding)`.

**Đầu vào:** các câu cần so sánh quanh điểm cắt.

**Xử lý:** dùng mô hình `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` chuyển từng câu thành vector embedding. Sau đó tính cosine similarity giữa hai vector để đo liên kết ngữ nghĩa của cặp câu.

```text
Câu bên trái  → _encode_sentence() → Vector bên trái ─┐
                                                    ├→ cos_sim() → Điểm tương đồng
Câu bên phải → _encode_sentence() → Vector bên phải ─┘
```

`_encode_sentence()` giữ cache tối đa 512 câu. Nếu câu đã có trong cache, dùng lại vector; nếu chưa có, gọi `model.encode()`. Model được khởi tạo khi module được import, còn embedding được tính khi có yêu cầu so sánh.

**Đầu ra:** vector biểu diễn của các câu được yêu cầu và điểm tương đồng cho từng cặp đang xét. Các điểm tương đồng được chuyển sang bước 4 để tính mức suy giảm liên kết; cùng cơ chế này còn được dùng ở bước 7 để xác định overlap.

#### Bước 4. Phân tích các điểm cắt

**Đầu vào:** các điểm cắt ứng viên và những câu xung quanh.

**Xử lý:** thu thập các tín hiệu phục vụ lựa chọn điểm cắt:

| Tín hiệu | Hàm xử lý | Thông tin thu được |
|---|---|---|
| Độ dài | `calculate_size_score(token_count)` | Điểm phù hợp với kích thước chunk mục tiêu |
| Ngữ nghĩa | `calculate_boundary_semantic_drop(boundary)` → `calculate_semantic_similarity_drop(s1, s2, s3, s4)` | Mức suy giảm liên kết tại điểm cắt, dựa trên điểm tương đồng của bước 3 |
| Phụ thuộc câu | `dependency_risk(left_text, right_text)` | Nguy cơ tách rời hai câu có liên hệ như “nếu–thì”, “tuy–nhưng” |

Đối với nhánh văn bản dài trên 240, `normalize_semantic_drops(boundaries)` chuẩn hoá mức suy giảm ngữ nghĩa giữa các ứng viên hợp lệ; `score_boundaries(boundaries)` kết hợp điểm độ dài, ngữ nghĩa và phụ thuộc thành danh sách điểm tổng hợp.

**Đầu ra:** mức suy giảm ngữ nghĩa hoặc điểm tổng hợp để xếp hạng điểm cắt. Tín hiệu thực sự sử dụng phụ thuộc vào nhánh lựa chọn ở bước 5; không phải mọi điểm cắt đều có đủ ngữ cảnh để đánh giá ngữ nghĩa.

#### Bước 5. Chọn điểm cắt và tạo đoạn nội dung mới

**Hàm điều phối:** `create_first_chunk(text)`. Hàm chọn điểm là `select_strong_semantic_boundary()` hoặc `select_best_boundary()`; hàm thực hiện cắt là `split_at_boundary()`.

**Đầu vào:** phần văn bản chưa xử lý và các điểm cắt đã phân tích.

**Xử lý:** lựa chọn theo độ dài của phần văn bản này:

| Độ dài phần còn lại | Hàm | Cách xử lý |
|---|---|---|
| Dưới 120 | `create_first_chunk()` | Giữ toàn bộ thành một đoạn |
| Từ 120 đến 240 | `select_strong_semantic_boundary()` | Chọn điểm có suy giảm liên kết ngữ nghĩa lớn nhất, với phần trái dài ít nhất 120; `create_first_chunk()` chỉ cắt khi đủ ngữ cảnh và mức suy giảm đạt 0.30 |
| Trên 240 | `select_best_boundary()` → `score_boundaries()` | Xếp hạng các điểm có phần trái dài từ 120 đến 240 dựa trên độ dài, ngữ nghĩa và phụ thuộc câu; chọn điểm cao nhất |

Trong nhánh trên 240, độ dài mục tiêu là 180. Phụ thuộc câu làm giảm điểm của ứng viên. Trong nhánh 120–240, quyết định cắt chỉ dựa trên ngữ nghĩa và điều kiện độ dài, chưa xét phụ thuộc câu.

Nếu không tìm được điểm cắt phù hợp, giữ toàn bộ phần còn lại thành một đoạn.

**Đầu ra:** một đoạn nội dung mới, chưa có overlap, và phần văn bản còn lại.

```text
Phần văn bản đang xét
          ↓ chọn điểm cắt
[Đoạn nội dung mới] | [Phần chưa xử lý]
```

#### Bước 6. Ghép ngữ cảnh và lưu chunk

**Hàm:** `semantic_chunk(text)`; dùng `split_sentences()` để lấy danh sách câu của đoạn mới và phần còn lại.

**Đầu vào:** đoạn nội dung mới và phần ngữ cảnh được giữ từ lượt trước.

**Xử lý:** ghép ngữ cảnh đó vào đầu đoạn mới, rồi lưu vào danh sách kết quả. Chunk đầu tiên không có ngữ cảnh kế thừa.

**Đầu ra:** một chunk hoàn chỉnh.

```text
Chunk đầu ra = Ngữ cảnh từ đoạn trước + Đoạn nội dung mới
```

#### Bước 7. Chuẩn bị ngữ cảnh cho đoạn tiếp theo

**Hàm:** `get_overlap_sentences(chunk_sentences, remaining_sentences)` → `determine_overlap_count(left_sentence, right_sentence)` → `calculate_semantic_similarity(left_sentence, right_sentence)`.

**Đầu vào:** đoạn nội dung mới ở bước 5 và phần văn bản chưa xử lý.

**Xử lý:** so sánh câu cuối của đoạn mới với câu đầu của phần còn lại để quyết định số câu cuối cần giữ làm overlap:

| Độ tương đồng | Ngữ cảnh giữ lại |
|---|---|
| Nhỏ hơn 0.30 | Không giữ câu nào |
| Từ 0.30 đến dưới 0.55 | Giữ 1 câu cuối |
| Từ 0.55 trở lên | Giữ tối đa 2 câu cuối |

**Đầu ra:** phần ngữ cảnh chờ ghép vào chunk tiếp theo và phần văn bản cần tiếp tục xử lý.

Phần overlap được lưu riêng, không đưa trở lại phần văn bản chưa xử lý để chọn điểm cắt lần nữa. Nếu vẫn còn văn bản, lặp lại từ bước 1 trên phần còn lại; nếu không, kết thúc pipeline.

### 3. Đầu ra

Danh sách các chunk theo thứ tự xuất hiện trong văn bản:

```text
[Chunk 1, Chunk 2, ..., Chunk k]
```

Mỗi chunk là một chuỗi văn bản gồm đoạn nội dung được chọn và phần overlap nếu có. Một số câu xuất hiện ở hai chunk liền nhau nhằm duy trì ngữ cảnh.

Ví dụ minh hoạ dòng dữ liệu, giả sử quá trình xử lý đã chọn được ba đoạn và lượng overlap như sau:

```text
Đầu vào:
S0 S1 S2 S3 S4 S5 S6 S7

Sau khi chọn điểm cắt:
[S0 S1 S2] | [S3 S4 S5] | [S6 S7]

Ngữ cảnh cần giữ:
Đoạn 1 → Đoạn 2: giữ S2
Đoạn 2 → Đoạn 3: giữ S4 S5

Đầu ra:
Chunk 1: S0 S1 S2
Chunk 2: S2 S3 S4 S5
Chunk 3: S4 S5 S6 S7
```

Đây là ví dụ cấu trúc đầu ra, không phải kết quả chạy mô hình trên một văn bản cụ thể. `semantic_chunk(text)` trả danh sách chuỗi `list[str]`. Embedding câu ở bước 3 phục vụ chọn điểm cắt và overlap; đầu ra hiện tại chưa bao gồm embedding cho toàn bộ chunk. Việc tạo embedding chunk và lập chỉ mục truy xuất là bước tiếp theo, nằm ngoài pipeline đang có.

### 4. Đánh giá đầu ra

Đánh giá là bước kiểm chứng sau chunking, không tham gia quyết định điểm cắt:

```text
Văn bản kiểm thử + Nhãn kỳ vọng — load_test_cases(path)
                ↓
Chạy pipeline — evaluate_case(case) → semantic_chunk(text)
                ↓
       Danh sách chunk thực tế
                ↓
Trích điểm cắt — extract_cut_positions(text, chunks)
Trích overlap — extract_overlap_counts(chunks)
                ↓
Đối chiếu kỳ vọng — evaluate_case(case)
                ↓
Kết quả đạt/không đạt → calculate_metrics(results)
```

Các tiêu chí gồm: cắt đúng vị trí bắt buộc, giữ nguyên vị trí không được cắt, tạo đủ số lần cắt tối thiểu và có lượng overlap phù hợp.

### 5. Quy ước khi đọc pipeline

- Độ dài 120, 180 và 240 được đo bằng số đơn vị phân cách bởi khoảng trắng, chưa phải token theo tokenizer của mô hình.
- Khoảng 120–240 áp dụng khi chọn điểm cắt; chưa bảo đảm kích thước chunk cuối cùng vì overlap được thêm sau đó và có trường hợp không tìm được điểm cắt.
- Các câu được ghép bằng dấu cách nên đầu ra không giữ nguyên định dạng xuống dòng của văn bản gốc.

## Đối chứng Max–Min

Baseline mới cho Vietnamese semantic chunking là Max–Min của Kiss, Nagy &
Szilágyi. Module: `baselines/maxmin_semantic_chunker.py`.

- Paper: https://doi.org/10.1007/s10791-025-09638-7
- Code tác giả, phiên bản đối chiếu cố định:
  https://github.com/hsdslab/MaxMinChunking/blob/0caba41d2ef30114729f95e5c788173e4f40920e/maxmin_chunker.py

### Phạm vi tích hợp

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
so sánh Max–Min với Ours (xem [hướng dẫn thực nghiệm](EXPERIMENTS.md)). `compare_chunkers.py`
và kết quả RAG cũ vẫn so sánh LlamaIndex với Ours.
[Tài liệu LlamaIndex lưu trữ](archive/BASELINE.md) mô tả baseline LlamaIndex cũ, không mô tả Max–Min.

### Quyết định JOIN / NEW CLUSTER

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

### Kiểm thử

```powershell
.\venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
.\venv\Scripts\python.exe -X utf8 evaluation/evaluate_chunking.py
```

Các test Max–Min kiểm tra ngưỡng khởi tạo, dấu `>`, ngưỡng thích nghi và thứ tự
cập nhật minimum, đối chiếu với câu không liền kề trong cluster, reset trạng
thái, giữ nguyên các câu lặp và adapter dùng model chung.

## Cấu hình trong ứng dụng

Giao diện Streamlit cho phép chọn Ours hoặc Max–Min và thay đổi MIN/TARGET/MAX, semantic drop cùng hai ngưỡng overlap của Ours. Cần có `0 < MIN < TARGET < MAX` và ngưỡng overlap thấp nhỏ hơn cao. Trọng số score và bộ luật liên kết giữ theo cấu hình triển khai; nút khôi phục trả các ngưỡng về mặc định. Xem [README](../README.md) để cài đặt và chạy.

`gui/chunkers.py` nạp các module thuật toán vào runtime riêng cho cấu hình được chọn; không sửa file nguồn hoặc monkeypatch module dùng chung. Model được cache bằng `st.cache_resource`, chỉ mục và kết quả nằm trong session. Thay đầu vào hoặc cấu hình làm chỉ mục cũ hết hiệu lực cho đến khi dựng lại. JSON tải xuống ghi cấu hình thực sự đã chạy; thao tác GUI không ghi đè benchmark.

Project đặt `server.fileWatcherType = "none"` trong `.streamlit/config.toml` để tránh watcher quét các lazy module ảnh của Transformers. Sau khi sửa code, khởi động lại Streamlit để nạp thay đổi. Nếu chạy từ ngoài thư mục gốc, có thể truyền `--server.fileWatcherType none` trực tiếp. Xem [cấu hình Streamlit](https://docs.streamlit.io/develop/api-reference/configuration/config.toml) và [resource cache](https://docs.streamlit.io/develop/concepts/architecture/caching).
