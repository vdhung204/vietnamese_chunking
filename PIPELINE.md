# Pipeline phân đoạn văn bản tiếng Việt theo ngữ nghĩa

## 1. Đầu vào

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

## 2. Tiến trình xử lý

### Bước 1. Tách văn bản thành câu

**Hàm:** `split_sentences(text)`, được gọi trong `create_first_chunk(text)`.

**Đầu vào:** văn bản cần phân đoạn.

**Xử lý:** loại khoảng trắng ở hai đầu và tách câu dựa trên dấu kết thúc câu `.`, `?`, `!` có khoảng trắng theo sau. Giữ thứ tự và dấu câu.

**Đầu ra:** danh sách câu có thứ tự.

```text
Văn bản → [S0, S1, S2, ..., Sn]
```

Nếu văn bản rỗng, pipeline kết thúc với danh sách chunk rỗng.

### Bước 2. Xác định các điểm cắt ứng viên

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

### Bước 3. Tạo embedding và tính độ tương đồng giữa các câu

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

### Bước 4. Phân tích các điểm cắt

**Đầu vào:** các điểm cắt ứng viên và những câu xung quanh.

**Xử lý:** thu thập các tín hiệu phục vụ lựa chọn điểm cắt:

| Tín hiệu | Hàm xử lý | Thông tin thu được |
|---|---|---|
| Độ dài | `calculate_size_score(token_count)` | Điểm phù hợp với kích thước chunk mục tiêu |
| Ngữ nghĩa | `calculate_boundary_semantic_drop(boundary)` → `calculate_semantic_similarity_drop(s1, s2, s3, s4)` | Mức suy giảm liên kết tại điểm cắt, dựa trên điểm tương đồng của bước 3 |
| Phụ thuộc câu | `dependency_risk(left_text, right_text)` | Nguy cơ tách rời hai câu có liên hệ như “nếu–thì”, “tuy–nhưng” |

Đối với nhánh văn bản dài trên 240, `normalize_semantic_drops(boundaries)` chuẩn hoá mức suy giảm ngữ nghĩa giữa các ứng viên hợp lệ; `score_boundaries(boundaries)` kết hợp điểm độ dài, ngữ nghĩa và phụ thuộc thành danh sách điểm tổng hợp.

**Đầu ra:** mức suy giảm ngữ nghĩa hoặc điểm tổng hợp để xếp hạng điểm cắt. Tín hiệu thực sự sử dụng phụ thuộc vào nhánh lựa chọn ở bước 5; không phải mọi điểm cắt đều có đủ ngữ cảnh để đánh giá ngữ nghĩa.

### Bước 5. Chọn điểm cắt và tạo đoạn nội dung mới

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

### Bước 6. Ghép ngữ cảnh và lưu chunk

**Hàm:** `semantic_chunk(text)`; dùng `split_sentences()` để lấy danh sách câu của đoạn mới và phần còn lại.

**Đầu vào:** đoạn nội dung mới và phần ngữ cảnh được giữ từ lượt trước.

**Xử lý:** ghép ngữ cảnh đó vào đầu đoạn mới, rồi lưu vào danh sách kết quả. Chunk đầu tiên không có ngữ cảnh kế thừa.

**Đầu ra:** một chunk hoàn chỉnh.

```text
Chunk đầu ra = Ngữ cảnh từ đoạn trước + Đoạn nội dung mới
```

### Bước 7. Chuẩn bị ngữ cảnh cho đoạn tiếp theo

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

## 3. Đầu ra

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

## 4. Đánh giá đầu ra

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

## 5. Quy ước khi đọc pipeline

- Độ dài 120, 180 và 240 được đo bằng số đơn vị phân cách bởi khoảng trắng, chưa phải token theo tokenizer của mô hình.
- Khoảng 120–240 áp dụng khi chọn điểm cắt; chưa bảo đảm kích thước chunk cuối cùng vì overlap được thêm sau đó và có trường hợp không tìm được điểm cắt.
- Các câu được ghép bằng dấu cách nên đầu ra không giữ nguyên định dạng xuống dòng của văn bản gốc.
