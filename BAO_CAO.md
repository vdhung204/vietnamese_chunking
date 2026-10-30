# PHÂN ĐOẠN VĂN BẢN TIẾNG VIỆT THEO NGỮ NGHĨA KẾT HỢP LUẬT LIÊN KẾT CÂU VÀ OVERLAP ĐỘNG CHO TRUY XUẤT TRONG RAG

*Báo cáo dựa trên mã nguồn và kết quả thực nghiệm lưu trong dự án utc_rag, cập nhật ngày 30/10/2026. Phạm vi đánh giá thực nghiệm là truy xuất bằng chứng; chưa đánh giá câu trả lời do LLM sinh ra.*

## 1. Giới thiệu bài toán

### 1.1. Vấn đề

Trong một hệ thống Retrieval-Augmented Generation (RAG), tài liệu được chia thành các đoạn nhỏ, gọi là chunk, trước khi chuyển thành vector và đưa vào bộ truy xuất. Khi người dùng đặt câu hỏi, hệ thống tìm các chunk liên quan để cung cấp ngữ cảnh cho bước sinh câu trả lời. Nghiên cứu của Lewis và cộng sự đặt nền tảng cho hướng kết hợp bộ nhớ tham số của mô hình với nguồn tri thức truy xuất bên ngoài [1]. Đề tài này tập trung vào bước phân đoạn tài liệu trong một pipeline phục vụ RAG.

Việc chọn điểm cắt tạo ra sự đánh đổi giữa lượng ngữ cảnh được giữ lại và mức độ tập trung của nội dung. Chunk nhỏ có thể tách rời thông tin cần thiết; chunk lớn có thể chứa nhiều ý khác nhau, làm tăng lượng văn bản truy xuất và gây khó khăn cho biểu diễn bằng một vector duy nhất. Với tiếng Việt, các quan hệ như “nếu–thì”, “tuy–nhưng” hoặc “không những–mà còn” còn gợi ý sự liên kết giữa hai câu ở hai phía điểm cắt.

Bài toán của đề tài là: **xây dựng và đánh giá một chiến lược phân đoạn tiếng Việt có xét ngữ nghĩa, độ dài và dấu hiệu liên kết câu, nhằm hỗ trợ truy xuất đúng bằng chứng cho câu hỏi.** Đầu vào là văn bản tiếng Việt; đầu ra là danh sách chunk giữ thứ tự nội dung và có thể chứa phần ngữ cảnh chồng lấn.

### 1.2. Hướng giải quyết

Nhóm xây dựng chiến lược kết hợp ba thành phần: chọn điểm cắt theo suy giảm tương đồng ngữ nghĩa và độ dài; giảm điểm các ranh giới có dấu hiệu phụ thuộc giữa câu; giữ lại 0–2 câu ở đầu chunk tiếp theo tùy mức tương đồng tại ranh giới. Trong báo cáo, chiến lược này được ký hiệu là **Ours** để thống nhất với các tệp kết quả.

Ours được đối chiếu với **Max–Min semantic chunking**, một phương pháp có bài báo và mã nguồn công bố [2, 3]. Cả hai được chạy trong cùng pipeline tiếng Việt. Nhóm sử dụng XQuAD tiếng Việt theo hai cách tổ chức đầu vào để xem kết quả có thay đổi khi chuyển từ context ngắn sang văn bản ghép dài hay không.

Các câu hỏi nghiên cứu gồm: Ours có truy xuất được bằng chứng tốt hơn Max–Min trong cấu hình đang xét không; kết quả phụ thuộc như thế nào vào cách tổ chức tài liệu; và mức cải thiện, nếu có, đi kèm bao nhiêu lượng context truy xuất? Đề tài không huấn luyện một LLM hoặc một mô hình embedding mới.

## 2. Tổng quan nghiên cứu

### 2.1. Các công trình liên quan và vai trò trong đề tài

| Công trình hoặc nguồn | Tác giả, năm | Nội dung liên quan | Vai trò trong đề tài |
|---|---|---|---|
| *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* [1] | Lewis và cộng sự, 2020 | Kết hợp truy xuất với sinh ngôn ngữ | Nền tảng RAG; không phải chunker đối chứng |
| *Max–Min semantic chunking of documents for RAG application* [2] | Kiss, Nagy và Szilágyi, 2025 | Phân đoạn theo quan hệ tương đồng giữa các câu | Baseline chính của thực nghiệm XQuAD |
| *Semantic Chunker*, tài liệu LlamaIndex [4] | LlamaIndex | Chọn ranh giới bằng embedding và ngưỡng percentile | Baseline thư viện đã có ở giai đoạn pilot |
| *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation* [5] | Reimers và Gurevych, 2020 | Biểu diễn câu đa ngôn ngữ bằng chưng cất tri thức | Cơ sở tham khảo về embedding đa ngôn ngữ |
| *On the Cross-lingual Transferability of Monolingual Representations* [7] | Artetxe, Ruder và Yogatama, 2020 | Công bố XQuAD | Nguồn bài báo của dữ liệu đánh giá |

Các nguồn trong bảng có vai trò khác nhau: Max–Min là đối tượng so sánh thuật toán; MiniLM là thành phần biểu diễn dùng chung; XQuAD là dữ liệu. Vì vậy, không đặt tất cả dưới tên “các mô hình so sánh”.

### 2.2. Baseline học thuật: Max–Min

Kiss, Nagy và Szilágyi công bố Max–Min trong *Discover Computing*, tập 28, bài 117, năm 2025. Phương pháp gom các câu liên tiếp dựa trên độ tương đồng. Bài báo đánh giá trên hai bộ đoạn Wikipedia xáo trộn và một bài toán hỏi đáp trắc nghiệm về tài liệu viễn thông 3GPP; sử dụng chỉ số phân cụm và độ chính xác hỏi đáp. Đây là kết quả trong thiết lập của tác giả [2], chưa trực tiếp xác nhận hiệu quả trên tiếng Việt.

Nhóm chọn Max–Min vì thuật toán giải quyết cùng bước phân đoạn và có mã nguồn để đối chiếu. Bản tích hợp được cố định theo commit `0caba41d2ef30114729f95e5c788173e4f40920e` của tác giả [3], thay đầu vào bằng bộ tách câu và embedding dùng chung trong dự án. Đây là **đánh giá bản thích nghi tiếng Việt**, không phải tái lập toàn bộ thí nghiệm gốc.

### 2.3. Baseline thư viện: LlamaIndex SemanticSplitter

LlamaIndex cung cấp `SemanticSplitterNodeParser`, chọn điểm cắt giữa câu dựa trên độ khác biệt embedding. Tài liệu chính thức minh họa với bài luận của Paul Graham và lưu ý về bộ tách câu cũng như việc điều chỉnh ngưỡng percentile. Đây là tài liệu triển khai, không nên ghi thành một bài báo phản biện khoa học riêng. Nguồn triển khai được dẫn tại [4].

Trong dự án, LlamaIndex đã được chạy trên bộ pilot 13 văn bản với 16 câu hỏi. Các kết quả đó chỉ hỗ trợ quá trình phát triển. Chưa có bảng LlamaIndex trên 957 câu hỏi XQuAD test trong các artifact hiện tại; do đó báo cáo không gán kết quả Max–Min cho LlamaIndex hoặc ghép hai thí nghiệm thành một bảng xếp hạng chung.

### 2.4. Vấn đề đề tài cần kiểm chứng

Từ phạm vi trên, nhóm đặt giả thuyết rằng kết hợp ngữ nghĩa với độ dài, luật liên kết tiếng Việt và overlap có thể giữ bằng chứng tốt hơn trong một số tình huống. Tuy nhiên, giữ được bằng chứng trong chunk chưa bảo đảm chunk đó sẽ được truy xuất ở thứ hạng cao. Vì vậy, cần đánh giá đồng thời khả năng bảo toàn bằng chứng, Recall, MRR và lượng context.

Đề tài xây dựng và đánh giá một chiến lược chunking tiếng Việt kết hợp ràng buộc độ dài khi chọn điểm cắt, relative semantic drop, dependency-aware penalty và dynamic overlap trong cùng pipeline retrieval. Đây là một phương án tích hợp và đánh giá có đối chứng; kết quả không xác lập tính mới tuyệt đối hoặc ưu thế tổng quát so với các phương pháp khác.

## 3. Kiến thức nền tảng

### 3.1. Chunking và overlap

Chunking là phép biến đổi tài liệu thành các đơn vị truy xuất. Ranh giới giữa hai chunk có thể theo độ dài, cấu trúc hoặc ngữ nghĩa. Overlap là phần nội dung được lặp lại giữa các chunk liên tiếp nhằm giữ thêm ngữ cảnh; đổi lại, nó tạo dữ liệu trùng và làm tăng lượng văn bản được lưu hoặc truy xuất.

Trong mã nguồn đề tài, độ dài được tính bằng `len(text.split())`. Đây là **số đơn vị phân cách bởi khoảng trắng**, không phải số từ đã tách theo ngôn ngữ học và cũng không phải số token của tokenizer. Sự phân biệt này quan trọng vì một từ tiếng Việt có thể gồm nhiều âm tiết cách nhau bằng khoảng trắng.

### 3.2. Embedding và cosine similarity

Mô hình `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` biểu diễn văn bản bằng vector 384 chiều. Model card công bố cấu hình `max_seq_length=128`; các đầu vào dài cần được xem xét theo giới hạn tokenizer thực tế. Thông số mô hình được dẫn tại [6].

Độ tương đồng giữa hai vector được tính bằng:

\[
\operatorname{sim}(u,v)=\frac{u\cdot v}{\|u\|\|v\|}.
\]

Trong hệ thống, embedding câu phục vụ quyết định phân đoạn; embedding toàn chunk phục vụ truy xuất; embedding câu hỏi dùng để xếp hạng các chunk. Các phép encode dùng chung mô hình, nhưng chúng biểu diễn những đơn vị đầu vào khác nhau.

### 3.3. Truy xuất và bằng chứng chuẩn

Với mỗi câu hỏi, retriever tính cosine với toàn bộ chunk trong tập tài liệu tương ứng, sau đó trả về top-k. Không dùng ID tài liệu chuẩn để thu hẹp không gian tìm kiếm trước khi truy xuất.

Một kết quả được tính là hit khi ít nhất một chunk trong top-k chứa toàn bộ `gold_evidence` và thuộc đúng tài liệu đã gán nhãn. Việc so khớp chuẩn hóa Unicode NFC, chữ hoa/thường và khoảng trắng. Đây là kiểm tra chuỗi bằng chứng trong đúng nguồn; không đánh giá mức độ tương đương ngữ nghĩa của mọi cách diễn đạt có thể chấp nhận.

### 3.4. Chỉ số đánh giá

Với N câu hỏi và hᵢ(k) bằng 1 nếu câu hỏi i có hit trong top-k:

\[
\operatorname{Evidence Recall@k}=\frac{1}{N}\sum_{i=1}^{N}h_i(k).
\]

Báo cáo dùng thuật ngữ Evidence Recall@3 và Evidence Recall@5 cho các chỉ số recall_at_3 và recall_at_5 trong mã nguồn. Với một gold evidence chính cho mỗi câu hỏi, Evidence Recall@k tương đương tỷ lệ câu hỏi có toàn bộ bằng chứng chuẩn xuất hiện trong ít nhất một chunk đúng nguồn của top-k. Đây là hit rate theo câu hỏi; không phải tỷ lệ mọi tài liệu liên quan được tìm thấy. Ký hiệu Recall trong các công thức mang cùng ý nghĩa này.

\[
\operatorname{MRR@5}=\frac{1}{N}\sum_{i=1}^{N}
\begin{cases}1/r_i,&r_i\leq5\\0,&\text{không có hit trong top-5}\end{cases}
\]

Ở đây rᵢ là thứ hạng đầu tiên của chunk hợp lệ. MRR ưu tiên các kết quả tìm thấy bằng chứng sớm hơn.

**Evidence preservation rate** đo tỷ lệ câu hỏi có bằng chứng nằm trọn trong ít nhất một chunk ở bất kỳ thứ hạng nào. **Mean context length@5** đo tổng độ dài năm chunk truy xuất, lấy trung bình theo câu hỏi và tính cả overlap. Hai chỉ số này giúp phân biệt lỗi phân đoạn với lỗi xếp hạng và chi phí context.

## 4. Đề xuất giải pháp

### 4.1. Quy trình tổng thể

```text
Tài liệu tiếng Việt
    → Tách câu
    → Tạo các ranh giới ứng viên
    → Chọn ranh giới theo độ dài và ngữ nghĩa
       (nhánh văn bản dài có thêm điểm phạt liên kết câu)
    → Tạo chunk mới
    → Ghép overlap đã lưu từ chunk trước
    → Chuẩn bị overlap cho lượt tiếp theo
    → Lặp trên phần nội dung còn lại
    → Danh sách chunk
```

Hiện thực chính nằm trong `src/semantic_chunker.py`, `src/boundary_scorer.py`, `src/semantic_similarity.py`, `src/dependency_rules.py` và `src/dynamic_overlap.py`. Đây là thuật toán kết hợp embedding với quy tắc, không có bước học trọng số từ tập XQuAD.

### 4.2. Phát hiện suy giảm liên kết ngữ nghĩa

Xét bốn câu liên tiếp s₁, s₂, s₃, s₄ và ranh giới giữa s₂ và s₃. Mức suy giảm liên kết được tính bằng:

\[
\Delta=\frac{\operatorname{sim}(s_1,s_2)+\operatorname{sim}(s_3,s_4)}{2}
-\operatorname{sim}(s_2,s_3).
\]

Giá trị lớn gợi ý liên kết qua ranh giới yếu hơn liên kết ở hai phía. Điểm cắt thiếu câu trước hoặc câu sau không có đủ ngữ cảnh cho phép tính này. Trong nhánh chấm điểm tổng hợp, các giá trị hợp lệ được chuẩn hóa min–max; trường hợp không có biến thiên được gán điểm ngữ nghĩa bằng 0.

### 4.3. Kết hợp độ dài và luật liên kết câu

Với độ dài phần bên trái ranh giới là n, điểm độ dài tăng tuyến tính từ 0 ở n=120 lên 1 ở n=180, rồi giảm xuống 0 ở n=240. Mã nguồn làm tròn điểm này đến hai chữ số thập phân.

Luật liên kết tìm các cặp dấu hiệu như “nếu–thì”, “vì–nên”, “do–nên”, “tuy–nhưng”, “mặc dù–nhưng”, “không những–mà còn”, “không phải–mà” và “hễ–thì”. Giá trị rủi ro R bằng 1 khi khớp luật, ngược lại bằng 0; có xử lý ngoại lệ cho “vì vậy” và “do đó”.

Đối với phần văn bản dài hơn 240 đơn vị, điểm ranh giới là:

\[
S(b)=0.4S_{size}(b)+0.4S_{semantic}(b)-0.6R(b).
\]

Đây là luật khớp chuỗi đơn giản, không phải bộ phân tích cú pháp phụ thuộc. Nó có thể bỏ sót liên kết diễn đạt khác hoặc nhận nhầm chuỗi con. Điểm phạt cũng không phải lệnh cấm cắt tuyệt đối.

### 4.4. Quyết định cắt theo độ dài phần còn lại

| Độ dài | Quy tắc trong hiện thực |
|---|---|
| Dưới 120 | Giữ nguyên phần còn lại |
| Từ 120 đến 240 | Xét ranh giới có phần trái ít nhất 120; cắt tại mức suy giảm lớn nhất nếu có đủ ngữ cảnh và Δ ≥ 0,30 |
| Trên 240 | Chọn ranh giới có điểm tổng hợp cao nhất trong khoảng phần trái 120–240 |
| Không có ranh giới phù hợp | Giữ nguyên phần còn lại |

Trong nhánh phân đoạn đối với phần văn bản dài hơn MAX = 240, dependency risk được đưa vào hàm scoring để giảm khả năng chọn boundary phá quan hệ liên câu. Đây là dependency-aware penalty, không phải hard constraint và không bảo đảm tuyệt đối tránh cắt. Nhánh 120–240 không sử dụng penalty này. Overlap được ghép sau khi chọn ranh giới; khi không tìm được ứng viên, phần còn lại được giữ nguyên, nên 240 không phải giới hạn cứng của chunk đầu ra.

### 4.5. Overlap động

Gọi t là cosine giữa câu cuối của đoạn nội dung mới và câu đầu của phần chưa xử lý:

| Điều kiện | Số câu cuối giữ lại cho chunk sau |
|---|---:|
| t < 0,30 | 0 |
| 0,30 ≤ t < 0,55 | 1 |
| t ≥ 0,55 | Tối đa 2 |

Overlap được lưu riêng để ghép vào đầu chunk kế tiếp. Nó không được đưa trở lại phần văn bản chưa xử lý để chọn ranh giới lần nữa. Cơ chế này thể hiện giả thuyết rằng ranh giới có liên kết mạnh nên giữ thêm ngữ cảnh; tác động riêng của giả thuyết cần được kiểm tra bằng thí nghiệm loại bỏ thành phần.

## 5. Thực nghiệm và kết quả

### 5.1. Mô hình và các chiến lược được so sánh

Pipeline thực nghiệm gồm: **tài liệu → chunker → embedding chunk → chỉ mục vector trong bộ nhớ → cosine retrieval → đánh giá evidence**. Câu hỏi được encode một lần để dùng chung cho hai phương pháp. Thực nghiệm lưu trong báo cáo không gọi Ollama hoặc LLM sinh câu trả lời.

| Thành phần | Max–Min thích nghi tiếng Việt | Ours |
|---|---|---|
| Bộ tách câu | `src/sentence_splitter.py` | Giống Max–Min |
| Embedding | Multilingual MiniLM-L12-v2 | Cùng instance mô hình |
| Quy tắc phân đoạn | So sánh tương đồng max–min trong cụm câu | Kết hợp độ dài, suy giảm ngữ nghĩa và luật liên kết |
| Tham số chính | `fixed_threshold=0.6`, `c=0.9`, `init_constant=1.5` | 120/180/240; trọng số 0,4/0,4/−0,6; ngưỡng Δ=0,30 |
| Overlap | Không thêm | 0–2 câu, ngưỡng 0,30 và 0,55 |
| Truy xuất | Toàn corpus, cosine, top-5 | Giống Max–Min |
| Chấm điểm | Cùng câu hỏi, evidence và quy tắc hit | Giống Max–Min |

Trong bản Max–Min tích hợp tại `baselines/maxmin_semantic_chunker.py`, với cụm C có ít nhất hai câu, m là cosine nhỏ nhất của các cặp trong cụm hiện tại, M là cosine lớn nhất giữa câu mới với các câu trong C. Câu mới được nhập cụm khi:

\[
M>\max(0.6,\;0.9m\,\sigma(|C|-1)),\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

Khi cụm chỉ có một câu, điều kiện là `1.5 × cosine > 0.6`. Hiện thực giữ phép so sánh `>` và cách dùng `sigmoid(size−1)` theo mã nguồn công bố [3]. Tài liệu đối chiếu trong dự án ghi nhận công thức bài báo dùng `sigmoid(size)`; vì vậy báo cáo xác định rõ lựa chọn bám mã nguồn ở commit cố định, tránh mô tả đây là một bản sao tuyệt đối của mọi công thức trong bài.

### 5.2. Dataset

XQuAD được Artetxe, Ruder và Yogatama công bố năm 2020, gồm các context và câu hỏi–đáp án dịch chuyên nghiệp từ một phần SQuAD v1.1. Nhóm sử dụng bản tiếng Việt từ kho chính thức [7, 8], cố định tại commit `7d30520c717524000f0d9d2f9c10a069acd9d285`. Bản chuyển đổi trong dự án giữ giấy phép CC BY-SA 4.0 và thông tin nguồn.

| Cách tổ chức | Tập | Số văn bản | Số câu hỏi | Số bài nguồn |
|---|---|---:|---:|---:|
| Context gốc (`paragraph`) | Dev | 50 | 233 | 10 |
| Context gốc (`paragraph`) | Test | 190 | 957 | 38 |
| Ghép theo bài (`article`) | Dev | 10 | 233 | 10 |
| Ghép theo bài (`article`) | Test | 38 | 957 | 38 |

Toàn bộ có 240 context, 1.190 câu hỏi và 48 bài nguồn. Dev/test được chia theo title với seed `utc-rag-xquad-vi-v1`; mọi context và câu hỏi của cùng bài thuộc cùng tập. Đây là cách chia cục bộ của dự án, không phải split chính thức của XQuAD. Các chunker được giữ nguyên khi chạy bộ dữ liệu này.

Vai trò dev và test: các tham số của hai chiến lược được giữ cố định khi chạy XQuAD test, theo README và cấu hình mã nguồn hiện có. Dev gồm 233 câu hỏi được tách riêng để phục vụ phát triển và kiểm tra pipeline; không có artifact ghi nhận một đợt tìm tham số trên XQuAD dev, nên báo cáo không tuyên bố đã tune bằng dev. Test gồm 957 câu hỏi không được dùng để chọn lại threshold/weight trong lần rà soát này. Phân tích bổ sung là hậu nghiệm sau khi đã xem kết quả test, không phải xác nhận độc lập. Hash mã nguồn hiện tại khớp các hash lưu ở cả hai lần chạy XQuAD.

Ở góc nhìn `paragraph`, mỗi context là một tài liệu; độ dài trên toàn bộ dữ liệu nằm trong khoảng 39–763 đơn vị khoảng trắng, trung vị 155. Ở góc nhìn `article`, năm context cùng title được ghép theo thứ tự nguồn; độ dài 496–2.169, trung vị 804,5. Văn bản ghép không phải bài Wikipedia đầy đủ và không bảo đảm các trích đoạn vốn nằm liền nhau.

Hai góc nhìn sử dụng cùng câu hỏi và nội dung, nên không được cộng thành hai bộ bằng chứng độc lập. Bộ pilot đã dùng phát triển thuật toán cũng không được trộn vào kết quả test chính.

`gold_evidence` được suy ra bằng cách lấy bao câu chứa đáp án đầu tiên của nguồn. Nhãn này chưa được con người rà soát đầy đủ; một câu chứa đáp án chưa chắc chứa mọi thông tin cần để suy luận. Dữ liệu có chín nhóm câu hỏi trùng trên toàn bộ tập và một số nhãn nguồn đáng nghi được ghi trong hồ sơ kiểm tra. Đây là benchmark **chuyển đổi cho retrieval**, không phải giao thức chấm EM/F1 chính thức của XQuAD.

### 5.3. Môi trường chạy

Các phiên bản dưới đây lấy từ trường `provenance` của artifact XQuAD, không suy ra từ phiên bản phần mềm mới nhất:

| Thành phần | Giá trị ghi nhận |
|---|---|
| Python | 3.13.5 |
| sentence-transformers | 6.1.0 |
| PyTorch | 2.14.1 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Retriever | Ma trận vector chuẩn hóa, cosine bằng NumPy |
| top-k | 5; báo Evidence Recall@3, Evidence Recall@5 và MRR@5 |
| Xử lý bằng điểm | Thứ tự corpus ổn định |
| LLM | `null`, không chạy generation |

Workspace hiện được thao tác trên Windows/PowerShell. Artifact của lần chạy không ghi đầy đủ hệ điều hành, CPU, RAM, GPU, thiết bị encode và revision của embedding model; do đó chưa đủ hồ sơ để tái lập chính xác môi trường phần cứng. Không sử dụng cấu hình máy hiện tại để suy diễn ngược về lần chạy trước.

Thời gian trong CSV là một lượt chạy sau khởi động mô hình, theo thứ tự Max–Min rồi Ours. Nó chưa phải benchmark tốc độ nhiều lần có kiểm soát tải máy. Vì vậy, báo cáo không kết luận ưu thế tốc độ thống kê từ các số thời gian này.

### 5.4. Kịch bản thử nghiệm và cách so sánh với bài báo

**Kịch bản A — context gốc:** lập chỉ mục 190 tài liệu test và truy xuất cho 957 câu hỏi. Thiết lập này đánh giá khi dữ liệu đã được nguồn chia thành context tương đối ngắn.

**Kịch bản B — văn bản ghép dài:** lập chỉ mục 38 tài liệu ghép, vẫn dùng 957 câu hỏi đó. Thiết lập này kiểm tra độ nhạy trước việc thay đổi đơn vị tài liệu và tăng độ dài đầu vào.

**Quy trình kiểm chứng sau truy xuất.** Hai kịch bản đều áp dụng cùng quy trình dưới đây cho từng câu hỏi và từng chiến lược chunking. Đối tượng được chấm là các chunk truy xuất, chưa phải câu trả lời sinh bởi LLM.

**Bước 1 — Chuẩn bị nhãn chuẩn.** Mỗi câu hỏi gồm nội dung câu hỏi, đáp án nguồn, ID tài liệu và gold_evidence. Bằng chứng chuẩn được suy ra từ bao câu chứa đáp án đầu tiên của nguồn. Đáp án và evidence chỉ phục vụ chuẩn bị nhãn và chấm kết quả, không được dùng để chọn chunk khi truy xuất. Nhãn evidence được trích tự động, chưa được rà soát đầy đủ bởi con người.

**Bước 2 — Phân đoạn và lập chỉ mục.** Chạy Max–Min và Ours riêng biệt trên cùng tập tài liệu. Mỗi chunk được gắn ID tài liệu nguồn và chuyển thành vector bằng cùng mô hình multilingual MiniLM. Với kịch bản B, ID nguồn dùng để chấm là ID tài liệu ghép đã được ánh xạ trong dữ liệu benchmark.

**Bước 3 — Truy xuất theo câu hỏi.** Encode câu hỏi và tính cosine similarity giữa vector câu hỏi với vector của mọi chunk trong corpus của từng phương pháp. Sắp xếp giảm dần và lấy top-5; khi bằng điểm, giữ thứ tự corpus ổn định. Hệ thống không dùng ID tài liệu chuẩn để giới hạn phạm vi tìm kiếm. Cosine ở bước này dùng để xếp hạng mức liên quan, không phải tiêu chí xác nhận đáp án đúng.

**Bước 4 — Đối chiếu với bằng chứng chuẩn.** Một chunk hợp lệ phải đồng thời thuộc đúng tài liệu đã gán nhãn và chứa toàn bộ chuỗi gold_evidence. Trước khi kiểm tra chuỗi con, văn bản được chuẩn hóa Unicode NFC, casefold và khoảng trắng. Một chunk chỉ chứa đáp án ngắn nhưng không chứa đủ evidence vẫn không đạt. Hai chunk cộng lại đủ evidence cũng không được tính là hit theo giao thức hiện tại. Đây là phép chấm đúng nguồn và độ bao phủ evidence, không phải đánh giá tương đương ngữ nghĩa của câu trả lời.

**Bước 5 — Tính điểm từng câu hỏi.** Gọi r là thứ hạng của chunk hợp lệ đầu tiên trong top-5. Hit@3 bằng 1 nếu r ≤ 3; Hit@5 bằng 1 nếu r ≤ 5; reciprocal rank@5 bằng 1/r. Nếu không có chunk hợp lệ trong top-5, cả ba điểm đều bằng 0. Trong benchmark này, Evidence Recall@k là trung bình Hit@k trên các câu hỏi; MRR@5 là trung bình reciprocal rank@5.

**Ví dụ minh họa cách chấm.** Ví dụ giả định, không phải mẫu thực nghiệm: câu hỏi “Trường A được thành lập năm nào?”, đáp án “1960”, evidence “Trường A được thành lập vào năm 1960.” Nếu Max–Min có chunk đúng tài liệu chứa đủ evidence ở hạng 4, điểm Hit@3, Hit@5 và reciprocal rank@5 lần lượt là 0; 1; 0,25. Nếu Ours đưa chunk hợp lệ lên hạng 2, ba điểm là 1; 1; 0,50. Hai bên cùng hit top-5, nhưng Ours có thứ hạng tốt hơn ở ví dụ này. Chunk ở tài liệu khác chỉ có thông tin tương tự không được tính là hit.

**Bước 6 — Tổng hợp và so sánh ghép cặp.** Lặp quy trình trên cùng 957 câu hỏi cho cả hai phương pháp. Ở context gốc, Max–Min có 832 câu hit top-5, tương ứng 832/957 = 86,94%; Ours có 879 câu, tương ứng 91,85%. Chênh lệch là 47 câu, hay 4,91 điểm phần trăm. Lưu kết quả từng câu hỏi để xác định câu nào chỉ một bên hit và tính khoảng tin cậy ghép cặp theo bài nguồn như mô tả bên dưới. Bảng kết quả do đó đo khả năng truy xuất bằng chứng trong cấu hình đã chạy, không chứng minh độ chính xác trả lời của LLM.

**Bước 7 — Phân biệt lỗi và chi phí.** Nếu không có chunk nào trong toàn bộ chỉ mục chứa đủ evidence ở đúng nguồn, ghi evidence_split. Nếu chunk hợp lệ tồn tại nhưng không lên top-5, ghi ranking_miss. Tỷ lệ bảo toàn evidence kiểm tra sự tồn tại của chunk hợp lệ ở bất kỳ thứ hạng nào, còn lượng context đo tổng số đơn vị khoảng trắng trong top-5, tính cả overlap. Phân tích này giúp phân biệt mất bằng chứng do phân đoạn với thất bại xếp hạng, đồng thời kiểm tra mức cải thiện có đi kèm nhiều context hơn hay không.

**Khả năng kiểm tra lại từng kết quả.** Các tệp rag_comparison.json của hai kịch bản lưu details cho từng câu hỏi, gồm question_id, document_id, gold_evidence, hit_at_3, hit_at_5, reciprocal_rank_at_5, evidence_preserved, failure_reason và retrieved. Danh sách retrieved chứa các chunk trả về cùng thứ hạng và điểm cosine. Người kiểm tra có thể đối chiếu lại evidence với từng chunk rồi tính lại các chỉ số. Quy tắc chấm được hiện thực trong evaluation/evaluate_retrieval.py; các đường dẫn artifact được dẫn ở tài liệu tham khảo [9–10].

**Phân biệt với đánh giá câu trả lời đầu cuối.** Đánh giá đầu cuối cần thêm bước đưa context truy xuất vào cùng một LLM, giữ chung prompt và cấu hình sinh, rồi chấm câu trả lời với đáp án chuẩn bằng EM/F1 hoặc rubric đúng–sai được xác định trước. Cosine giữa embedding câu trả lời và đáp án chuẩn có thể là chỉ số bổ sung, nhưng không nên dùng riêng để kết luận đúng sai: hai câu khác năm hoặc khác phủ định vẫn có thể gần nhau về ngữ nghĩa. Thực nghiệm hiện tại chưa thực hiện bước sinh và chấm câu trả lời, vì vậy chưa có các điểm QA này.

**Nguyên tắc đối chứng:** chạy lại Max–Min và Ours trên cùng dữ liệu, giữ nguyên embedding, bộ tách câu, phép xếp hạng, top-k, nhãn và metric. Kết quả vì vậy trả lời câu hỏi “hai chiến lược hoạt động thế nào trong pipeline tiếng Việt này”. Nó không trực tiếp xác nhận hay bác bỏ các con số AMI hoặc QA Accuracy trong bài báo gốc.

| Vấn đề khi so sánh | Cách xử lý |
|---|---|
| Bài báo và đề tài dùng dữ liệu khác nhau | Không so điểm công bố trực tiếp; chạy baseline trên XQuAD tiếng Việt |
| Thay embedding có thể thay hành vi baseline | Công khai đây là bản thích nghi, dùng cùng embedding cho hai bên |
| Chunker tạo kích thước khác nhau | Báo số chunk, độ dài chunk và lượng context cùng Recall/MRR |
| Cùng top-k chưa cùng ngân sách context | Xem đây là so sánh cấu hình; cần thêm thí nghiệm cùng ngân sách token |
| Nhiều câu hỏi cùng bài có tương quan | Bootstrap paired theo bài nguồn |
| Muốn xác định phần đóng góp của overlap hoặc luật | Cần ablation; không suy nguyên nhân từ hai cấu hình đầy đủ |

Khoảng tin cậy được tính bằng bootstrap ghép cặp theo 38 bài nguồn, 5.000 lần, seed 2026, lấy percentile 95% của chênh lệch Ours trừ Max–Min. Đây là phân tích khám phá trên tập hiện có, không thay thế việc đánh giá trên miền dữ liệu mới.

Phân tích hậu nghiệm tại mục 5.7 áp cùng trần 512/1024 tokenizer token lên top-5 đã lưu, giữ nguyên thứ hạng và chunk. Phân tích này không thay thế thí nghiệm fixed-budget trên toàn bộ danh sách xếp hạng: còn giới hạn tối đa năm ứng viên, có thể để dư ngân sách và chưa có quy tắc phân bổ context được đăng ký trước. Các cấu hình bổ sung không được dùng để chọn lại tham số hoặc sửa kết quả test gốc.

### 5.5. Kết quả thực nghiệm

Các bảng chính dưới đây giữ nguyên số liệu từ CSV/JSON của lần chạy ngày 30/10/2026 [9, 10]. Nhóm đã tính lại chỉ số từ các kết quả từng câu hỏi để kiểm tra tính nhất quán. Các phân tích bổ sung ở mục 5.7–5.9 được tính riêng từ artifact đã lưu [11], không chạy lại chunking hoặc thay dữ liệu test.

| Đầu vào | Chiến lược | Số chunk | Evidence Recall@3 | Evidence Recall@5 | MRR@5 |
|---|---|---:|---:|---:|---:|
| Context gốc | Max–Min | 583 | 81,61% | 86,94% (832/957) | 0,7421 |
| Context gốc | Ours | 226 | 87,98% | 91,85% (879/957) | 0,8062 |
| Văn bản ghép dài | Max–Min | 556 | 80,56% | 85,27% (816/957) | 0,7259 |
| Văn bản ghép dài | Ours | 191 | 73,67% | 80,56% (771/957) | 0,6543 |

| Đầu vào | Chênh lệch Evidence Recall@5, Ours − Max–Min | Khoảng tin cậy bootstrap 95% |
|---|---:|---:|
| Context gốc | +4,91 điểm phần trăm | [+2,49; +7,45] |
| Văn bản ghép dài | −4,70 điểm phần trăm | [−8,24; −1,32] |

Cả hai khoảng không chứa 0 trong phân tích hiện tại, nhưng chiều chênh lệch đổi theo cách tổ chức tài liệu. Vì vậy, kết quả không hỗ trợ kết luận Ours luôn tốt hơn Max–Min.

| Đầu vào | Chiến lược | Độ dài chunk trung bình | Độ dài context top-5 trung bình | Bảo toàn evidence |
|---|---|---:|---:|---:|
| Context gốc | Max–Min | 56,90 | 361,20 | 98,85% |
| Context gốc | Ours | 153,57 | 783,85 | 100% |
| Văn bản ghép dài | Max–Min | 59,66 | 386,55 | 98,85% |
| Văn bản ghép dài | Ours | 194,08 | 977,40 | 100% |

Đơn vị độ dài là khoảng trắng, có tính nội dung lặp. Context top-5 của Ours lớn hơn khoảng 2,17 lần ở context gốc và 2,53 lần ở văn bản ghép dài. Do vậy, tăng Recall trong kịch bản A chưa đồng nghĩa với hiệu quả cao hơn trên cùng ngân sách context.

### 5.6. Đánh giá: tốt, chưa tốt và nguyên nhân

Trong cấu hình top-5 chunk hiện tại, Ours có Evidence Recall@5 cao hơn trên context gốc, nhưng đồng thời sử dụng lượng context lớn hơn đáng kể. Có 81 câu chỉ Ours hit và 34 câu chỉ Max–Min hit, chênh lệch ròng 47 câu. MRR@5 cũng cao hơn. Nhiều context vốn ngắn khiến Ours có thể giữ phần lớn hoặc toàn bộ context; chưa thể quy mức tăng Recall hoàn toàn cho chất lượng xác định boundary. Kết quả dưới trần token ở mục 5.7 cho thấy cần xét cả ngân sách context khi so sánh.

**Trên văn bản ghép dài, Ours kém hơn về cả Recall và MRR.** Có 127 câu chỉ Max–Min hit và 82 câu chỉ Ours hit, chênh lệch ròng là 45 câu có lợi cho Max–Min. Ours tạo ít chunk hơn nhưng mỗi chunk lớn hơn, trong khi lượng context truy xuất cao hơn đáng kể.

Phân loại thất bại giúp xác định nơi xảy ra vấn đề:

| Đầu vào | Chiến lược | Evidence không nằm trọn trong chunk nào | Có chunk đủ evidence nhưng trượt top-5 |
|---|---|---:|---:|
| Context gốc | Max–Min | 11 | 114 |
| Context gốc | Ours | 0 | 78 |
| Văn bản ghép dài | Max–Min | 11 | 130 |
| Văn bản ghép dài | Ours | 0 | 186 |

Theo tiêu chí evidence hiện tại, Ours giữ được bằng chứng trong cả hai kịch bản. Với văn bản dài, cả 186 câu miss của Ours thuộc nhóm xếp hạng: bằng chứng tồn tại trong chunk nhưng chunk đó không vào top-5. Đây là điều quan sát được trực tiếp, cho thấy bảo toàn bằng chứng và truy xuất đúng bằng chứng là hai mục tiêu khác nhau.

Giả thuyết giải thích: chunk lớn có thể làm vector tổng hợp ít tập trung vào chi tiết câu hỏi; overlap có thể làm thay đổi nội dung vector hoặc tạo lặp trong context. Phân tích tokenizer tại mục 5.8 đo được nhiều chunk Ours vượt giới hạn 128 token của cấu hình SentenceTransformer đang lưu. Truncation là một cơ chế có thể góp phần giải thích kết quả retrieval trên chunk dài, nhưng chưa được chứng minh là nguyên nhân của chênh lệch giữa hai phương pháp. Chưa đo vị trí evidence trong vùng encode và chưa có đối chứng loại trừ truncation.

Hướng thực nghiệm tiếp theo: fixed context-token budget trên toàn bộ ranking; phân tích offset evidence so với vùng tokenizer được encode; dependency challenge set có văn bản trên 240 đơn vị; ablation bỏ overlap, bỏ penalty và chỉ dùng độ dài; cuối cùng là đánh giá câu trả lời bằng cùng LLM, prompt và rubric/EM/F1. Các thử nghiệm cần được định nghĩa trước, dùng dev để chọn tham số và xác nhận trên dữ liệu độc lập mới sau khi đã xem test hiện tại.

**Giới hạn kết luận:** evidence được suy ra tự động, có câu hỏi trùng và nhãn đáng nghi; chỉ có 38 nhóm bài nguồn test; dữ liệu ghép không phải văn bản dài tự nhiên hoàn chỉnh; benchmark không có nhãn ranh giới đầy đủ và không có câu hỏi unanswerable. Chưa chạy generation, nên không thể báo QA Accuracy, EM/F1 câu trả lời, mức giảm hallucination hoặc chất lượng trả lời của Qwen từ các kết quả này.

### 5.7. Phân tích trần token trên top-5 đã lưu

Thiết kế hậu nghiệm: dùng tokenizer multilingual MiniLM ở snapshot cục bộ e8f8c211226b894fcb81acc59f3b34ba3efd5f42. Với mỗi câu hỏi, duyệt top-5 đã lưu theo thứ hạng gốc, nối nguyên chunk bằng hai ký tự xuống dòng và dừng trước chunk đầu tiên làm vượt trần 512 hoặc 1024 token. Không cắt chunk, không bỏ qua chunk quá dài để lấy chunk sau, không bổ sung ứng viên ngoài top-5 và không dùng evidence để chọn context. Token budget chỉ tính nội dung context, không gồm special token, câu hỏi hoặc prompt; đây là tokenizer của embedding model, chưa phải ngân sách của một LLM cụ thể.

Evidence Recall@budget là tỷ lệ câu hỏi có chunk hợp lệ trong phần context được giữ; MRR@budget dùng nghịch đảo hạng của chunk hợp lệ đầu tiên, bằng 0 nếu không có. Vì chỉ giữ tiền tố của top-5, thứ hạng các chunk được chọn không đổi. Cùng trần không có nghĩa cùng số token thực dùng; giới hạn top-5 có thể ngăn Max–Min sử dụng hết ngân sách. Kết quả là chẩn đoán “top-5 có trần token”, không phải phép so sánh fixed-budget đầy đủ trên toàn ranking. Không tính khoảng tin cậy mới và không chọn mức budget theo kết quả thắng/thua.

| Đầu vào | Chiến lược | Trần token | Evidence Recall@budget | MRR@budget | Token thực dùng TB |
| --- | --- | --- | --- | --- | --- |
| Context gốc | Max–Min | 512 | 85,58% | 0,7391 | 368,26 |
| Context gốc | Max–Min | 1024 | 86,94% | 0,7421 | 432,84 |
| Context gốc | Ours | 512 | 85,16% | 0,7875 | 414,14 |
| Context gốc | Ours | 1024 | 91,22% | 0,8050 | 879,26 |
| Ghép dài | Max–Min | 512 | 83,59% | 0,7218 | 379,50 |
| Ghép dài | Max–Min | 1024 | 85,27% | 0,7259 | 463,17 |
| Ghép dài | Ours | 512 | 63,64% | 0,5975 | 395,00 |
| Ghép dài | Ours | 1024 | 78,16% | 0,6491 | 901,28 |

Ở trần 512 token, context gốc cho 819/957 hit với Max–Min và 815/957 với Ours; chênh lệch nhỏ này chưa được kiểm định thống kê. Trên văn bản ghép dài, số hit tương ứng là 800 và 609. Ở trần 1024 token, context gốc cho 832 và 873 hit; văn bản ghép dài cho 816 và 748 hit. MRR trên context gốc vẫn cao hơn ở Ours trong hai mức trần, nên không đồng nhất kết luận Recall với mọi khía cạnh xếp hạng. Đây là bằng chứng về độ nhạy với chính sách context trong tập đã quan sát, không chứng minh cơ chế nhân quả hoặc ưu thế tổng quát [11].

### 5.8. Phân tích độ dài tokenizer và nguy cơ truncation

Đo độ dài tất cả chunk đã lưu bằng tokenizer snapshot nêu ở mục 5.7, không truncation và có special tokens. Trung bình, trung vị, cực đại và tỷ lệ vượt 128 được tính theo chunk, không theo câu hỏi. sentence_bert_config.json của snapshot ghi max_seq_length=128; tokenizer có thể có model_max_length khác, nên không dùng giới hạn tokenizer chung để thay cho cấu hình SentenceTransformer. Artifact retrieval cũ không cố định revision tokenizer: các số sau mô tả độ dài dưới snapshot cục bộ được ghi hash trong [11].

| Đầu vào | Chiến lược | TB token | Trung vị | Lớn nhất | Chunk >128 | Tỷ lệ |
| --- | --- | --- | --- | --- | --- | --- |
| Context gốc | Max–Min | 70,28 | 54 | 493 | 76/583 | 13,04% |
| Context gốc | Ours | 186,03 | 187 | 333 | 196/226 | 86,73% |
| Ghép dài | Max–Min | 73,60 | 57 | 493 | 80/556 | 14,39% |
| Ghép dài | Ours | 234,26 | 233 | 414 | 187/191 | 97,91% |

Kết quả đo cho thấy tỷ lệ chunk Ours vượt 128 token cao hơn Max–Min ở cả hai cách tổ chức đầu vào. Nếu encode theo cấu hình SentenceTransformer 128 token, các chuỗi vượt giới hạn sẽ bị cắt trước khi biểu diễn. Tuy nhiên, độ dài vượt giới hạn không tự cho biết evidence bị mất: chưa có ánh xạ offset bằng chứng sang từng vùng token, đặc biệt khi có overlap hoặc nhiều lần xuất hiện. Vì vậy, không báo tỷ lệ evidence bị truncation và không kết luận truncation chắc chắn gây ra mức giảm Recall. Cần kiểm tra offset và chạy đối chứng có kiểm soát để xác định tác động [11].

### 5.9. Chẩn đoán vi phạm ranh giới dependency

Targeted diagnostic evaluation sử dụng annotation có sẵn trong data/benchmark/annotations.json và chunk của bộ pilot đã lưu tại evaluation/results/rag_comparison.json. Có ba ranh giới dương tính: case_001 tại vị trí 0, case_007 và case_011 tại vị trí 4; vị trí được đánh số từ 0. case_012 là tài liệu đối chứng ngoại lệ với danh sách dependency rỗng, không tăng mẫu số. Không tự sinh nhãn dependency từ XQuAD. Metric là số ranh giới dependency bị chọn làm điểm cắt chia cho tổng ba ranh giới được annotate; nó đo quyết định ranh giới, không đo mọi quan hệ ngữ nghĩa hay khả năng overlap sửa mất ngữ cảnh.

| Chiến lược | Ranh giới vi phạm | Ranh giới đã gán nhãn | Dependency Violation Rate |
| --- | --- | --- | --- |
| Max–Min | 2 | 3 | 66,67% |
| Ours | 0 | 3 | 0,00% |

Max–Min cắt tại hai ranh giới được gán nhãn của case_007 và case_011; Ours không cắt tại ba ranh giới dương tính. Tuy nhiên, độ dài toàn văn của case_001, case_007, case_011 và case_012 lần lượt chỉ là 16, 228, 205 và 216 đơn vị khoảng trắng. Do đó không testcase nào kích hoạt nhánh trên 240 có dependency penalty. Kết quả 0/3 của Ours phản ánh hành vi của chiến lược đầy đủ trên các mẫu nhỏ, không chứng minh tác dụng riêng của penalty. Bộ pilot đã được dùng phát triển, không độc lập và quá nhỏ để suy rộng; cần thêm văn bản trên MAX, annotation đáng tin cậy và ablation penalty [11].

### 5.10. Tổng hợp đánh đổi và khả năng tái lập

| Tiêu chí trong hai thiết lập đã đo | Max–Min | Ours |
| --- | --- | --- |
| Bảo toàn evidence | 98,85% ở cả hai | 100% ở cả hai |
| Độ dài chunk và context top-5 | Nhỏ hơn | Lớn hơn |
| Evidence Recall@5 trên context gốc | 86,94% | 91,85% |
| Evidence Recall@5 trên văn bản ghép dài | 85,27% | 80,56% |
| Luật liên kết tiếng Việt | Không thêm luật riêng | Penalty chỉ ở nhánh trên MAX |
| Overlap động | Không thêm | Giữ 0–2 câu |

Bảng tổng hợp mô tả hai cấu hình đã quan sát, không xếp hạng phương pháp cho mọi dữ liệu. Evidence preservation và retrieval ranking là hai mục tiêu khác nhau: Ours có evidence_split = 0 nhưng vẫn có 78 ranking_miss ở context gốc và 186 ở văn bản ghép dài. Giả thuyết vector của chunk nhỏ tập trung hơn cần thí nghiệm nhân quả, không được suy trực tiếp từ bảng.

Tái lập phân tích: chạy .\venv\Scripts\python.exe -X utf8 scripts/audit_report_evidence.py từ thư mục gốc. Script chỉ đọc chunk, ranking và nhãn đã lưu, không gọi chunker hoặc LLM. Kết quả nằm tại evaluation/results/report_audit/: audit.json lưu cấu hình, snapshot/hash tokenizer, phiên bản phần mềm, checksum đầu vào và mã nguồn; các CSV lưu thống kê tokenizer, độ dài từng chunk, kết quả theo budget và từng câu hỏi. Không có lấy mẫu ngẫu nhiên nên seed không áp dụng; seed 2026 và 5.000 lượt chỉ thuộc bootstrap gốc.

Các metric gốc được tính lại từ details và đối chiếu với JSON/CSV: số tài liệu, câu hỏi, chunk, Evidence Recall@3/@5, MRR@5, độ dài chunk/context và bảo toàn evidence. Việc bổ sung phân tích không thay threshold, weight, implementation Max–Min, nhãn test hoặc các bảng kết quả gốc. Các phân tích này chưa cung cấp evidence truncation theo offset, full-ranking fixed-budget, ablation hoặc QA generation/EM/F1.

## 6. Kết luận

Đề tài đã xây dựng một chiến lược chunking tiếng Việt kết hợp độ dài, suy giảm tương đồng ngữ nghĩa, luật liên kết câu và overlap động; đồng thời tích hợp Max–Min làm baseline có nguồn học thuật và mã nguồn đối chiếu. Hai chiến lược được đánh giá trong cùng pipeline retrieval trên bản chuyển đổi XQuAD tiếng Việt.

Trên 957 câu hỏi test, Ours đạt Evidence Recall@5 91,85% khi dùng context gốc, cao hơn Max–Min 4,91 điểm phần trăm. Khi ghép thành văn bản dài, Ours đạt 80,56%, thấp hơn Max–Min 4,70 điểm phần trăm. Ours bảo toàn evidence theo nhãn hiện tại nhưng sử dụng lượng context top-5 lớn hơn và có nhiều lỗi xếp hạng ở kịch bản văn bản dài.

Ours chưa chứng minh ưu thế tổng quát so với Max–Min. Lợi thế về bảo toàn evidence và retrieval trên context gốc trong cấu hình top-5 đi kèm lượng context lớn hơn, trong khi lợi thế retrieval đảo chiều trên văn bản ghép dài. Phân tích trần token và tokenizer bổ sung cho thấy kết luận phụ thuộc chính sách context; chưa xác lập quan hệ nhân quả giữa kích thước chunk, truncation và thứ hạng. Hướng tiếp theo là fixed-budget trên toàn ranking, evidence truncation theo offset, dependency diagnostic ở nhánh trên MAX, ablation và đánh giá sinh câu trả lời đầu cuối.

## 7. Tài liệu tham khảo

1. Lewis, P., và cộng sự. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. [Bài báo](https://arxiv.org/abs/2005.11401).
2. Kiss, C., Nagy, M., & Szilágyi, P. (2025). *Max–Min semantic chunking of documents for RAG application*. Discover Computing, 28, 117. [DOI: 10.1007/s10791-025-09638-7](https://doi.org/10.1007/s10791-025-09638-7).
3. HSDSLab. *MaxMinChunking — mã nguồn tác giả*. Phiên bản đối chiếu: `0caba41d2ef30114729f95e5c788173e4f40920e`. [maxmin_chunker.py](https://github.com/hsdslab/MaxMinChunking/blob/0caba41d2ef30114729f95e5c788173e4f40920e/maxmin_chunker.py).
4. LlamaIndex. *Semantic Chunker*. Tài liệu kỹ thuật, truy cập ngày 30/10/2026. [Tài liệu chính thức](https://developers.llamaindex.ai/python/examples/node_parsers/semantic_chunking/).
5. Reimers, N., & Gurevych, I. (2020). *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*. EMNLP 2020, tr. 4512–4525. [ACL Anthology](https://aclanthology.org/2020.emnlp-main.365/).
6. Sentence Transformers. *paraphrase-multilingual-MiniLM-L12-v2*. Model card, truy cập ngày 30/10/2026. [Mô hình và cấu hình](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
7. Artetxe, M., Ruder, S., & Yogatama, D. (2020). *On the Cross-lingual Transferability of Monolingual Representations*. ACL 2020, tr. 4623–4637. [ACL Anthology](https://aclanthology.org/2020.acl-main.421/).
8. Google DeepMind. *XQuAD — Cross-lingual Question Answering Dataset*. [Kho dữ liệu chính thức](https://github.com/google-deepmind/xquad). Bản chuyển đổi của dự án: [README và quy trình](data/benchmark/xquad_vi/README.md), [manifest](data/benchmark/xquad_vi/manifest.json).
9. Dự án utc_rag. *Kết quả XQuAD tiếng Việt — context gốc, tập test*. [CSV](evaluation/results/xquad_vi_paragraph_test/rag_comparison.csv), [JSON chi tiết](evaluation/results/xquad_vi_paragraph_test/rag_comparison.json), [thống kê](evaluation/results/xquad_vi_paragraph_test/rag_statistical_summary.json).
10. Dự án utc_rag. *Kết quả XQuAD tiếng Việt — văn bản ghép dài, tập test*. [CSV](evaluation/results/xquad_vi_article_test/rag_comparison.csv), [JSON chi tiết](evaluation/results/xquad_vi_article_test/rag_comparison.json), [thống kê](evaluation/results/xquad_vi_article_test/rag_statistical_summary.json).

11. Dự án utc_rag. Phân tích hậu nghiệm phục vụ rà soát báo cáo (30/10/2026). Script: scripts/audit_report_evidence.py. Artifact và hướng dẫn: evaluation/results/report_audit/README.md; dữ liệu kết quả: evaluation/results/report_audit/audit.json và các CSV cùng thư mục.

Các mục [9–10] là artifact thực nghiệm nội bộ, không phải bài báo đã công bố. Mô tả triển khai được đối chiếu với [pipeline](PIPELINE.md), [baseline Max–Min](MAXMIN.md) và mã nguồn hiện tại của dự án.
