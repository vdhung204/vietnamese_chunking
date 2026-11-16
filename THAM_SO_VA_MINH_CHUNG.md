# Giải trình tham số và minh chứng thực nghiệm

Ngày bổ sung: 16/11/2026. Đây là nghiên cứu bổ sung sau khi đã có kết quả test ngày 30/10/2026.

**Code hiện tại đã chốt trọng số semantic/size/risk = 0,5/0,3/0,6.** Đây là bộ được chọn theo Recall@5 dev cao nhất trong nhóm trọng số; penalty giữ 0,6 vì các mức đã thử đồng hạng. Các tham số khác giữ nguyên. Trong các bảng lịch sử của phụ lục này, “mặc định” hoặc “cấu hình ban đầu” là mốc đối chiếu 0,4/0,4/0,6 của nghiên cứu đã lưu, không phải trọng số production hiện tại. Các số test cũ vẫn thuộc cấu hình ban đầu. Kiểm tra code hiện tại bằng `scripts/verify_selected_weights.py`; kết quả lưu ở `evaluation/results/parameter_study/active_configuration.json`.

Case pilot 010 thay đổi điểm cắt khi dùng trọng số mới, nên overlap thành một câu thay vì hai câu theo kỳ vọng lịch sử. Nhãn gốc được giữ để đối chiếu. Đây là thay đổi hành vi cần công bố; kiểm tra riêng xác nhận overlap vẫn tuân theo cosine của ranh giới mới, không điều chỉnh ngưỡng để ép khớp kết quả cũ.

## 1. Kết luận có thể bảo vệ

Đã chạy thật 180 cấu hình Ours trên XQuAD tiếng Việt dev, kiểm tra lại 360 kết quả cấu hình/góc nhìn từ vector đã lưu, và tính lại các bảng thực nghiệm gốc. Kết quả không chứng minh mọi hằng số hiện tại là tối ưu. Một số cấu hình khác cho điểm dev cao hơn; nhiều mức risk đồng hạng vì dữ liệu ít kích hoạt luật.

Vì vậy không nên viết “thực nghiệm chứng minh 0,4/0,4/0,6 và mọi threshold đều tốt nhất”. Phát biểu có bằng chứng là: “Cấu hình ban đầu là lựa chọn thiết kế; sau nghiên cứu độ nhạy, nhóm chọn tỷ lệ semantic/size = 0,5/0,3 theo Recall@5 dev trong lưới trọng số và giữ penalty 0,6 trong nhóm đồng hạng”. Không dựng lại lịch sử rằng các giá trị ban đầu được chọn bằng grid search: trước nghiên cứu này, repo chưa có artifact của quá trình đó.

Số liệu đo được cần tính lại; siêu tham số cần so sánh thực nghiệm; tham số kế thừa cần dẫn nguồn; định nghĩa toán học và cấu hình vận hành cần giải thích vai trò. Không phải loại số nào cũng có ý nghĩa “tối ưu”.

## 2. Giao thức trước khi xem kết quả dev mới

- Tập phát triển: 233 câu hỏi, 10 bài nguồn; paragraph gồm 50 tài liệu và article gồm 10 tài liệu ghép. Hai góc nhìn dùng cùng câu hỏi, không phải 466 mẫu độc lập.
- Tiêu chí chính: trung bình bằng trọng số giữa Evidence Recall@5 của hai góc nhìn. Tiêu chí phụ: trung bình MRR@5. Lưu toàn bộ cấu hình đồng hạng; không suy ra ưu thế về chi phí chỉ từ hai metric này.
- 54 bộ trọng số, các dải MIN/TARGET/MAX, 10 mức ngưỡng drop kể cả chế độ tắt, 15 cặp ngưỡng overlap, số câu overlap, độ chính xác làm tròn, ablation và lưới kết hợp định trước 81 cấu hình. Sau loại cấu hình trùng còn 180 cấu hình riêng biệt.
- Giữ nguyên MiniLM, cosine, quy tắc evidence, bộ tách câu, nội dung, top-5 và thuật toán nền. Chỉ thay các hằng số trong module riêng của tiến trình; mã nguồn production không đổi. Bộ adapter mặc định đã đối chiếu bằng toàn bộ 60 tài liệu dev với implementation gốc.
- Đọc model trong cache, CPU, 4 thread, seed 2026. Embedding câu được tính thật bằng model rồi cache; embedding chunk dùng chung theo chuỗi nội dung giống nhau. Không có vector giả hoặc request LLM. Snapshot và hash cả model/tokenizer được ghi trong summary.json.
- protocol.json được ghi trước các lệnh load model/chạy/chấm điểm. Đây là giao thức đóng băng cục bộ, không phải đăng ký nghiên cứu công khai.
- Không dùng metric test để xếp hạng cấu hình mới và không thay kết quả test gốc. Do đã biết kết quả test trước khi thiết kế nghiên cứu, đây vẫn là phân tích bổ sung mang tính khám phá. Cần dữ liệu độc lập mới để xác nhận một cấu hình mới.

Việc tách dữ liệu chọn siêu tham số khỏi dữ liệu đánh giá phù hợp với [hướng dẫn model selection của scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html). Lưới hữu hạn chỉ xác định cấu hình tốt nhất trong các lựa chọn đã thử, không chứng minh cực trị toàn cục. Điểm của cấu hình thắng trên chính dev có thiên lệch lựa chọn.

## 3. Ý nghĩa toán học của các số trong công thức

Với A là size score, B là semantic score và R là cờ risk:

S = 0,4A + 0,4B − 0,6R.

A và B nằm trong [0,1]; R thuộc {0,1}. Hai số 0,4 thể hiện ưu tiên bằng nhau cho hai tín hiệu đã chuẩn hóa. Số 0,6 phạt ranh giới có dấu hiệu liên kết. Đây là utility score, không phải xác suất; không có yêu cầu các hệ số phải cộng thành 1.

Nhân tất cả hệ số với cùng một số dương không đổi argmax trong số học chính xác. Vì vậy 0,4/0,4/0,6 tương đương 1/1/1,5 hoặc 0,5/0,5/0,75. Cần đánh giá tỷ lệ, không thể xác định duy nhất độ lớn tuyệt đối của ba hệ số bằng thứ tự điểm cắt. Lưới trọng số giữ semantic + size = 0,8 để loại bớt các bộ tương đương do co giãn.

Điểm của ứng viên không risk nằm trong [0;0,8], còn ứng viên có risk nằm trong [−0,6;0,2]. Do đó penalty mạnh hơn mức đóng góp tối đa của từng tín hiệu đơn lẻ nhưng vẫn chưa phải hard constraint: ứng viên risk có điểm 0,2 vẫn có thể thắng ứng viên không risk có điểm 0. Đây là lý do thiết kế hợp lý về mặt hành vi, không phải chứng cứ 0,6 tối ưu thực nghiệm.

Mẫu số 2 trong semantic drop là trung bình đều của liên kết trước và sau. Giá trị 0 và 1 trong min–max normalization/binary risk là định nghĩa biểu diễn. Các con số đó không được học từ dữ liệu; không gọi chúng là xác suất đã hiệu chỉnh. Ngưỡng drop 0,30 áp dụng cho raw drop, trong khi hai ngưỡng overlap áp dụng cho cosine; cùng bằng 0,30 nhưng có vai trò khác nhau.

## 4. Kết quả dev bổ sung


| Cấu hình | Recall@5 paragraph | Recall@5 article | Recall@5 TB | MRR@5 TB | Context TB | ID |
| --- | --- | --- | --- | --- | --- | --- |
| Mặc định | 97,00% | 95,71% | 96,35% | 0,8251 | 857.91 | cdd476acb6137 |
| Tốt nhất trong 180 cấu hình | 97,85% | 96,14% | 97,00% | 0,8363 | 811.37 | c3f3fd94a6633 |
| Tốt nhất nhóm trọng số | 97,00% | 96,57% | 96,78% | 0,8321 | 851.98 | c1dbc6b0dc5c6 |
| Tốt nhất nhóm ngưỡng drop | 97,42% | 95,71% | 96,57% | 0,8278 | 846.51 | c5fa627b282f6 |


Cấu hình có điểm cao nhất trong 180 cấu hình: MIN/TARGET/MAX=60/180/240; semantic/size/risk=0.4/0.4/0.6; drop=0.3; overlap=0.3/0.55. Những giá trị không nêu thay đổi vẫn theo mặc định. Context đếm theo khoảng trắng, gồm overlap; không phải số token LLM.


Chênh lệch Recall@5 TB giữa cấu hình thắng và mặc định là 0.644 điểm phần trăm. Khoảng bootstrap nguồn 95% thăm dò: [-0.226; 1.681] điểm phần trăm. Đã chọn cấu hình bằng chính dev nên khoảng này không phải kiểm định xác nhận độc lập, không hiệu chỉnh việc thử nhiều cấu hình. Chỉ có 10 nhóm bài nguồn.


### 4.1. Trọng số final score


| Semantic | Size | Risk | Recall@5 TB | MRR@5 TB |
| --- | --- | --- | --- | --- |
| 0.0 | 0.8 | 0.6 | 95,28% | 0,8350 |
| 0.1 | 0.7 | 0.6 | 95,28% | 0,8398 |
| 0.2 | 0.6 | 0.6 | 95,06% | 0,8337 |
| 0.3 | 0.5 | 0.6 | 95,71% | 0,8292 |
| 0.4 | 0.4 | 0.6 | 96,35% | 0,8251 |
| 0.5 | 0.3 | 0.6 | 96,78% | 0,8321 |
| 0.6 | 0.2 | 0.6 | 96,57% | 0,8310 |
| 0.7 | 0.1 | 0.6 | 96,35% | 0,8327 |
| 0.8 | 0.0 | 0.6 | 96,35% | 0,8349 |


Các bộ đồng hạng tốt nhất trong nhóm trọng số: (0.5; 0.3; 1.2); (0.5; 0.3; 0.0); (0.5; 0.3; 0.6); (0.5; 0.3; 0.8); (0.5; 0.3; 0.4); (0.5; 0.3; 0.2). Thứ tự là semantic, size, risk. Không được lấy bộ đầu tiên theo ID làm bằng chứng penalty đó tốt hơn các penalty đồng hạng.


Với cấu hình mặc định, paragraph có 6 lượt scoring và 0 ứng viên risk; article có 42 lượt scoring và 1 ứng viên risk. Khi giữ semantic=size=0,4, sáu mức penalty 0/0,2/0,4/0,6/0,8/1,2 tạo cùng corpus ở từng góc nhìn. Không đủ tín hiệu để xác định 0,6 là tốt nhất. Cần văn bản trên MAX có nhãn dependency do người rà soát; không tự lấy luật hiện tại làm gold để chứng minh chính luật đó.


### 4.2. Độ dài MIN TARGET MAX


| MIN | TARGET | MAX | Recall@5 TB | MRR@5 TB |
| --- | --- | --- | --- | --- |
| 60 | 90 | 120 | 93,56% | 0,8062 |
| 60 | 180 | 240 | 97,00% | 0,8363 |
| 80 | 120 | 160 | 93,78% | 0,8198 |
| 90 | 180 | 240 | 96,78% | 0,8351 |
| 100 | 150 | 200 | 94,21% | 0,8119 |
| 120 | 140 | 240 | 94,85% | 0,8274 |
| 120 | 160 | 240 | 96,35% | 0,8369 |
| 120 | 180 | 200 | 96,35% | 0,8395 |
| 120 | 180 | 240 | 96,35% | 0,8251 |
| 120 | 180 | 280 | 95,28% | 0,8185 |
| 120 | 180 | 320 | 95,06% | 0,8160 |
| 120 | 180 | 360 | 94,42% | 0,8147 |
| 120 | 200 | 240 | 95,71% | 0,8306 |
| 120 | 220 | 240 | 95,28% | 0,8293 |
| 150 | 180 | 240 | 93,78% | 0,8243 |
| 160 | 240 | 320 | 93,35% | 0,7908 |
| 200 | 300 | 400 | 93,99% | 0,8317 |


120/180/240 đứng đầu trong các bộ co giãn cùng tỷ lệ 2:3:4 đã thử, nhưng không đứng đầu mọi lựa chọn độ dài. Giảm riêng MIN xuống 60 cho điểm dev cao hơn. Vì vậy chỉ có thể bảo vệ 120/180/240 như một cấu hình ban đầu có kết quả khả quan trong một dải kiểm tra, không phải bộ tối ưu. Đơn vị độ dài là phần tử cách nhau bằng khoảng trắng, không phải subword token. MAX giới hạn ứng viên phần lõi, không bảo đảm giới hạn cứng sau overlap hoặc fallback.


### 4.3. Ngưỡng semantic drop


| Ngưỡng raw drop | Recall@5 TB | MRR@5 TB |
| --- | --- | --- |
| 0.0 | 95,92% | 0,8255 |
| 0.1 | 96,14% | 0,8255 |
| 0.2 | 96,57% | 0,8278 |
| 0.25 | 96,35% | 0,8251 |
| 0.3 | 96,35% | 0,8251 |
| 0.35 | 96,14% | 0,8214 |
| 0.4 | 96,14% | 0,8214 |
| 0.5 | 96,14% | 0,8214 |
| 0.6 | 96,14% | 0,8214 |
| Tắt optional cut (3) | 96,14% | 0,8214 |


Raw drop có miền lý thuyết [−2,2] do cosine thuộc [−1,1], nên ngưỡng 3 là chế độ đối chứng tắt optional cut. Ngưỡng 0,20 có điểm dev cao hơn 0,30 trong lưới này; không thể kết luận 0,30 tối ưu. Chỉ nhánh phần còn lại từ MIN đến MAX dùng ngưỡng raw drop; nhánh trên MAX dùng điểm tổng hợp và min–max normalization.


### 4.4. Ngưỡng và số câu overlap


| Ngưỡng thấp | Ngưỡng cao | Recall@5 TB | MRR@5 TB | Context TB |
| --- | --- | --- | --- | --- |
| 0.0 | 0.4 | 95,92% | 0,8339 | 899.64 |
| 0.0 | 0.55 | 96,35% | 0,8291 | 885.77 |
| 0.0 | 0.7 | 96,35% | 0,8372 | 872.64 |
| 0.0 | 0.85 | 96,35% | 0,8372 | 872.64 |
| 0.15 | 0.4 | 95,71% | 0,8354 | 896.62 |
| 0.15 | 0.55 | 96,35% | 0,8348 | 882.68 |
| 0.15 | 0.7 | 96,35% | 0,8428 | 869.51 |
| 0.15 | 0.85 | 96,35% | 0,8428 | 869.51 |
| 0.3 | 0.4 | 95,92% | 0,8240 | 872.21 |
| 0.3 | 0.55 | 96,35% | 0,8251 | 857.91 |
| 0.3 | 0.7 | 96,35% | 0,8361 | 844.27 |
| 0.3 | 0.85 | 96,35% | 0,8361 | 844.27 |
| 0.45 | 0.55 | 94,42% | 0,8163 | 830.43 |
| 0.45 | 0.7 | 94,42% | 0,8273 | 815.73 |
| 0.45 | 0.85 | 94,42% | 0,8273 | 815.73 |


| Cách overlap | Recall@5 TB | MRR@5 TB | Context TB |
| --- | --- | --- | --- |
| Động 0/1/1 câu | 96,35% | 0,8361 | 844.27 |
| Động 0/1/2 câu | 96,35% | 0,8251 | 857.91 |
| Động 0/1/3 câu | 95,92% | 0,8226 | 868.84 |
| Cố định 0 câu | 93,99% | 0,8166 | 796.48 |
| Cố định 1 câu | 96,35% | 0,8374 | 872.62 |
| Cố định 2 câu | 96,35% | 0,8350 | 959.04 |
| Cố định 3 câu | 93,13% | 0,7895 | 1031.66 |


Nhiều cặp ngưỡng đồng hạng Recall@5 nhưng khác MRR hoặc context. Cặp 0,30/0,55 và tối đa 2 câu không tối ưu duy nhất. Overlap cố định một câu và giới hạn động một câu là những đối chứng quan trọng; không thể kết luận “overlap động luôn tốt hơn”. Lượng context và giới hạn encode phải được xét cùng điểm retrieval.


### 4.5. Làm tròn và ablation


| Chữ số thập phân size score | Recall@5 TB | MRR@5 TB |
| --- | --- | --- |
| Không làm tròn | 96,35% | 0,8251 |
| 1 | 96,35% | 0,8264 |
| 2 | 96,35% | 0,8251 |
| 3 | 96,35% | 0,8251 |


| Ablation | Recall@5 TB | MRR@5 TB |
| --- | --- | --- |
| without_size | 96,35% | 0,8349 |
| semantic_only_score | 96,35% | 0,8349 |
| without_risk | 96,35% | 0,8251 |
| without_optional_cut | 96,14% | 0,8214 |
| size_only_score | 95,28% | 0,8350 |
| without_semantic_score | 95,28% | 0,8350 |
| without_overlap | 93,99% | 0,8166 |


Ablation “without_semantic_score” chỉ bỏ thành phần semantic trong final score; optional semantic cut vẫn tồn tại. “without_size” chỉ bỏ size score, vẫn giữ điều kiện MIN/MAX. Các tên này không được diễn giải thành một thuật toán hoàn toàn không dùng ngữ nghĩa hoặc hoàn toàn không có ràng buộc độ dài.


### 4.6. Số chunk truy xuất


| Góc nhìn | k | Evidence Recall@k | MRR@k | Context TB |
| --- | --- | --- | --- | --- |
| paragraph | 1 | 80,26% | 0,8026 | 153.64 |
| paragraph | 3 | 94,42% | 0,8677 | 454.94 |
| paragraph | 5 | 97,00% | 0,8732 | 767.87 |
| paragraph | 10 | 99,14% | 0,8758 | 1565.40 |
| article | 1 | 66,52% | 0,6652 | 187.24 |
| article | 3 | 88,84% | 0,7611 | 570.41 |
| article | 5 | 95,71% | 0,7770 | 947.96 |
| article | 10 | 97,85% | 0,7797 | 1902.64 |


Bảng dùng Ours mặc định trên dev. Tăng k từ 5 lên 10 tăng Recall nhưng làm lượng context gần gấp đôi; chưa có hàm mục tiêu về chi phí hoặc đánh giá generation để xác định k tối ưu. Top-5 là điểm vận hành đã chọn cho đối chứng, không phải cực đại Recall. File nguồn: evaluation/results/parameter_study/retrieval_k.csv.


## 5. Giải trình các số còn lại


| Con số | Căn cứ và trạng thái | Có được gọi là tối ưu không |
| --- | --- | --- |
| 0/1 trong risk; 0/1 trong normalization | Định nghĩa cờ khớp luật và miền chuẩn hóa, không phải xác suất được học | Không đặt vấn đề tối ưu độ lớn tuyệt đối |
| 2 trong trung bình semantic drop; 4 câu lân cận | Thiết kế đối xứng trước/qua/sau boundary, đủ ngữ cảnh để tính hai liên kết nội bộ | Chưa so sánh mọi loại cửa sổ khác |
| 512 câu cache | Giới hạn bộ nhớ LRU trong production; chỉ ảnh hưởng tái sử dụng tính toán | Không phải tham số quyết định chất lượng chunk |
| Max–Min 0,6; 0,9; 1,5 | Kế thừa mặc định trong mã tác giả tại commit cố định; không phải tìm bằng XQuAD của nhóm | Không tuyên bố tối ưu trên tiếng Việt |
| LlamaIndex buffer=1; percentile=95 | Mặc định thư viện 0.14.25, baseline lịch sử; percentile không phải cosine 0,95 | Không tuyên bố nhóm đã tune |
| MiniLM 128 token | max_seq_length trong sentence_bert_config.json tại snapshot được ghi hash | Giới hạn cấu hình encoder, không phải ngưỡng nhóm tối ưu |
| Top-3 và top-5 | Mốc chấm retrieval và chính sách lấy context; đã bổ sung độ nhạy k=1/3/5/10 | Không thể tối đa Recall đơn thuần để chứng minh k=5 tối ưu chi phí |
| 512 và 1024 token context | Hai trần chẩn đoán hậu nghiệm trên tiền tố top-5 đã lưu; không skip/truncate/backfill | Hai mức thử nghiệm, không có nghĩa mức tối ưu |
| ~20% dev; seed chia dữ liệu | Quy tắc hash theo title trong từng lĩnh vực; kiểm tra dev/test tách nguồn | Lựa chọn giao thức, không chọn split theo điểm cao |
| 240 context; 1.190 QA; 48 bài; 7 lĩnh vực | Đếm lại dữ liệu và tái tạo conversion trong bộ nhớ; 7 lĩnh vực là nhãn rộng do dự án gán | Số đếm, không phải siêu tham số |
| 233 dev/957 test; 50/190 paragraph; 10/38 article | Đếm lại và đối chiếu manifest; mọi câu cùng nguồn đi cùng split | Số đếm có thể tái lập |
| 5 context mỗi bài ghép; 1.181 câu hỏi duy nhất; 9 nhóm trùng | Đối chiếu conversion và duplicate audit; hai góc nhìn không độc lập | Số đếm có thể tái lập |
| 39–763; median 155; 496–2.169; median 804,5 | Đếm lại đơn vị khoảng trắng của toàn bộ tài liệu trong manifest | Thống kê, không phải giới hạn tối ưu |
| 95% CI; 5.000 bootstrap; seed 2026 | Quy ước mức khoảng và cấu hình tái lập; resample theo bài nguồn; đã bổ sung 1.000/5.000/10.000 lượt | Không chứng minh 5.000 hay seed 2026 là tối ưu |
| Recall/MRR; số chunk; bảo toàn evidence; context; paired hits/misses | Tính lại từ ranking gốc và nhãn, đối chiếu JSON/CSV/statistical summary | Kết quả quan sát có phạm vi benchmark |
| 13 pilot; 16 câu pilot; 3 dependency/4 tài liệu; 2/3 so với 0/3 | Đếm nhãn và chunk pilot; tài liệu dependency đều dưới 240 nên chưa kiểm chứng riêng penalty | Không phải bằng chứng tổng quát về risk |
| 86,73% và 97,91% chunk Ours vượt 128 | Tokenize lại mọi chunk với cùng snapshot; đúng độ dài không đồng nghĩa evidence chắc chắn bị truncation | Thống kê độ dài, không phải tỷ lệ mất evidence |
| Temperature=0; output tối đa 512 token; timeout=180 giây | Cấu hình adapter Ollama để vận hành; benchmark chính không gọi generation | Chưa có thực nghiệm chứng minh tối ưu QA |
| Phiên bản package; năm xuất bản; commit/hash; ngày chạy | Metadata tái lập hoặc thông tin thư mục tài liệu tham khảo; phiên bản cũ là metadata lần chạy cũ | Không phải kết quả tối ưu tham số |
| Thời gian chạy đã lưu | Một lần chạy warm model, phụ thuộc phần cứng/cache/thứ tự; không thể tái tạo cùng số giây tuyệt đối | Không phải benchmark tốc độ thống kê |



Nguồn gốc các cấu hình kế thừa:

- [Mã nguồn Max–Min của tác giả tại commit cố định](https://raw.githubusercontent.com/hsdslab/MaxMinChunking/0caba41d2ef30114729f95e5c788173e4f40920e/maxmin_chunker.py) có mặc định 0,6/0,9/1,5. Ở cụm một câu, 1,5 × cosine > 0,6 tương đương cosine > 0,4. Công thức code dùng sigmoid(size−1); không nhầm với cách viết sigmoid(size) trong bài báo.
- [Mã nguồn SemanticSplitter phiên bản 0.14.25](https://raw.githubusercontent.com/run-llama/llama_index/v0.14.25/llama-index-core/llama_index/core/node_parser/text/semantic_splitter.py) ghi buffer=1 và percentile=95. Buffer=1 thực tế lấy câu trước/câu hiện tại/câu sau khi xây nhóm.
- [Cấu hình MiniLM tại snapshot đã kiểm tra](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/raw/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/sentence_bert_config.json) ghi max_seq_length=128. Không tự nâng giới hạn chỉ để làm kết quả đẹp hơn.
- [README XQuAD ở commit nguồn](https://raw.githubusercontent.com/google-deepmind/xquad/7d30520c717524000f0d9d2f9c10a069acd9d285/README.md) mô tả 240 paragraph và 1.190 câu hỏi–đáp án; split dev/test và nhãn evidence trong repo là bản chuyển đổi cục bộ.

## 6. Kiểm tra lại số liệu thực nghiệm cũ

Script verify_parameter_evidence.py tái tạo dữ liệu chuyển đổi trong bộ nhớ và kiểm tra checksum; tính lại Recall/MRR từ vector nghiên cứu mới; tái lập bootstrap cũ; chạy lại audit_report_evidence.py vào một thư mục mới để tính tokenizer, ngân sách context và dependency. Các mảng số liệu gốc khớp hoàn toàn, không ghi đè kết quả cũ.

Các chi tiết nằm trong:

- evaluation/results/parameter_study/verification.json: các điều kiện đã kiểm chứng.
- numeric_audit/original_metrics_verified.csv: số tài liệu/câu hỏi/chunk, Recall@3/@5, MRR@5, độ dài và preservation của kết quả cũ.
- numeric_audit/tokenizer_summary.csv và chunk_token_lengths.csv: độ dài token và tỷ lệ vượt 128.
- numeric_audit/budget_summary.csv và budget_questions.csv: kết quả từng budget và từng câu hỏi.
- numeric_audit/audit.json: dependency, provenance và hash nguồn.
- bootstrap_stability.csv: độ nhạy 1.000/5.000/10.000 lượt bootstrap cho ba metric; seed cố định để cô lập ảnh hưởng số lượt.
- report_numeric_mentions.csv: chỉ mục các lần xuất hiện con số trong phần thân Markdown để rà soát thủ công. Không xem việc trích số bằng regex là chứng minh nội dung câu văn.

Các phiên bản package và thời gian là metadata của lần chạy được lưu; không giả định phần cứng hoặc phiên bản hiện tại trùng lần cũ. Nhãn evidence tự động chỉ bao câu chứa đáp án đầu tiên; không phải supporting facts đã được người xác nhận. Tính đúng các con số không tự chứng minh chất lượng nhãn hoặc kết luận nhân quả.

## 7. Cách trình bày trong báo cáo và khi bảo vệ

Đoạn có thể dùng trực tiếp:

> Các tham số 120/180/240, trọng số 0,4/0,4/0,6 và ngưỡng 0,30/0,55 ban đầu được xác định theo thiết kế heuristic nhằm cân bằng độ dài, ngữ nghĩa và liên kết câu. Nhóm bổ sung nghiên cứu độ nhạy gồm 180 cấu hình trên 233 câu hỏi dev với hai cách tổ chức cùng dữ liệu. Cấu hình hiện tại đạt kết quả khả quan nhưng không tối ưu trong toàn bộ lưới đã thử; một số cấu hình khác có điểm dev cao hơn. Riêng penalty dependency chưa được xác định đáng tin cậy do ít ứng viên kích hoạt luật. Vì vậy nhóm không tuyên bố các hằng số hiện tại là tối ưu toàn cục; các kết quả test gốc tiếp tục mô tả đúng cấu hình ban đầu, còn cấu hình hiệu chỉnh cần được xác nhận trên dữ liệu độc lập.

Nếu được hỏi “vì sao dùng 0,4/0,4/0,6?”: hai tín hiệu [0,1] được đặt trọng số bằng nhau, risk được phạt mạnh hơn từng tín hiệu; đây là lý do thiết kế. Bảng dev là kiểm tra thực nghiệm bổ sung và hiện không xác nhận đây là bộ tốt nhất. Không dùng một bài báo khác để gán thành bằng chứng rằng chính ba con số của nhóm tối ưu.

Nếu được hỏi “vì sao 0,30 và 0,55?”: đó là ngưỡng ban đầu, sau đó đã thử các mức lân cận và đối chứng overlap cố định. Có các cấu hình đồng hạng hoặc tốt hơn tùy metric. Cần nêu rõ kết quả, không chỉ nêu trực giác “thấp/vừa/cao”.

Nếu cần công bố một cấu hình mới được chọn bằng thực nghiệm: đóng băng tiêu chí gồm cả chi phí context, dùng dev để chọn; xác nhận một lần trên corpus mới có nhãn đáng tin cậy, có văn bản dài và dependency. Nghiên cứu hiện tại không cung cấp bằng chứng rằng bất kỳ cấu hình nào là tốt nhất với mọi corpus hoặc mọi LLM.

## 8. Tái lập và dữ liệu kiểm tra

Chạy từ thư mục gốc:

```powershell
.\venv\Scripts\python.exe -X utf8 evaluation/parameter_sensitivity.py
.\venv\Scripts\python.exe -X utf8 scripts/verify_parameter_evidence.py
.\venv\Scripts\python.exe -X utf8 scripts/write_parameter_report.py
.\venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

Chỉ cần các dependency hiện có và model/tokenizer đã lưu trong cache. Không cần Ollama hoặc API key.

protocol.json lưu không gian tìm kiếm và tiêu chí; sensitivity.csv lưu từng góc nhìn; aggregate.csv lưu tiêu chí xếp hạng; summary.json ghi phiên bản/hash/snapshot; details.json.gz lưu hit/miss và ID chunk theo từng câu hỏi; corpora.json.gz và embeddings.npz cho phép tính lại ranking mà không tải model. Các file .gz là JSON nén gzip thông thường. dev_exploratory_intervals.json ghi các khoảng thăm dò với cảnh báo thiên lệch do lựa chọn. Không tự động đổi tham số production hay ghi lại benchmark test.

