"""Write the Vietnamese research manuscript from current-run evidence."""

import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'evaluation/results/scientific_paper_current'
STUDY=ROOT/'evaluation/results/parameter_study'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def decimal(value,digits=2):
    return f'{float(value):.{digits}f}'.replace('.',',')


def percent(value):
    return decimal(100*float(value))+'%'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(str(value) for value in row)+' |' for row in rows])


PAPER=r'''# Phân đoạn văn bản tiếng Việt theo ngữ nghĩa kết hợp luật liên kết câu và overlap động cho truy xuất trong RAG

**Vương Đức Hùng**

Khoa Công nghệ thông tin, Trường Đại học Giao thông vận tải

## Tóm tắt

Phân đoạn tài liệu ảnh hưởng trực tiếp đến lượng bằng chứng và ngữ cảnh mà hệ thống sinh tăng cường truy xuất có thể sử dụng. Đối với tiếng Việt, một điểm cắt giữa các câu có quan hệ điều kiện, nguyên nhân hoặc tương phản có thể làm phân tán thông tin cần thiết cho câu hỏi. Bài báo trình bày một chiến lược phân đoạn kết hợp suy giảm tương đồng ngữ nghĩa cục bộ, độ dài đoạn, luật liên kết câu và overlap động. Điểm ranh giới của cấu hình triển khai sử dụng trọng số 0,5 cho ngữ nghĩa, 0,3 cho độ dài và penalty 0,6 cho dấu hiệu liên kết. Nghiên cứu độ nhạy gồm 180 cấu hình trên 233 câu hỏi phát triển được sử dụng để đánh giá các lựa chọn tham số; tỷ lệ trọng số triển khai được chọn trong nhóm 54 bộ trọng số. Phương pháp được đối chiếu với Max–Min trong cùng pipeline embedding và truy xuất trên bản chuyển đổi XQuAD tiếng Việt. Trên 957 câu hỏi đánh giá, phương pháp đề xuất đạt Evidence Recall@5 __P_OURS__ với context gốc, so với __P_BASE__ của Max–Min. Khi ghép các context cùng bài thành tài liệu dài, hai giá trị tương ứng là __A_OURS__ và __A_BASE__. Phương pháp đề xuất bảo toàn toàn bộ evidence theo nhãn hiện có, nhưng sử dụng nhiều context hơn và có tỷ lệ chunk vượt giới hạn encode cao. Kết quả cho thấy hiệu quả phụ thuộc cách tổ chức tài liệu và ngân sách ngữ cảnh, đồng thời chưa xác nhận lợi ích riêng của penalty liên kết câu hoặc ưu thế tổng quát của phương pháp. Thực nghiệm tập trung vào truy xuất bằng chứng, chưa đánh giá chất lượng câu trả lời sinh bởi mô hình ngôn ngữ.

**Từ khóa:** phân đoạn ngữ nghĩa; tiếng Việt; liên kết câu; overlap động; truy xuất bằng chứng; RAG.

## 1 Giới thiệu

Sinh tăng cường truy xuất (Retrieval-Augmented Generation, RAG) kết hợp khả năng sinh ngôn ngữ với thông tin được tìm từ nguồn bên ngoài [1]. Trong hệ thống sử dụng chỉ mục vector, tài liệu thường được chia thành các đoạn trước khi mã hóa. Vì vậy, đơn vị phân đoạn quyết định nội dung được biểu diễn bởi mỗi vector và lượng ngữ cảnh có thể được đưa vào bước sinh câu trả lời. Việc chọn chunk là một thành phần của bài toán truy xuất, không chỉ là thao tác chia nhỏ tài liệu.

Kích thước chunk tạo ra một đánh đổi. Đoạn ngắn có thể biểu diễn một chi tiết tập trung nhưng thiếu thông tin liên quan ở câu lân cận. Đoạn dài có thể giữ đủ ngữ cảnh, song một vector phải đại diện cho nhiều nội dung và văn bản có thể vượt giới hạn đầu vào của encoder. Overlap giúp đưa lại một phần ngữ cảnh vào đoạn kế tiếp, nhưng cũng làm tăng dữ liệu lặp và lượng văn bản được truy xuất. Do đó, cùng số chunk top-k chưa bảo đảm hai phương pháp sử dụng cùng lượng ngữ cảnh.

Đối với tiếng Việt, dấu hiệu như “nếu–thì”, “vì–nên” hoặc “tuy–nhưng” có thể biểu thị quan hệ giữa hai câu liền kề. Một bộ phân đoạn dựa trên độ dài hoặc chỉ một độ đo tương đồng không sử dụng trực tiếp các dấu hiệu này. Chúng tôi khảo sát việc đưa luật liên kết vào điểm ranh giới, đồng thời thay đổi số câu overlap theo mức tương đồng tại vị trí cắt. Luật được sử dụng như một penalty mềm; nó không được xem là bộ phân tích cú pháp hay cơ chế bảo đảm giữ nguyên mọi quan hệ liên câu.

Nghiên cứu đặt ra ba câu hỏi. Thứ nhất, chiến lược kết hợp có truy xuất được evidence tốt hơn Max–Min khi giữ nguyên encoder, bộ tách câu và retriever hay không? Thứ hai, kết quả thay đổi như thế nào khi cùng nội dung và câu hỏi được tổ chức thành context ngắn hoặc tài liệu ghép dài? Thứ ba, trọng số và các ngưỡng ảnh hưởng ra sao đến chất lượng truy xuất và lượng context được sử dụng?

Đóng góp của bài báo là xây dựng và mô tả một chiến lược phân đoạn tiếng Việt có thể tái lập, đánh giá có đối chứng trên hai cách tổ chức cùng dữ liệu, và phân tích độ nhạy tham số cùng các đánh đổi về ngữ cảnh. Bài báo sử dụng embedding có sẵn, không đề xuất hoặc huấn luyện một encoder mới. Kết quả được trình bày như bằng chứng trong phạm vi benchmark đã khảo sát; nghiên cứu không tuyên bố tính mới tuyệt đối của từng thành phần hoặc ưu thế trên mọi loại tài liệu.

Phần 2 thảo luận các hướng liên quan. Phần 3 trình bày phương pháp. Dữ liệu và thiết lập được mô tả ở Phần 4; Phần 5 báo cáo kết quả, độ nhạy và giới hạn; Phần 6 đưa ra kết luận.

## 2 Nghiên cứu liên quan

Các chiến lược phân đoạn theo kích thước sử dụng số từ, số token hoặc số câu làm đơn vị chia. Chúng có ưu điểm dễ kiểm soát độ dài nhưng có thể đặt ranh giới ở vị trí không phù hợp về nội dung. Phân đoạn dựa trên cấu trúc tài liệu sử dụng tiêu đề, đoạn văn và các khối nội dung; hiệu quả phụ thuộc vào chất lượng cấu trúc đầu vào. Overlap thường được bổ sung để giảm thiếu hụt ngữ cảnh, song không tự bảo đảm rằng đoạn truy xuất chứa đúng bằng chứng cần thiết.

Phân đoạn ngữ nghĩa sử dụng biểu diễn câu hoặc cửa sổ câu để phát hiện thay đổi nội dung. LlamaIndex SemanticSplitter tính mức khác biệt giữa các nhóm câu và chọn ranh giới theo percentile [5]. Trong implementation được đối chiếu, buffer bằng một tạo nhóm gồm câu hiện tại cùng câu trước và câu sau khi chúng tồn tại. Cửa sổ ngữ cảnh phục vụ tính ranh giới này khác với việc lặp câu ở các chunk đầu ra. LlamaIndex đã được sử dụng trong thí nghiệm pilot của dự án; đối chứng chính của bài báo là Max–Min.

Max–Min của Kiss, Nagy và Szilágyi gom các câu liên tiếp dựa trên tương đồng giữa câu mới và cụm hiện tại [2]. Nghiên cứu này sử dụng implementation công bố của tác giả, thay đầu vào bằng bộ tách câu và encoder đa ngôn ngữ dùng chung cho cả hai chiến lược [8]. Đây là bản thích nghi cho đối chứng tiếng Việt; chúng tôi không so trực tiếp điểm retrieval ở đây với chỉ số phân cụm hoặc hỏi đáp của thí nghiệm gốc.

Reimers và Gurevych trình bày cách xây dựng biểu diễn câu đa ngôn ngữ bằng chưng cất tri thức [3]. Trong nghiên cứu này, mô hình `paraphrase-multilingual-MiniLM-L12-v2` đóng vai trò encoder dùng chung [7]. Việc dùng cùng encoder giúp kiểm soát một thành phần của pipeline, nhưng không làm hai chunker mã hóa cùng đơn vị nội dung: câu được dùng để quyết định ranh giới, còn chunk được mã hóa riêng để truy xuất.

Qu và cộng sự khảo sát chi phí của semantic chunking và cho thấy lợi ích không nhất quán trên các tác vụ họ đánh giá [6]. Nhận xét này đặt ra yêu cầu đo cả chất lượng truy xuất lẫn lượng ngữ cảnh và chi phí xử lý. Bài báo hiện tại tập trung vào một phương án tích hợp tín hiệu tiếng Việt, đồng thời công bố cả các thiết lập có lợi và bất lợi cho phương pháp đề xuất.

## 3 Phương pháp

### 3.1 Biểu diễn tài liệu và ranh giới

Cho tài liệu D được tách thành dãy câu có thứ tự \(s_1,\ldots,s_n\). Bộ tách câu hiện tại sử dụng dấu chấm, chấm hỏi hoặc chấm than có khoảng trắng theo sau. Phương pháp chỉ đặt điểm cắt giữa các câu; nó không chia một câu thành nhiều mệnh đề. Mỗi câu được biểu diễn bởi vector \(E(s_i)\). Độ tương đồng được xác định bằng cosine:

\[
c(s_i,s_j)=\frac{E(s_i)^\top E(s_j)}{\|E(s_i)\|\,\|E(s_j)\|}. \tag{1}
\]

Xét phần tài liệu chưa xử lý bắt đầu tại câu \(s_p\). Ranh giới \(b_i\) nằm giữa \(s_i\) và \(s_{i+1}\), với độ dài tích lũy bên trái:

\[
L_p(b_i)=\sum_{j=p}^{i}\ell(s_j),\qquad
\ell(s)=|\operatorname{split}_{ws}(s)|. \tag{2}
\]

Hàm \(\ell\) đếm đơn vị phân cách bởi khoảng trắng. Đơn vị này không đồng nhất với từ đã phân đoạn tiếng Việt hoặc subword token của encoder. Phần overlap kế thừa từ lượt trước được giữ riêng, không cộng vào \(L_p\) khi quyết định điểm cắt mới.

### 3.2 Suy giảm liên kết ngữ nghĩa cục bộ

Đối với ranh giới có đủ một câu trước và một câu sau cặp câu kề biên, chúng tôi tính trên các câu thuộc phần tài liệu chưa xử lý:

\[
\Delta_i=\frac{c(s_{i-1},s_i)+c(s_{i+1},s_{i+2})}{2}
-c(s_i,s_{i+1}). \tag{3}
\]

Giá trị lớn cho biết liên kết qua ranh giới yếu hơn liên kết ở hai phía. Đây là một tín hiệu tương đối trong lân cận bốn câu, không phải xác suất chuyển chủ đề. Khi thiếu câu lân cận, \(\Delta_i\) không được xác định.

Ở nhánh chọn ranh giới cho văn bản dài, tập ứng viên hợp lệ là \(\mathcal B_p=\{b_i:m\le L_p(b_i)\le M\}\). Các giá trị drop hợp lệ được chuẩn hóa trong tập này:

\[
S_{sem}(b_i)=\frac{\Delta_i-\Delta_{min}}{\Delta_{max}-\Delta_{min}}. \tag{4}
\]

Nếu không có drop hợp lệ, không có biến thiên hoặc ứng viên thiếu ngữ cảnh, semantic score của trường hợp tương ứng được gán bằng 0. Ngưỡng cắt tùy chọn ở Mục 3.5 áp dụng trực tiếp lên raw drop của (3), không áp dụng lên điểm chuẩn hóa của (4).

### 3.3 Độ dài và dấu hiệu liên kết câu

Với độ dài tối thiểu m, mục tiêu t và tối đa M, điểm độ dài có dạng tam giác:

\[
S_{size}(L)=
\begin{cases}
0,&L<m\ \text{hoặc}\ L>M,\\
(L-m)/(t-m),&m\le L\le t,\\
(M-L)/(M-t),&t<L\le M.
\end{cases} \tag{5}
\]

Implementation làm tròn điểm này đến hai chữ số thập phân. Cấu hình triển khai dùng \((m,t,M)=(120,180,240)\). Các giá trị biểu diễn ưu tiên độ dài phần lõi tại thời điểm chọn ranh giới; chúng không phải cam kết độ dài cuối cùng của mọi chunk.

Hàm \(R(b_i)\in\{0,1\}\) nhận giá trị 1 khi hai phía ranh giới khớp một cặp dấu hiệu liên kết. Tập luật gồm “nếu–thì”, “vì–nên”, “do–nên”, “tuy–nhưng”, “mặc dù–nhưng”, “không những–mà còn”, “không phải–mà” và “hễ–thì”. Văn bản được chuyển về chữ thường trước khi khớp chuỗi con. Luật “vì–nên” được bỏ qua nếu phía trái chứa “vì vậy”; luật “do–nên” được bỏ qua nếu phía trái chứa “do đó”. Cách khớp này có thể bỏ sót cách diễn đạt khác hoặc nhận nhầm một chuỗi nằm bên trong từ; R chỉ là cờ heuristic, không phải nhãn phụ thuộc cú pháp được suy luận đầy đủ.

### 3.4 Hàm điểm ranh giới

Các ứng viên hợp lệ trong nhánh văn bản dài được xếp hạng theo:

\[
F(b_i)=\alpha S_{sem}(b_i)+\beta S_{size}(b_i)-\gamma R(b_i). \tag{6}
\]

Cấu hình triển khai hiện tại là:

\[
\boxed{F(b_i)=0.5S_{sem}(b_i)+0.3S_{size}(b_i)-0.6R(b_i).} \tag{7}
\]

Tỷ lệ \(\alpha/\beta\) được chọn theo Evidence Recall@5 trung bình trên dev trong lưới trọng số mô tả ở Mục 4.4. Các mức penalty được thử đồng hạng ở tỷ lệ được chọn; vì vậy giữ \(\gamma=0.6\) là quyết định giữa các cấu hình đồng hạng, không phải bằng chứng nó tối ưu duy nhất.

Các điểm \(S_{sem}\) và \(S_{size}\) nằm trong [0,1], nên ứng viên không risk có F trong [0;0,8], còn ứng viên risk có F trong [−0,6;0,2]. Penalty vì vậy có thể giảm ưu tiên mạnh nhưng không cấm tuyệt đối việc chọn ranh giới risk. Khi bằng điểm, thuật toán chọn ứng viên xuất hiện sớm nhất. Đồng thời, nhân cả ba hệ số với cùng một hằng số dương không đổi argmax trong số học chính xác; ý nghĩa lựa chọn nằm ở tỷ lệ các hệ số.

### 3.5 Quyết định cắt theo phần văn bản còn lại

Gọi \(L_{rem}\) là độ dài phần chưa xử lý. Khi \(L_{rem}<m\), toàn bộ phần này được giữ thành một đoạn lõi. Khi \(m\le L_{rem}\le M\), thuật toán xét những ranh giới có phần trái dài ít nhất m và có drop hợp lệ, chọn raw drop lớn nhất rồi chỉ cắt nếu \(\Delta_i\ge\theta\), với \(\theta=0.30\). Nhánh này không sử dụng penalty R và không bắt buộc phần còn lại sau cắt phải dài ít nhất m.

Khi \(L_{rem}>M\), thuật toán chọn ứng viên có F lớn nhất trong \(\mathcal B_p\). Nếu không có ứng viên phù hợp, toàn bộ phần còn lại được giữ nguyên. Trường hợp này có thể xảy ra với câu dài hoặc khi độ dài tích lũy vượt qua khoảng hợp lệ mà không có ranh giới câu bên trong. Vì overlap được thêm sau bước chọn biên và còn có cơ chế giữ nguyên khi không tìm được biên, M không phải trần cứng của chunk đầu ra.

### 3.6 Chồng lấn ngữ cảnh động và thuật toán tổng thể

Chồng lấn ngữ cảnh (overlap) đưa một số câu cuối của đoạn trước vào đầu đoạn kế tiếp. Sau khi chọn đoạn lõi, đặt q là cosine giữa câu cuối của đoạn này và câu đầu của phần chưa xử lý. Số câu cuối được chuyển sang đầu chunk kế tiếp là:

\[
o(q)=\begin{cases}
0,&q<0.30,\\
1,&0.30\le q<0.55,\\
2,&q\ge0.55.
\end{cases} \tag{8}
\]

Số câu thực giữ được giới hạn bởi số câu của đoạn lõi. Overlap không được đưa ngược vào phần chưa xử lý; mỗi lượt luôn tiến qua phần nội dung mới. Giả thuyết thiết kế là ranh giới còn liên kết mạnh cần thêm ngữ cảnh, trong khi ranh giới khác biệt mạnh có thể không cần lặp lại câu. Giả thuyết này được khảo sát bằng các ngưỡng và chính sách overlap đối chứng ở Phần 5.

**Thuật toán 1. Phân đoạn ngữ nghĩa kết hợp luật liên kết và overlap động.**

```text
Đầu vào: văn bản D; m, t, M; α, β, γ; θ; ngưỡng overlap τ_low, τ_high
Đầu ra: danh sách chunk C
1. remaining ← các câu của D; pending ← rỗng; C ← rỗng
2. Trong khi remaining khác rỗng:
3.     Tính độ dài remaining và tạo các ranh giới câu khi cần
4.     Nếu độ dài < m: core ← toàn bộ remaining
5.     Nếu m ≤ độ dài ≤ M:
6.         Chọn raw drop lớn nhất ở biên đủ ngữ cảnh, phần trái ≥ m
7.         Cắt nếu drop ≥ θ; nếu không, core ← toàn bộ remaining
8.     Nếu độ dài > M:
9.         Chọn argmax F trong các biên có phần trái thuộc [m,M]
10.        Nếu không có biên hợp lệ: core ← toàn bộ remaining
11.    Bỏ phần core mới khỏi remaining
12.    Thêm pending + core vào C
13.    Nếu còn remaining: pending ← tối đa o(q) câu cuối của core
14.    Nếu không: pending ← rỗng
15. Trả về C
```

Embedding câu phục vụ chọn ranh giới và overlap; embedding chunk được tính riêng cho retrieval. Cache câu có giới hạn 512 mục trong code chạy thông thường. Cache hỗ trợ tái sử dụng phép encode, không thay đổi công thức quyết định. Các chunk giữ thứ tự nội dung, nhưng khoảng trắng và xuống dòng được chuẩn hóa khi ghép câu.

## 4 Dữ liệu và thiết lập thực nghiệm

### 4.1 XQuAD tiếng Việt và hai cách tổ chức tài liệu

Nghiên cứu sử dụng bản tiếng Việt của XQuAD do Artetxe và cộng sự công bố [4,9]. Dữ liệu được cố định tại commit `7d30520c717524000f0d9d2f9c10a069acd9d285`, gồm 240 context và 1.190 cặp hỏi–đáp thuộc 48 bài nguồn. Bản chuyển đổi giữ nguyên nội dung câu hỏi, đáp án và offset nguồn; nhãn lĩnh vực, split cục bộ và evidence phục vụ retrieval được bổ sung bởi nghiên cứu này. Giấy phép dữ liệu nguồn và chuyển đổi là CC BY-SA 4.0.

Góc nhìn **context gốc** xem mỗi paragraph XQuAD là một tài liệu. Góc nhìn **ghép dài** nối năm context có cùng tiêu đề theo thứ tự phát hành thành một tài liệu. Văn bản ghép chỉ bao gồm các đoạn trích được nguồn cung cấp, không phải bài Wikipedia đầy đủ và không bảo đảm các đoạn liền kề trong bản gốc. Hai góc nhìn sử dụng cùng nội dung và câu hỏi, được xây dựng nhằm khảo sát ảnh hưởng của đơn vị tài liệu.

**Bảng 1. Quy mô dữ liệu theo split cục bộ và cách tổ chức tài liệu.**

__DATA_TABLE__

Các bài nguồn được gán vào bảy nhóm lĩnh vực rộng. Trong mỗi nhóm, thứ tự hash tiêu đề với seed `utc-rag-xquad-vi-v1` xác định khoảng 20% bài làm dev; phần còn lại dùng đánh giá. Mọi context và câu hỏi của cùng bài nằm trong cùng split. Đây là cách chia cục bộ, không phải split chính thức của XQuAD. Các kiểm tra dữ liệu xác nhận split không trùng bài nguồn, context hoặc ID câu hỏi. Không cộng hai góc nhìn thành 2.380 câu hỏi độc lập.

Evidence chuẩn được lấy bằng bao câu nhỏ nhất chứa đáp án đầu tiên trong dữ liệu nguồn. Offset được kiểm tra để bảo đảm evidence thuộc đúng tài liệu và chứa đáp án. Nhãn này được suy ra tự động, chưa được rà soát đầy đủ như supporting facts do người gán. Toàn bộ dữ liệu có 1.181 chuỗi câu hỏi chuẩn hóa duy nhất và chín nhóm câu hỏi lặp; ba nhãn nguồn đáng nghi được ghi trong hồ sơ review. Các nhãn không được sửa theo kết quả của chunker.

### 4.2 Phương pháp đối chứng và pipeline dùng chung

Max–Min được triển khai theo mã tác giả tại commit `0caba41d2ef30114729f95e5c788173e4f40920e` [8]. Các tham số \(hard\_thr=0.6\), \(c=0.9\) và \(init\_const=1.5\) là cấu hình khởi đầu được tác giả đề xuất [2]. Đối chứng không được thêm overlap, luật tiếng Việt hoặc giới hạn độ dài của phương pháp đề xuất. Chúng tôi giữ điều kiện so sánh nghiêm ngặt và cách dùng sigmoid(size−1) trong code công bố; cách viết này khác sigmoid(size) ở công thức (4) của bài gốc.

Hai phương pháp dùng chung bộ tách câu, instance MiniLM và tập tài liệu. Từng chunk được gắn ID nguồn, mã hóa, chuẩn hóa vector và đưa vào chỉ mục cosine trong bộ nhớ. Vector câu hỏi được tính một lần cho mỗi góc nhìn để dùng chung. Retrieval tìm trên toàn corpus, không được giới hạn bởi ID tài liệu chuẩn. Khi cosine bằng nhau, thứ tự tài liệu/chunk trong corpus được giữ ổn định. Không có reranker, không sử dụng gold evidence để chọn chunk và không gọi LLM sinh câu trả lời.

**Bảng 2. Cấu hình cố định trong lần đánh giá với trọng số triển khai.**

__CONFIG_TABLE__

Snapshot MiniLM được sử dụng là `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`; cấu hình SentenceTransformer giới hạn đầu vào ở 128 token [7]. Giới hạn này không đồng nhất với số đơn vị khoảng trắng của chunker. Pipeline giữ nguyên chunk trước khi encode và không chia lại chunk để tránh truncation. Mọi phương pháp chịu cùng cấu hình encoder.

### 4.3 Chỉ số truy xuất và phân loại thất bại

Với câu hỏi q, evidence chuẩn \(e_q\), tài liệu nguồn \(d_q\) và danh sách truy xuất \(\mathcal R_k(q)\), hit được xác định bởi:

\[
h_k(q)=\mathbf 1\{\exists c\in\mathcal R_k(q):d(c)=d_q\land e_q\subseteq c\}. \tag{9}
\]

Quan hệ chứa trong (9) là phép tìm chuỗi con sau chuẩn hóa Unicode NFC, chữ hoa/thường và khoảng trắng. Evidence phải nằm trọn trong một chunk đúng nguồn. Hai chunk ghép lại đủ evidence vẫn không được tính hit. Vì mỗi câu hỏi có một evidence mục tiêu, chỉ số dùng ở đây là:

\[
\operatorname{EvidenceRecall}@k=\frac{1}{|Q|}\sum_{q\in Q}h_k(q). \tag{10}
\]

Đây là tỷ lệ câu hỏi truy xuất trúng evidence theo nhãn, không phải recall trên mọi tài liệu có liên quan. Nếu \(r_q\) là hạng đầu tiên chứa đủ evidence trong top-5, MRR được tính bằng:

\[
\operatorname{MRR}@5=\frac{1}{|Q|}\sum_{q\in Q}
\begin{cases}1/r_q,&r_q\le5,\\0,&\text{không có hit trong top-5.}\end{cases} \tag{11}
\]

Tỷ lệ bảo toàn evidence kiểm tra có ít nhất một chunk phù hợp ở bất kỳ vị trí nào trong corpus. Một câu hỏi không có chunk như vậy được phân loại *evidence_split*. Nếu chunk đủ evidence tồn tại nhưng không lên top-5, thất bại được gọi là *ranking_miss*. Ngoài chất lượng truy xuất, chúng tôi đo số chunk, độ dài trung bình và tổng context top-5, có tính nội dung lặp do overlap.

### 4.4 Lựa chọn tham số và phân tích độ nhạy

Nghiên cứu dev gồm 180 cấu hình riêng biệt: 54 bộ trọng số, các thay đổi MIN/TARGET/MAX, 10 mức ngưỡng drop kể cả chế độ tắt cắt tùy chọn, 15 cặp ngưỡng overlap, số câu overlap, làm tròn và ablation. Một lưới kết hợp nhỏ gồm 81 cấu hình được xác định trước khi chạy. Không gian tìm kiếm cùng tiêu chí được lưu trong protocol, trước load model và chấm điểm; đây là đóng băng giao thức cục bộ, không phải đăng ký nghiên cứu công khai.

Tiêu chí chính là trung bình bằng trọng số giữa Evidence Recall@5 của hai góc nhìn trên cùng 233 câu dev. MRR@5 trung bình là tiêu chí phụ. Lưới trọng số dùng \(\alpha\in\{0,0.1,\ldots,0.8\}\), \(\beta=0.8-\alpha\) và \(\gamma\in\{0,0.2,0.4,0.6,0.8,1.2\}\). Giữ tổng hai trọng số dương ở 0,8 giúp tránh một phần cấu hình tương đương do co giãn toàn bộ score. Tập tham chiếu để thay từng nhóm tham số sử dụng \((\alpha,\beta,\gamma)=(0.4,0.4,0.6)\) và các ngưỡng ban đầu.

Sau nghiên cứu này, cấu hình triển khai chỉ cập nhật tỷ lệ trọng số thành 0,5/0,3 và giữ penalty 0,6 trong số các mức đồng hạng; các ngưỡng vẫn theo cấu hình tham chiếu. Lựa chọn này giữ nguyên ràng buộc độ dài của implementation khi chuyển sang tỷ lệ trọng số được chọn. Nó không tương đương lựa chọn cấu hình tốt nhất trong toàn bộ 180 thử nghiệm: một cấu hình thay MIN đạt dev cao hơn. Vì vậy bài báo không tuyên bố đã tối ưu đồng thời mọi tham số.

Việc chọn tham số trên dev được tách khỏi phép xếp hạng cấu hình trên test. Tuy nhiên, nghiên cứu độ nhạy được thiết kế sau khi kết quả của cấu hình tham chiếu trên test đã được xem. Cấu hình triển khai được đóng băng trước lần chạy test mới, nhưng tập này không còn là nguồn xác nhận hoàn toàn chưa quan sát. Các kết luận tổng quát cần được xác nhận trên corpus độc lập mới.

### 4.5 Thống kê và tái lập

Chênh lệch hai phương pháp được phân tích ghép cặp theo câu hỏi. Khoảng 95% được ước lượng bằng bootstrap percentile 5.000 lượt, seed 2026, với đơn vị lấy mẫu là 38 bài nguồn test. Khi một bài được lấy mẫu, toàn bộ câu hỏi của bài đó được giữ cùng nhau cho cả hai phương pháp. Cách làm này phản ánh tương quan nội bộ nguồn tốt hơn việc mặc nhiên coi 957 câu hỏi là độc lập. Hai góc nhìn được phân tích riêng, không được xem như hai tập mẫu độc lập.

Lần chạy phục vụ bản thảo sử dụng CPU với bốn thread và seed 2026. Python cùng phiên bản các thư viện được ghi trực tiếp trong provenance. Thời gian được đo sau warm-up, tách bước chunking, embedding/index và ranking; thứ tự phương pháp là Max–Min rồi phương pháp đề xuất. Đây là một lượt chạy, chưa có kiểm soát tải máy hoặc độ lệch chuẩn nhiều lượt. Bản thảo không suy diễn phần cứng của các lần chạy trước từ máy hiện tại.

## 5 Kết quả

### 5.1 Chất lượng truy xuất với cấu hình triển khai

**Bảng 3. Kết quả truy xuất trên 957 câu hỏi với trọng số semantic/size/risk = 0,5/0,3/0,6 của phương pháp đề xuất.**

__RESULT_TABLE__

Với context gốc, phương pháp đề xuất cao hơn Max–Min __P_DELTA__ điểm phần trăm Evidence Recall@5. Với tài liệu ghép dài, chênh lệch là __A_DELTA__ điểm phần trăm. MRR@5 thay đổi cùng chiều với chênh lệch Recall trong từng góc nhìn. Kết quả cho thấy thay đổi đơn vị tài liệu có thể đảo chiều so sánh, dù nội dung và câu hỏi được giữ nguyên.

**Bảng 4. So sánh ghép cặp Evidence Recall@5 giữa phương pháp đề xuất và Max–Min. Chênh lệch tính theo điểm phần trăm.**

__PAIRED_TABLE__

Các khoảng bootstrap mô tả độ bất định theo nguồn của phép so sánh trên benchmark này. Chúng không khắc phục lỗi nhãn, sự hạn chế về miền dữ liệu hoặc việc tập test đã được quan sát trong giai đoạn nghiên cứu trước. Khoảng không chứa 0 cũng không tự chứng minh quan hệ nhân quả giữa một thành phần cụ thể của thuật toán và mức thay đổi metric.

Phương pháp đề xuất bảo toàn evidence theo nhãn hiện có ở cả hai góc nhìn. Với context gốc có __P_MISS__ câu hỏi của phương pháp đề xuất thuộc nhóm ranking_miss; với tài liệu ghép dài con số là __A_MISS__. Ngược lại, Max–Min có __BASE_SPLIT__ câu hỏi evidence_split ở mỗi góc nhìn. Như vậy, giữ evidence trong corpus và xếp chunk chứa evidence vào top-5 là hai yêu cầu khác nhau. Bộ nhãn evidence chưa đầy đủ supporting facts cũng giới hạn ý nghĩa của tỷ lệ bảo toàn 100%.

### 5.2 Độ nhạy của trọng số final score

**Bảng 5. Thay đổi tỷ lệ semantic/size trên dev, giữ risk = 0,6 và các tham số khác theo cấu hình tham chiếu.**

__WEIGHT_TABLE__

Tỷ lệ 0,5/0,3 đạt điểm cao nhất theo tiêu chí chính trong nhóm trọng số, với Evidence Recall@5 trung bình __SELECTED_DEV__, so với __REFERENCE_DEV__ của tỷ lệ 0,4/0,4. Kết quả không tăng đơn điệu theo trọng số semantic. Khi tăng quá mức, Recall không tiếp tục cải thiện. Bộ có MRR cao nhất trong bảng cũng khác bộ được chọn theo Recall; do đó phát biểu “tốt nhất” cần gắn với tiêu chí đã xác định.

Với tỷ lệ 0,5/0,3, sáu mức penalty được thử đồng hạng cả Recall và MRR. Ở cấu hình tham chiếu, paragraph dev không có ứng viên risk hợp lệ trong sáu lượt scoring; article dev chỉ có một ứng viên risk trong 42 lượt scoring. Sự thiếu vắng tình huống kích hoạt có thể làm nhiều mức penalty không phân biệt được. Vì vậy nghiên cứu không xác lập ưu thế riêng của 0,6, dù vẫn giữ giá trị này khi triển khai. Cần tập văn bản trên MAX với nhãn liên kết do người rà soát và ablation độc lập để kiểm tra hiệu quả của luật.

### 5.3 Độ nhạy của ngưỡng và các thành phần

**Bảng 6. Một số lựa chọn tham số đối chứng trên dev. Mỗi dòng chỉ thay phần được nêu so với cấu hình tham chiếu 0,4/0,4/0,6.**

__SENSITIVITY_TABLE__

Giảm riêng MIN xuống 60 tạo cấu hình đứng đầu trong 180 lựa chọn đã thử, nhưng không được áp dụng vào cấu hình triển khai của bản thảo này. Ngưỡng raw drop 0,20 có điểm dev cao hơn 0,30 trong lưới; nhiều ngưỡng lân cận lại tạo cùng kết quả. Những quan sát này cho thấy 120/180/240 và 0,30 là các giá trị vận hành tham chiếu, chưa được chứng minh là tối ưu. Lưới hữu hạn cũng không cho biết nghiệm tốt nhất nằm giữa các mức đã thử hay không.

Các cặp ngưỡng overlap có thể đồng hạng Recall nhưng khác MRR hoặc lượng context. Việc lấy nhiều câu overlap hơn không đồng nghĩa truy xuất tốt hơn. Với score đã chuẩn hóa, thay số chữ số làm tròn size score cũng có thể đổi thứ hạng ứng viên ở các trường hợp gần nhau; lựa chọn hai chữ số chưa có ưu thế duy nhất trong nghiên cứu dev.

**Bảng 7. Ablation trên cấu hình tham chiếu của nghiên cứu dev.**

__ABLATION_TABLE__

“Bỏ semantic score” chỉ bỏ thành phần tương ứng trong F, không loại cắt tùy chọn theo raw drop. “Bỏ size score” vẫn giữ điều kiện MIN/MAX. Vì vậy các dòng ablation đo ảnh hưởng của một thành phần trong cơ chế hiện tại, không đại diện cho một thuật toán hoàn toàn không có ngữ nghĩa hoặc hoàn toàn không có kiểm soát độ dài. Các phép ablation này thuộc cấu hình tham chiếu; không được gán thành ablation của trọng số triển khai mới.

### 5.4 Kích thước chunk và ngân sách context

**Bảng 8. Độ dài chunk, lượng context top-5 và tỷ lệ vượt giới hạn encode trong lần chạy hiện tại.**

__LENGTH_TABLE__

Context top-5 của phương pháp đề xuất lớn hơn Max–Min __P_RATIO__ lần với context gốc và __A_RATIO__ lần với tài liệu ghép dài. Do đó lợi thế Recall trong một góc nhìn đi kèm việc lấy nhiều văn bản hơn. Số chunk ít hơn cũng không trực tiếp đồng nghĩa chất lượng xếp hạng cao hơn.

Độ dài token ở Bảng 8 được đo bằng tokenizer của encoder, có special tokens và không truncation. Nhiều chunk dài hơn 128 token nên encoder có thể không sử dụng toàn bộ văn bản khi tạo vector. Tuy nhiên, tỷ lệ chunk vượt giới hạn không phải tỷ lệ evidence bị mất: nghiên cứu chưa ánh xạ vị trí evidence sang vùng token được giữ cho mọi lần xuất hiện. Tác động của truncation lên chênh lệch retrieval vì vậy vẫn là giả thuyết cần đối chứng.

Để khảo sát độ nhạy với chính sách context, chúng tôi áp trần 512 và 1024 token lên ranking top-5 đã lưu. Context được tạo bằng cách nối nguyên chunk theo thứ hạng, dừng trước chunk đầu tiên gây vượt trần, không bỏ qua, không cắt ngắn và không lấy thêm ứng viên ngoài top-5. Token budget chỉ đếm nội dung context, không bao gồm special tokens, câu hỏi hoặc prompt, và dùng tokenizer MiniLM thay vì tokenizer của một LLM sinh đáp án.

**Bảng 9. Truy xuất dưới trần token áp lên tiền tố top-5.**

__BUDGET_TABLE__

Phép đo này mô tả ảnh hưởng của trần token đối với chính sách top-5 hiện tại. Nó chưa phải so sánh fixed-budget trên toàn ranking: số ứng viên giới hạn ở năm và hai phương pháp có thể sử dụng số token thực tế khác nhau. Không chọn lại budget, threshold hoặc trọng số theo kết quả thắng/thua trên test.

### 5.5 Thời gian xử lý

**Bảng 10. Thời gian một lượt chạy sau warm-up, đơn vị giây.**

__TIME_TABLE__

Chunking của phương pháp đề xuất bao gồm encode câu cần thiết để tính ranh giới; embedding/index là encode các chunk đầu ra và xây chỉ mục. Chi phí encode query được chia sẻ giữa hai phương pháp trong mỗi góc nhìn. Các số ở Bảng 10 không có độ lệch chuẩn nhiều lượt, chịu ảnh hưởng cache, thứ tự chạy và tải hệ thống; vì vậy chỉ mô tả lần chạy đã thực hiện, không hỗ trợ kết luận ưu thế tốc độ có ý nghĩa thống kê. Bộ nhớ CPU/GPU chưa được đo.

### 5.6 Thảo luận và giới hạn

Kết quả cho thấy chiến lược đề xuất có thể hỗ trợ truy xuất khi tài liệu đã được tổ chức thành context tương đối ngắn, đồng thời bảo toàn evidence theo tiêu chí đang dùng. Với tài liệu ghép dài, lợi thế bảo toàn evidence không chuyển thành ưu thế retrieval. Một giả thuyết là đoạn lớn và overlap làm biểu diễn ít tập trung hơn vào chi tiết câu hỏi; truncation có thể làm mất một phần nội dung khỏi biểu diễn. Những cơ chế này chưa được xác lập nhân quả vì chưa có đối chứng tách riêng giới hạn encoder, kích thước chunk và vị trí evidence.

Benchmark có các hạn chế về giá trị suy rộng. Chỉ 38 bài nguồn thuộc tập đánh giá; các lĩnh vực là nhóm rộng do nghiên cứu gán. Văn bản dài được ghép từ đoạn trích cùng chủ đề, không phải corpus tài liệu dài tự nhiên hoàn chỉnh. Evidence được trích tự động từ đáp án đầu tiên, có câu hỏi lặp và nhãn nguồn đáng nghi. Chấm đúng tài liệu và đủ một evidence mục tiêu không đo mọi chứng cứ hợp lệ có thể trả lời câu hỏi.

Việc lựa chọn trọng số mới diễn ra sau khi nhóm đã quan sát kết quả test của cấu hình tham chiếu. Dù lần chạy hiện tại đóng băng tham số và không dùng test để xếp hạng các ứng viên mới, nó chưa thay thế xác nhận trên dữ liệu độc lập. Nghiên cứu độ nhạy mới gồm 10 nhóm bài dev, nên các khác biệt nhỏ cần được xem xét cùng độ bất định và thiên lệch do thử nhiều cấu hình. Giữ một ngưỡng chưa có bằng chứng tối ưu phải được hiểu là quyết định triển khai có phạm vi, không phải kết quả tối ưu hóa toàn cục.

Cuối cùng, nghiên cứu chưa chạy sinh đáp án cho bộ đánh giá này, nên không báo QA Accuracy, EM/F1 câu trả lời hoặc mức giảm hallucination. XQuAD chuyển đổi ở đây cũng không cung cấp nhãn ranh giới hoặc dependency đầy đủ; các số retrieval không thể thay thế Boundary F1 hay đánh giá riêng chất lượng luật liên kết câu.

## 6 Kết luận

Bài báo trình bày một chiến lược phân đoạn tiếng Việt kết hợp semantic drop, độ dài, luật liên kết câu và overlap động trong cùng pipeline truy xuất. Nghiên cứu dev hỗ trợ lựa chọn tỷ lệ trọng số semantic/size = 0,5/0,3 trong lưới đã thử; penalty 0,6 được giữ trong nhóm đồng hạng. Các threshold và độ dài tham chiếu chưa được chứng minh tối ưu đồng thời.

Trên 957 câu hỏi đánh giá với cấu hình triển khai hiện tại, phương pháp đề xuất đạt Evidence Recall@5 __P_OURS__ với context gốc và __A_OURS__ với tài liệu ghép dài; Max–Min lần lượt đạt __P_BASE__ và __A_BASE__. Lợi thế thay đổi theo cách tổ chức đầu vào, trong khi phương pháp đề xuất sử dụng nhiều context hơn. Vì vậy, bằng chứng hiện tại hỗ trợ việc xem xét đồng thời bảo toàn evidence, xếp hạng retrieval và ngân sách ngữ cảnh, thay vì lựa chọn phương pháp chỉ từ một bảng top-k.

Hướng nghiên cứu tiếp theo là đánh giá trên corpus độc lập mới, lựa chọn context theo budget trên toàn ranking, đo offset evidence trong vùng token được encode, xây tập dependency có văn bản trên MAX và thực hiện ablation của cấu hình triển khai. Đánh giá đầu cuối bằng cùng LLM, prompt và rubric cần được thực hiện riêng để xác định ảnh hưởng đến chất lượng câu trả lời.

## Khả dụng của dữ liệu và mã nguồn

Dữ liệu XQuAD tiếng Việt cùng giấy phép và thông tin chuyển đổi nằm trong [thư mục benchmark](data/benchmark/xquad_vi/README.md). Manifest lưu commit, checksum, cách chia nguồn và quy mô từng split. Code phương pháp đề xuất nằm trong [bộ chấm điểm ranh giới](src/boundary_scorer.py) và [bộ điều phối chunking](src/semantic_chunker.py); đối chứng có [implementation Max–Min](baselines/maxmin_semantic_chunker.py).

Các bảng trong Phần 5.1 và 5.4–5.5 được lấy từ **lần chạy với code hiện tại**, lưu tại [kết quả phục vụ bản thảo](evaluation/results/scientific_paper_current/protocol.json), gồm JSON/CSV từng góc nhìn, ranking từng câu hỏi, bootstrap và chẩn đoán token. Thí nghiệm được tái lập bằng `scripts/run_paper_experiments.py`; bản thảo được sinh từ artifact bằng `scripts/write_scientific_paper.py`. Script không thay trọng số trong lúc đánh giá.

Bảng độ nhạy và ablation dev lấy từ [protocol 180 cấu hình](evaluation/results/parameter_study/protocol.json), [kết quả từng góc nhìn](evaluation/results/parameter_study/sensitivity.csv) và [bảng tổng hợp](evaluation/results/parameter_study/aggregate.csv). [Phụ lục tham số](THAM_SO_VA_MINH_CHUNG.md) lưu toàn bộ các mức thử và giải trình các hằng số. Các artifact test lịch sử của cấu hình 0,4/0,4/0,6 được giữ riêng để truy vết; chúng không được sử dụng thay cho các bảng test của trọng số mới trong bản thảo này.

## Tài liệu tham khảo

1. Lewis P, Perez E, Piktus A, và cộng sự. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*. 2020;33:9459–9474. [Toàn văn](https://arxiv.org/abs/2005.11401).
2. Kiss C, Nagy M, Szilágyi P. Max–Min semantic chunking of documents for RAG application. *Discover Computing*. 2025;28:117. [DOI](https://doi.org/10.1007/s10791-025-09638-7).
3. Reimers N, Gurevych I. Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. *Proceedings of EMNLP*. 2020:4512–4525. [ACL Anthology](https://aclanthology.org/2020.emnlp-main.365/).
4. Artetxe M, Ruder S, Yogatama D. On the Cross-lingual Transferability of Monolingual Representations. *Proceedings of ACL*. 2020:4623–4637. [ACL Anthology](https://aclanthology.org/2020.acl-main.421/).
5. LlamaIndex. SemanticSplitterNodeParser. Mã nguồn phiên bản 0.14.25. [Implementation](https://github.com/run-llama/llama_index/blob/v0.14.25/llama-index-core/llama_index/core/node_parser/text/semantic_splitter.py). Truy cập ngày 16/11/2026.
6. Qu R, Tu R, Bao F. Is Semantic Chunking Worth the Computational Cost? *arXiv preprint*. 2024. [arXiv:2410.13070](https://arxiv.org/abs/2410.13070).
7. Sentence Transformers. paraphrase-multilingual-MiniLM-L12-v2. [Model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2); [cấu hình snapshot được sử dụng](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/raw/e8f8c211226b894fcb81acc59f3b34ba3efd5f42/sentence_bert_config.json). Truy cập ngày 16/11/2026.
8. HSDSLab. MaxMinChunking. Mã tác giả tại commit `0caba41d2ef30114729f95e5c788173e4f40920e`. [Implementation](https://github.com/hsdslab/MaxMinChunking/blob/0caba41d2ef30114729f95e5c788173e4f40920e/maxmin_chunker.py). Truy cập ngày 16/11/2026.
9. Google DeepMind. XQuAD. Dữ liệu nguồn tại commit `7d30520c717524000f0d9d2f9c10a069acd9d285`. [Kho dữ liệu](https://github.com/google-deepmind/xquad/tree/7d30520c717524000f0d9d2f9c10a069acd9d285). Truy cập ngày 16/11/2026.
'''


def main():
    reports={view:read(RUN/f'xquad_vi_{view}_test/rag_comparison.json') for view in ('paragraph','article')}
    summaries={view:read(RUN/f'xquad_vi_{view}_test/rag_statistical_summary.json') for view in reports}
    methods={view:{r['method']:r for r in report['results']} for view,report in reports.items()}
    diagnostic=read(RUN/'diagnostics.json')
    study=read(STUDY/'summary.json')
    with (STUDY/'aggregate.csv').open(encoding='utf-8-sig',newline='') as stream:
        aggregate=list(csv.DictReader(stream))
    protocol=read(RUN/'protocol.json')
    assert protocol['parameters']['semantic_weight']==.5 and protocol['parameters']['size_weight']==.3
    for report in reports.values():
        assert report['configuration']['ours_parameters']==protocol['parameters']
        for path,digest in report['provenance']['chunking_sources'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    def name(view):
        return 'Context gốc' if view=='paragraph' else 'Ghép dài'
    def method(label):
        return 'Max–Min' if label=='Max-Min' else 'Đề xuất'
    values={}
    for prefix,view in (('P','paragraph'),('A','article')):
        values[prefix+'_OURS']=percent(methods[view]['Ours']['recall_at_5'])
        values[prefix+'_BASE']=percent(methods[view]['Max-Min']['recall_at_5'])
        values[prefix+'_DELTA']=decimal(100*summaries[view]['metrics']['recall_at_5']['delta'])
        values[prefix+'_MISS']=str(summaries[view]['failure_counts']['Ours'].get('ranking_miss',0))
        values[prefix+'_RATIO']=decimal(methods[view]['Ours']['mean_context_words']/methods[view]['Max-Min']['mean_context_words'])
    values['BASE_SPLIT']=str(summaries['paragraph']['failure_counts']['Max-Min'].get('evidence_split',0))
    manifest=read(ROOT/'data/benchmark/xquad_vi/manifest.json')
    values['DATA_TABLE']=table(['Góc nhìn','Tập','Tài liệu','Câu hỏi','Bài nguồn'],[
        [name(view),split,manifest['summaries'][f'{view}/{split}']['documents'],
         manifest['summaries'][f'{view}/{split}']['questions'],
         manifest['summaries'][f'{view}/{split}']['independent_source_titles']]
        for view in ('paragraph','article') for split in ('dev','test')])
    packages=reports['paragraph']['provenance']['packages']
    values['CONFIG_TABLE']=table(['Thành phần','Giá trị'],[
        ['Encoder','paraphrase-multilingual-MiniLM-L12-v2'],['Độ dài encoder tối đa','128 token'],
        ['Độ dài m/t/M','120/180/240 đơn vị khoảng trắng'],['Trọng số semantic/size/risk','0,5/0,3/0,6'],
        ['Ngưỡng raw drop','0,30'],['Ngưỡng overlap thấp/cao','0,30/0,55'],['Số câu overlap','0–2'],
        ['Truy xuất','Cosine toàn corpus, top-5, thứ tự ổn định khi bằng điểm'],
        ['Python',reports['paragraph']['provenance']['python']],
        ['sentence-transformers / PyTorch',packages['sentence-transformers']+' / '+packages['torch']],
        ['NumPy / scikit-learn',packages['numpy']+' / '+packages['scikit-learn']],
        ['Sinh câu trả lời','Không thực hiện']])
    values['RESULT_TABLE']=table(['Góc nhìn','Phương pháp','Số chunk','Evidence Recall@3','Evidence Recall@5','MRR@5'],[
        [name(view),method(label),r['num_chunks'],percent(r['recall_at_3']),percent(r['recall_at_5']),decimal(r['mrr_at_5'],4)]
        for view in reports for label,r in methods[view].items()])
    paired=[]
    for view,summary in summaries.items():
        metric=summary['metrics']['recall_at_5']
        counts=summary['paired_hit_counts_at_5']
        paired.append([name(view),decimal(100*metric['delta']),
                       '['+decimal(100*metric['delta_ci95'][0])+'; '+decimal(100*metric['delta_ci95'][1])+']',
                       counts['only_first'],counts['only_second']])
    values['PAIRED_TABLE']=table(['Góc nhìn','Đề xuất − Max–Min','Khoảng bootstrap 95%','Chỉ Max–Min hit','Chỉ đề xuất hit'],paired)
    weights=sorted([r for r in aggregate if 'weights' in r['families'].split(';') and float(r['risk_weight'])==.6],key=lambda r:float(r['semantic_weight']))
    values['WEIGHT_TABLE']=table(['Semantic','Size','Risk','Evidence Recall@5 TB','MRR@5 TB'],[
        [decimal(r['semantic_weight'],1),decimal(r['size_weight'],1),decimal(r['risk_weight'],1),percent(r['mean_recall_at_5']),decimal(r['mean_mrr_at_5'],4)] for r in weights])
    selected=next(r for r in weights if float(r['semantic_weight'])==.5)
    values['SELECTED_DEV']=percent(selected['mean_recall_at_5'])
    values['REFERENCE_DEV']=percent(study['default']['mean_recall_at_5'])
    variants=[('Cấu hình tham chiếu',study['default_config_id']),('MIN = 60',study['by_family']['minimum']['best_config_id']),
              ('Raw drop = 0,20',study['by_family']['semantic_drop']['best_config_id']),
              ('Overlap thấp/cao = 0,15/0,85',study['by_family']['overlap_thresholds']['best_config_id']),
              ('Overlap cố định 1 câu',study['by_family']['fixed_overlap']['best_config_id'])]
    values['SENSITIVITY_TABLE']=table(['Thay đổi','Evidence Recall@5 TB','MRR@5 TB','Context TB'],[
        [label,percent(row['mean_recall_at_5']),decimal(row['mean_mrr_at_5'],4),decimal(row['mean_context_words'])]
        for label,identifier in variants for row in aggregate if row['config_id']==identifier])
    ablation_names={'without_size':'Bỏ size score','without_semantic_score':'Bỏ semantic score',
                    'without_risk':'Bỏ risk penalty','without_overlap':'Không overlap',
                    'without_optional_cut':'Tắt cắt tùy chọn'}
    values['ABLATION_TABLE']=table(['Thành phần thay đổi','Evidence Recall@5 TB','MRR@5 TB'],[
        [label,percent(row['mean_recall_at_5']),decimal(row['mean_mrr_at_5'],4)]
        for suffix,label in ablation_names.items() for row in aggregate if 'ablation_'+suffix in row['families'].split(';')])
    length_rows=[]
    for view in reports:
        for label,r in methods[view].items():
            tok=next(t for t in diagnostic['tokenizer_stats'] if t['view']==view and t['method']==label)
            length_rows.append([name(view),method(label),decimal(r['avg_chunk_words']),decimal(r['mean_context_words']),
                                decimal(tok['mean']),percent(tok['fraction_over_128'])])
    values['LENGTH_TABLE']=table(['Góc nhìn','Phương pháp','Độ dài chunk TB (khoảng trắng)',
        'Context top-5 TB (khoảng trắng)','Token/chunk TB','Chunk >128 token'],length_rows)
    values['BUDGET_TABLE']=table(['Góc nhìn','Phương pháp','Trần token','Evidence Recall','MRR','Token thực dùng TB'],[
        [name(r['view']),method(r['method']),r['budget'],percent(r['evidence_recall']),decimal(r['mrr'],4),decimal(r['mean_context_tokens'])]
        for r in diagnostic['budget_results']])
    values['TIME_TABLE']=table(['Góc nhìn','Phương pháp','Chunking (s)','Embedding/index (s)','Ranking (s)'],[
        [name(view),method(label),decimal(r['chunking_time'],3),decimal(r['embedding_time'],3),decimal(r['retrieval_time'],3)]
        for view in reports for label,r in methods[view].items()])
    text=PAPER
    for key,value in values.items():
        text=text.replace('__'+key+'__',value)
    assert '__' not in text, 'Unresolved manuscript placeholder'
    (ROOT/'BAO_CAO.md').write_text(text.strip()+'\n',encoding='utf-8')
    write_manifest=dict(manuscript_sha256=hashlib.sha256((ROOT/'BAO_CAO.md').read_bytes()).hexdigest(),
        manuscript='BAO_CAO.md',structure_reference='https://link.springer.com/article/10.1007/s10791-025-09638-7',
        production_parameters=protocol['parameters'],current_test_outputs=True,
        evidence_inputs_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [RUN/'protocol.json',RUN/'diagnostics.json',STUDY/'aggregate.csv',STUDY/'summary.json',
             *[RUN/f'xquad_vi_{view}_test/rag_comparison.json' for view in reports],
             *[RUN/f'xquad_vi_{view}_test/rag_statistical_summary.json' for view in reports]]})
    (RUN/'manuscript_evidence.json').write_text(json.dumps(write_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Wrote BAO_CAO.md as a Vietnamese research manuscript with current-run evidence.')


if __name__=='__main__':
    main()
