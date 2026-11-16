"""Build the Vietnamese evidence appendix from verified experiment artifacts."""

import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/results/parameter_study'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def csv_read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def pct(value):
    return f'{100*float(value):.2f}%'.replace('.',',')


def number(value):
    return f'{float(value):.4f}'.replace('.',',')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def main():
    summary=read('summary.json')
    verified=read('verification.json')
    rows=csv_read('sensitivity.csv')
    aggregate=csv_read('aggregate.csv')
    default=summary['default']
    default_id=summary['default_config_id']
    intervals=read('dev_exploratory_intervals.json')['comparisons']

    def member(row,family):
        return family in row['families'].split(';')

    def config_label(row):
        return (f"MIN/TARGET/MAX={row['minimum']}/{row['target']}/{row['maximum']}; "
                f"semantic/size/risk={row['semantic_weight']}/{row['size_weight']}/{row['risk_weight']}; "
                f"drop={row['drop']}; overlap={row['overlap_low']}/{row['overlap_high']}")

    blocks=[r'''# Giải trình tham số và minh chứng thực nghiệm

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
''']
    selected=[('Mặc định',default_id),('Tốt nhất trong 180 cấu hình',summary['best']['config_id']),
              ('Tốt nhất nhóm trọng số',summary['by_family']['weights']['best_config_id']),
              ('Tốt nhất nhóm ngưỡng drop',summary['by_family']['semantic_drop']['best_config_id'])]
    records=[]
    for name,identifier in selected:
        record=next(r for r in aggregate if r['config_id']==identifier)
        pair={r['view']:r for r in rows if r['config_id']==identifier}
        records.append([name,pct(pair['paragraph']['recall_at_5']),pct(pair['article']['recall_at_5']),
                        pct(record['mean_recall_at_5']),number(record['mean_mrr_at_5']),
                        f"{float(record['mean_context_words']):.2f}",identifier])
    blocks.append(table(['Cấu hình','Recall@5 paragraph','Recall@5 article','Recall@5 TB','MRR@5 TB','Context TB','ID'],records))
    blocks.append('\nCấu hình có điểm cao nhất trong 180 cấu hình: '+config_label(summary['best'])+'. Những giá trị không nêu thay đổi vẫn theo mặc định. Context đếm theo khoảng trắng, gồm overlap; không phải số token LLM.\n')
    delta=intervals['best_tested']['hit_at_5']
    blocks.append(f"Chênh lệch Recall@5 TB giữa cấu hình thắng và mặc định là {100*delta['delta']:.3f} điểm phần trăm. Khoảng bootstrap nguồn 95% thăm dò: [{100*delta['ci95'][0]:.3f}; {100*delta['ci95'][1]:.3f}] điểm phần trăm. Đã chọn cấu hình bằng chính dev nên khoảng này không phải kiểm định xác nhận độc lập, không hiệu chỉnh việc thử nhiều cấu hình. Chỉ có 10 nhóm bài nguồn.\n")

    blocks.append('### 4.1. Trọng số final score\n')
    weights=sorted([r for r in aggregate if member(r,'weights') and float(r['risk_weight'])==.6],key=lambda r:float(r['semantic_weight']))
    blocks.append(table(['Semantic','Size','Risk','Recall@5 TB','MRR@5 TB'],
                        [[r['semantic_weight'],r['size_weight'],r['risk_weight'],pct(r['mean_recall_at_5']),number(r['mean_mrr_at_5'])] for r in weights]))
    tie_ids=verified['family_tied_best_config_ids']['weights']
    ties=[r for r in aggregate if r['config_id'] in tie_ids]
    blocks.append('\nCác bộ đồng hạng tốt nhất trong nhóm trọng số: '+ '; '.join(
        f"({r['semantic_weight']}; {r['size_weight']}; {r['risk_weight']})" for r in ties)+'. Thứ tự là semantic, size, risk. Không được lấy bộ đầu tiên theo ID làm bằng chứng penalty đó tốt hơn các penalty đồng hạng.\n')
    default_pair={r['view']:r for r in rows if r['config_id']==default_id}
    blocks.append(f"Với cấu hình mặc định, paragraph có {default_pair['paragraph']['scoring_calls']} lượt scoring và {default_pair['paragraph']['risky_candidates']} ứng viên risk; article có {default_pair['article']['scoring_calls']} lượt scoring và {default_pair['article']['risky_candidates']} ứng viên risk. Khi giữ semantic=size=0,4, sáu mức penalty 0/0,2/0,4/0,6/0,8/1,2 tạo cùng corpus ở từng góc nhìn. Không đủ tín hiệu để xác định 0,6 là tốt nhất. Cần văn bản trên MAX có nhãn dependency do người rà soát; không tự lấy luật hiện tại làm gold để chứng minh chính luật đó.\n")

    blocks.append('### 4.2. Độ dài MIN TARGET MAX\n')
    sizes=[r for r in aggregate if any(member(r,f) for f in ('minimum','target','maximum','size_scale'))]
    sizes.sort(key=lambda r:(int(r['minimum']),int(r['target']),int(r['maximum'])))
    blocks.append(table(['MIN','TARGET','MAX','Recall@5 TB','MRR@5 TB'],[[r['minimum'],r['target'],r['maximum'],pct(r['mean_recall_at_5']),number(r['mean_mrr_at_5'])] for r in sizes]))
    blocks.append('\n120/180/240 đứng đầu trong các bộ co giãn cùng tỷ lệ 2:3:4 đã thử, nhưng không đứng đầu mọi lựa chọn độ dài. Giảm riêng MIN xuống 60 cho điểm dev cao hơn. Vì vậy chỉ có thể bảo vệ 120/180/240 như một cấu hình ban đầu có kết quả khả quan trong một dải kiểm tra, không phải bộ tối ưu. Đơn vị độ dài là phần tử cách nhau bằng khoảng trắng, không phải subword token. MAX giới hạn ứng viên phần lõi, không bảo đảm giới hạn cứng sau overlap hoặc fallback.\n')

    blocks.append('### 4.3. Ngưỡng semantic drop\n')
    drops=sorted([r for r in aggregate if member(r,'semantic_drop')],key=lambda r:float(r['drop']))
    blocks.append(table(['Ngưỡng raw drop','Recall@5 TB','MRR@5 TB'],[[r['drop'] if float(r['drop'])<3 else 'Tắt optional cut (3)',pct(r['mean_recall_at_5']),number(r['mean_mrr_at_5'])] for r in drops]))
    blocks.append('\nRaw drop có miền lý thuyết [−2,2] do cosine thuộc [−1,1], nên ngưỡng 3 là chế độ đối chứng tắt optional cut. Ngưỡng 0,20 có điểm dev cao hơn 0,30 trong lưới này; không thể kết luận 0,30 tối ưu. Chỉ nhánh phần còn lại từ MIN đến MAX dùng ngưỡng raw drop; nhánh trên MAX dùng điểm tổng hợp và min–max normalization.\n')

    blocks.append('### 4.4. Ngưỡng và số câu overlap\n')
    overlaps=sorted([r for r in aggregate if member(r,'overlap_thresholds')],key=lambda r:(float(r['overlap_low']),float(r['overlap_high'])))
    blocks.append(table(['Ngưỡng thấp','Ngưỡng cao','Recall@5 TB','MRR@5 TB','Context TB'],
        [[r['overlap_low'],r['overlap_high'],pct(r['mean_recall_at_5']),number(r['mean_mrr_at_5']),f"{float(r['mean_context_words']):.2f}"] for r in overlaps]))
    counts=sorted([r for r in aggregate if member(r,'overlap_cap') or member(r,'fixed_overlap')],key=lambda r:(r['fixed_overlap'],r['overlap_cap']))
    blocks.append('\n'+table(['Cách overlap','Recall@5 TB','MRR@5 TB','Context TB'],
        [[f"Cố định {r['fixed_overlap']} câu" if r['fixed_overlap'] else f"Động 0/1/{r['overlap_cap']} câu",pct(r['mean_recall_at_5']),number(r['mean_mrr_at_5']),f"{float(r['mean_context_words']):.2f}"] for r in counts]))
    blocks.append('\nNhiều cặp ngưỡng đồng hạng Recall@5 nhưng khác MRR hoặc context. Cặp 0,30/0,55 và tối đa 2 câu không tối ưu duy nhất. Overlap cố định một câu và giới hạn động một câu là những đối chứng quan trọng; không thể kết luận “overlap động luôn tốt hơn”. Lượng context và giới hạn encode phải được xét cùng điểm retrieval.\n')

    blocks.append('### 4.5. Làm tròn và ablation\n')
    rounding=sorted([r for r in aggregate if member(r,'size_rounding')],key=lambda r:r['size_rounding'])
    blocks.append(table(['Chữ số thập phân size score','Recall@5 TB','MRR@5 TB'],
        [[r['size_rounding'] or 'Không làm tròn',pct(r['mean_recall_at_5']),number(r['mean_mrr_at_5'])] for r in rounding]))
    ablations=[r for r in aggregate if 'ablation_' in r['families']]
    blocks.append('\n'+table(['Ablation','Recall@5 TB','MRR@5 TB'],
        [[', '.join(f.removeprefix('ablation_') for f in r['families'].split(';') if f.startswith('ablation_')),pct(r['mean_recall_at_5']),number(r['mean_mrr_at_5'])] for r in ablations]))
    blocks.append('\nAblation “without_semantic_score” chỉ bỏ thành phần semantic trong final score; optional semantic cut vẫn tồn tại. “without_size” chỉ bỏ size score, vẫn giữ điều kiện MIN/MAX. Các tên này không được diễn giải thành một thuật toán hoàn toàn không dùng ngữ nghĩa hoặc hoàn toàn không có ràng buộc độ dài.\n')

    blocks.append('### 4.6. Số chunk truy xuất\n')
    k_rows=csv_read('retrieval_k.csv')
    blocks.append(table(['Góc nhìn','k','Evidence Recall@k','MRR@k','Context TB'],
        [[r['view'],r['k'],pct(r['recall']),number(r['mrr']),f"{float(r['mean_context_words']):.2f}"] for r in k_rows]))
    blocks.append('\nBảng dùng Ours mặc định trên dev. Tăng k từ 5 lên 10 tăng Recall nhưng làm lượng context gần gấp đôi; chưa có hàm mục tiêu về chi phí hoặc đánh giá generation để xác định k tối ưu. Top-5 là điểm vận hành đã chọn cho đối chứng, không phải cực đại Recall. File nguồn: evaluation/results/parameter_study/retrieval_k.csv.\n')

    blocks.append('## 5. Giải trình các số còn lại\n')
    blocks.append(table(['Con số','Căn cứ và trạng thái','Có được gọi là tối ưu không'],[
        ['0/1 trong risk; 0/1 trong normalization','Định nghĩa cờ khớp luật và miền chuẩn hóa, không phải xác suất được học','Không đặt vấn đề tối ưu độ lớn tuyệt đối'],
        ['2 trong trung bình semantic drop; 4 câu lân cận','Thiết kế đối xứng trước/qua/sau boundary, đủ ngữ cảnh để tính hai liên kết nội bộ','Chưa so sánh mọi loại cửa sổ khác'],
        ['512 câu cache','Giới hạn bộ nhớ LRU trong production; chỉ ảnh hưởng tái sử dụng tính toán','Không phải tham số quyết định chất lượng chunk'],
        ['Max–Min 0,6; 0,9; 1,5','Kế thừa mặc định trong mã tác giả tại commit cố định; không phải tìm bằng XQuAD của nhóm','Không tuyên bố tối ưu trên tiếng Việt'],
        ['LlamaIndex buffer=1; percentile=95','Mặc định thư viện 0.14.25, baseline lịch sử; percentile không phải cosine 0,95','Không tuyên bố nhóm đã tune'],
        ['MiniLM 128 token','max_seq_length trong sentence_bert_config.json tại snapshot được ghi hash','Giới hạn cấu hình encoder, không phải ngưỡng nhóm tối ưu'],
        ['Top-3 và top-5','Mốc chấm retrieval và chính sách lấy context; đã bổ sung độ nhạy k=1/3/5/10','Không thể tối đa Recall đơn thuần để chứng minh k=5 tối ưu chi phí'],
        ['512 và 1024 token context','Hai trần chẩn đoán hậu nghiệm trên tiền tố top-5 đã lưu; không skip/truncate/backfill','Hai mức thử nghiệm, không có nghĩa mức tối ưu'],
        ['~20% dev; seed chia dữ liệu','Quy tắc hash theo title trong từng lĩnh vực; kiểm tra dev/test tách nguồn','Lựa chọn giao thức, không chọn split theo điểm cao'],
        ['240 context; 1.190 QA; 48 bài; 7 lĩnh vực','Đếm lại dữ liệu và tái tạo conversion trong bộ nhớ; 7 lĩnh vực là nhãn rộng do dự án gán','Số đếm, không phải siêu tham số'],
        ['233 dev/957 test; 50/190 paragraph; 10/38 article','Đếm lại và đối chiếu manifest; mọi câu cùng nguồn đi cùng split','Số đếm có thể tái lập'],
        ['5 context mỗi bài ghép; 1.181 câu hỏi duy nhất; 9 nhóm trùng','Đối chiếu conversion và duplicate audit; hai góc nhìn không độc lập','Số đếm có thể tái lập'],
        ['39–763; median 155; 496–2.169; median 804,5','Đếm lại đơn vị khoảng trắng của toàn bộ tài liệu trong manifest','Thống kê, không phải giới hạn tối ưu'],
        ['95% CI; 5.000 bootstrap; seed 2026','Quy ước mức khoảng và cấu hình tái lập; resample theo bài nguồn; đã bổ sung 1.000/5.000/10.000 lượt','Không chứng minh 5.000 hay seed 2026 là tối ưu'],
        ['Recall/MRR; số chunk; bảo toàn evidence; context; paired hits/misses','Tính lại từ ranking gốc và nhãn, đối chiếu JSON/CSV/statistical summary','Kết quả quan sát có phạm vi benchmark'],
        ['13 pilot; 16 câu pilot; 3 dependency/4 tài liệu; 2/3 so với 0/3','Đếm nhãn và chunk pilot; tài liệu dependency đều dưới 240 nên chưa kiểm chứng riêng penalty','Không phải bằng chứng tổng quát về risk'],
        ['86,73% và 97,91% chunk Ours vượt 128','Tokenize lại mọi chunk với cùng snapshot; đúng độ dài không đồng nghĩa evidence chắc chắn bị truncation','Thống kê độ dài, không phải tỷ lệ mất evidence'],
        ['Temperature=0; output tối đa 512 token; timeout=180 giây','Cấu hình adapter Ollama để vận hành; benchmark chính không gọi generation','Chưa có thực nghiệm chứng minh tối ưu QA'],
        ['Phiên bản package; năm xuất bản; commit/hash; ngày chạy','Metadata tái lập hoặc thông tin thư mục tài liệu tham khảo; phiên bản cũ là metadata lần chạy cũ','Không phải kết quả tối ưu tham số'],
        ['Thời gian chạy đã lưu','Một lần chạy warm model, phụ thuộc phần cứng/cache/thứ tự; không thể tái tạo cùng số giây tuyệt đối','Không phải benchmark tốc độ thống kê'],
    ]))
    blocks.append(r'''

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
''')
    (ROOT/'THAM_SO_VA_MINH_CHUNG.md').write_text('\n\n'.join(blocks)+'\n',encoding='utf-8')
    print('Wrote THAM_SO_VA_MINH_CHUNG.md')


if __name__=='__main__':
    main()
