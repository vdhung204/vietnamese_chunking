"""Run: python -m streamlit run streamlit_app.py"""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from gui.chunkers import OursSettings, configured_ours

st.set_page_config(page_title='Vietnamese Chunking Lab', page_icon='🧩', layout='wide')


@st.cache_resource
def shared_embedder():
    from rag.embedder import SharedEmbedder
    return SharedEmbedder()


def reset_settings():
    for key, value in asdict(OursSettings()).items():
        st.session_state[key] = value


st.title('Vietnamese Chunking Lab')
st.caption('Thử cách chia văn bản tiếng Việt và tìm những đoạn liên quan đến câu hỏi của bạn.')

with st.sidebar:
    st.header('Chiến lược chunking')
    strategy = st.radio('Chọn phương pháp', ['Ours', 'Max–Min'], key='strategy')
    if strategy == 'Ours':
        st.subheader('Tham số Ours')
        st.button('Khôi phục mặc định', on_click=reset_settings)
        minimum = st.number_input('MIN — độ dài tối thiểu', min_value=1, value=120, step=10, key='min_size')
        target = st.number_input('TARGET — độ dài mục tiêu', min_value=2, value=180, step=10, key='target_size')
        maximum = st.number_input('MAX — độ dài tối đa', min_value=3, value=240, step=10, key='max_size')
        drop = st.number_input('Ngưỡng semantic drop', min_value=0., max_value=2., value=.30, step=.05, key='semantic_drop',
                               help='Áp dụng cho nhánh cắt tùy chọn khi độ dài phần còn lại nằm giữa MIN và MAX.')
        low = st.number_input('Ngưỡng overlap thấp', min_value=-1., max_value=1., value=.30, step=.05, key='overlap_low')
        high = st.number_input('Ngưỡng overlap cao', min_value=-1., max_value=1., value=.55, step=.05, key='overlap_high')
        settings = OursSettings(minimum, target, maximum, drop, low, high)
        st.caption('Overlap: dưới ngưỡng thấp → 0 câu; dưới ngưỡng cao → 1 câu; còn lại → tối đa 2 câu.')
        st.caption('Độ dài tính theo khoảng trắng. Chunk cuối có thể vượt MAX do overlap hoặc không tìm được điểm cắt.')
    else:
        settings = None
        st.info('Max–Min dùng cấu hình cố định: threshold 0.6 · c 0.9 · init 1.5.')
    st.divider()
    st.caption('Tham số tùy chỉnh chỉ áp dụng cho lần chạy trên giao diện. Benchmark giữ cấu hình mặc định.')

text = st.text_area('Văn bản đầu vào', height=260, placeholder='Dán văn bản tiếng Việt bạn muốn chia đoạn…', key='document_text')
st.caption(f'{len(text.split()):,} đơn vị cách nhau bằng khoảng trắng')
current_config = asdict(settings) if settings else dict(fixed_threshold=.6, c=.9, init_constant=1.5)
signature = (text, strategy, json.dumps(current_config, sort_keys=True))

if st.button('Chia đoạn và tạo chỉ mục', type='primary', key='index_button'):
    try:
        if not text.strip():
            raise ValueError('Hãy nhập văn bản trước khi chạy.')
        if settings:
            settings.validate()
        with st.spinner('Đang chia đoạn và tạo embedding… Lần đầu có thể cần tải model.'):
            from rag.rag_pipeline import RAGPipeline
            from baselines.maxmin_semantic_chunker import maxmin_semantic_chunk
            chunker = configured_ours(settings) if settings else maxmin_semantic_chunk
            pipeline = RAGPipeline(chunker, shared_embedder()).index_documents([
                dict(id='input', text=text)])
        st.session_state['run'] = dict(pipeline=pipeline, signature=signature,
                                       strategy=strategy, config=current_config)
        st.session_state.pop('retrieval', None)
        st.success('Đã chia đoạn và tạo chỉ mục.')
    except Exception as exc:
        st.error(f'Không thể xử lý văn bản: {exc}')

run = st.session_state.get('run')
if run is None:
    st.info('Nhập văn bản, chọn chiến lược rồi bấm “Chia đoạn và tạo chỉ mục” để bắt đầu.')
    st.stop()

stale = signature != run['signature']
if stale:
    st.warning('Bạn đã đổi văn bản hoặc cấu hình. Kết quả bên dưới thuộc lần chạy trước; hãy tạo lại chỉ mục để truy xuất.')
pipeline = run['pipeline']
st.subheader(f"Kết quả lần chạy: {run['strategy']}")
cols = st.columns(4)
cols[0].metric('Số chunk', len(pipeline.chunks))
cols[1].metric('Độ dài TB', f"{sum(len(c['text'].split()) for c in pipeline.chunks) / len(pipeline.chunks):.1f}")
cols[2].metric('Chunking', f'{pipeline.chunking_time:.3f} s')
cols[3].metric('Embedding / index', f'{pipeline.embedding_time:.3f} s')
chunks_tab, retrieval_tab = st.tabs(['Các chunk', 'Hỏi đáp RAG'])

with chunks_tab:
    with st.expander('Cấu hình đã chạy'):
        st.json(run['config'])
    for i, chunk in enumerate(pipeline.chunks, 1):
        with st.expander(f"Chunk {i} · {len(chunk['text'].split())} đơn vị độ dài", expanded=True):
            st.text(chunk['text'])

with retrieval_tab:
    st.caption('Nhập câu hỏi rồi bấm Trả lời: hệ thống tự truy xuất context và gọi Qwen 2.5 3B qua Ollama local.')
    with st.form('query_form'):
        question = st.text_input('Câu hỏi', placeholder='Bạn muốn tìm thông tin gì?', key='question')
        top_k = st.number_input('Top-k', min_value=1, max_value=100, value=5, step=1, key='top_k')
        search = st.form_submit_button('Trả lời', disabled=stale)
    if search:
        if not question.strip():
            st.error('Hãy nhập câu hỏi.')
        else:
            st.session_state.pop('retrieval', None)
            try:
                from evaluation.rag_retrieval import build_rag_prompt
                from rag.llm_client import OllamaClient
                with st.spinner('Đang truy xuất…'):
                    retrieved = pipeline.retrieve(question, top_k)
                st.session_state['retrieval'] = dict(question=question, top_k=top_k,
                    retrieved=retrieved, prompt=build_rag_prompt(question, retrieved))
                result = st.session_state['retrieval']
                client = OllamaClient()
                try:
                    with st.spinner('Đang chờ Qwen sinh câu trả lời…'):
                        result['answer'] = client.generate(result['prompt'])
                    result['generation'] = dict(**asdict(client.config), base_url=client.base_url)
                except Exception as exc:
                    result['generation_error'] = str(exc)
            except Exception as exc:
                st.error(f'Không thể truy xuất: {exc}')
    retrieval = st.session_state.get('retrieval')
    if retrieval:
        st.write(f"Kết quả cho câu hỏi: {retrieval['question']}")
        st.subheader('Câu trả lời')
        if 'answer' in retrieval:
            st.write(retrieval['answer'])
        elif 'generation_error' in retrieval:
            st.error(f"Không thể sinh câu trả lời: {retrieval['generation_error']}")
            st.caption('Context đã truy xuất được giữ bên dưới. Kiểm tra Ollama rồi bấm Trả lời để thử lại.')
        st.subheader(f"Context truy xuất · {len(retrieval['retrieved'])} chunk / top-k {retrieval['top_k']}")
        for chunk in retrieval['retrieved']:
            with st.expander(f"Hạng {chunk['rank']} · {chunk['chunk_id']} · cosine {chunk['score']:.4f}", expanded=True):
                st.text(chunk['text'])
        with st.expander('Prompt đã ghép từ context'):
            st.code(retrieval['prompt'], language=None)

payload = dict(strategy=run['strategy'], parameters=run['config'], input_text=run['signature'][0],
               chunks=pipeline.chunks, chunking_time=pipeline.chunking_time,
               embedding_time=pipeline.embedding_time, retrieval=st.session_state.get('retrieval'))
st.download_button('Tải kết quả JSON', json.dumps(payload, ensure_ascii=False, indent=2),
                   file_name='chunking_playground.json', mime='application/json')
