# Streamlit playground

Từ thư mục project:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-gui.txt
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Mở http://localhost:8501. Dán văn bản, chọn Ours hoặc Max–Min và bấm **Chia đoạn
và tạo chỉ mục**. Sang **Hỏi đáp RAG**, nhập câu hỏi và top-k, bấm **Trả lời**.
Ứng dụng tự truy xuất rồi sinh đáp án trong một lần bấm; đáp án ở trên, context
top-k và prompt bên dưới. Có thể tải JSON của lần chạy.

Project đặt `server.fileWatcherType = "none"` trong `.streamlit/config.toml`.
Watcher của Streamlit có thể quét lazy module xử lý ảnh của Transformers và gây
`ModuleNotFoundError: No module named 'torchvision'` dù retrieval đã thành công.
Ứng dụng này chỉ dùng embedding văn bản, không cần cài torchvision để xử lý lỗi
watcher. Sau khi đổi config, dừng Streamlit bằng Ctrl+C rồi chạy lại. Vì watcher
đã tắt, khi sửa code hãy khởi động lại server để nạp thay đổi.
Nếu chạy ngoài thư mục project, truyền `--server.fileWatcherType none` trực tiếp.
Tham khảo [cấu hình Streamlit](https://docs.streamlit.io/develop/api-reference/configuration/config.toml).

Ours cho chỉnh MIN/TARGET/MAX (đơn vị phân cách bằng khoảng trắng), semantic drop,
ngưỡng overlap thấp/cao. Nút khôi phục đưa các giá trị về 120/180/240, .30, .30/.55.
Phải có `0 < MIN < TARGET < MAX` và ngưỡng overlap thấp < cao. Trọng số scoring
0.4/0.4/0.6 và các luật dependency vẫn cố định như thuật toán gốc.
Semantic drop chỉ điều khiển nhánh cắt tùy chọn; không phải ngưỡng chung cho mọi
điểm cắt. MAX không phải giới hạn cứng của chunk sau khi ghép overlap.

Max–Min giữ nguyên các tham số .6/.9/1.5. Cả hai dùng RAGPipeline hiện có và
cùng model MiniLM. Nút **Trả lời** tự gửi prompt sau retrieval đến
`http://localhost:11434/api/generate`, model `qwen2.5:3b`.
Bạn tự chạy Ollama; ứng dụng không khởi động dịch vụ hoặc pull model.
Chỉ bấm nút trả lời mới gọi LLM. Temperature 0, giới hạn sinh 512 token,
timeout 180 giây. Câu trả lời và cấu hình được đưa vào JSON tải xuống.

`gui/chunkers.py` nạp các file chunking Ours vào module runtime riêng, gán các
constant trên bản riêng và nối các hàm scoring/overlap tương ứng. Không sửa file
nguồn, không monkeypatch module chung, không sao chép thuật toán. Vì vậy cấu hình
của các phiên GUI độc lập và benchmark CLI giữ nguyên mặc định. Model embedding
được cache bằng `st.cache_resource`; chỉ mục nằm trong session của từng người.

Thay văn bản/chiến lược/tham số làm kết quả cũ được đánh dấu và vô hiệu hóa truy
xuất đến khi tạo lại chỉ mục. Mọi kết quả xuất JSON có cấu hình đã thực sự chạy.
Kết quả thử tham số không tự ghi vào các file benchmark chính thức.

Streamlit sử dụng session state và resource cache theo
[tài liệu chính thức](https://docs.streamlit.io/develop/concepts/architecture/caching).
