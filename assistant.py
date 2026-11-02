from __future__ import annotations

import requests
import streamlit as st

from knowledge_base import (
    DB_PATH, DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, OllamaError,
    add_document, delete_document, list_documents, ollama_status,
)
from ai_chat_logic import build_messages
from ai_runtime import gpu_diagnostics, stream_chat

st.set_page_config(page_title="Portfolio AI Assistant", page_icon=":material/smart_toy:", layout="wide")

st.markdown("""
<style>
[data-testid="stChatMessage"] {border-radius:16px;padding:.75rem .95rem;margin:.45rem 0;
border:1px solid rgba(148,163,184,.16);}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {background:rgba(29,78,216,.16);}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {background:rgba(51,65,85,.24);}
.ai-hero{padding:1.2rem 1.4rem;border:1px solid rgba(120,140,170,.25);border-radius:14px;
background:linear-gradient(120deg,rgba(35,73,112,.30),rgba(20,25,38,.20));margin-bottom:1rem}
.ai-muted{color:#9aa8ba;font-size:.88rem}
.ai-source{border-left:3px solid #4d91d1;padding:.35rem .7rem;margin:.35rem 0;background:rgba(100,130,170,.08);border-radius:0 7px 7px 0}
</style>
""", unsafe_allow_html=True)

st.title("Portfolio AI Assistant")
st.caption("Trợ lý kiến thức tài chính định lượng · Local LLM · Retrieval-Augmented Generation")
st.markdown("""
<div class="ai-hero">
<strong>Học công thức, hiểu khái niệm, tra cứu từ chính tài liệu của bạn.</strong><br>
<span class="ai-muted">Tài liệu và cơ sở tri thức được lưu cục bộ. Câu trả lời dựa trên các đoạn tài liệu truy xuất được; nếu thiếu nguồn, AI sẽ nói rõ.</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Cấu hình AI local")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    model_options = ["qwen3:1.7b", "qwen3:4b", "qwen3:8b"]
    saved_model = st.session_state.get("assistant_chat_model", DEFAULT_CHAT_MODEL)
    if saved_model not in model_options:
        saved_model = DEFAULT_CHAT_MODEL
    chat_model = st.selectbox("Mô hình chat", model_options, index=model_options.index(saved_model))
    st.session_state["assistant_chat_model"] = chat_model
    embed_model = st.text_input("Mô hình embedding", value=DEFAULT_EMBED_MODEL)
    if st.button("Kiểm tra kết nối", width="stretch"):
        ok, message = ollama_status(ollama_url)
        (st.success if ok else st.error)(message)
        diag = gpu_diagnostics(ollama_url)
        st.write(f"**GPU:** {diag['status']}")
        st.caption(diag["detail"])
        if diag.get("hardware"):
            st.caption(diag["hardware"])
    st.caption("Có thể chọn Qwen3 1.7B (nhanh), 4B (cân bằng) hoặc 8B (chất lượng cao, cần VRAM lớn).")

chat_tab, library_tab, guide_tab = st.tabs(["💬 Hỏi đáp", "📚 Kho kiến thức", "⚙️ Hướng dẫn cài đặt"])

with chat_tab:
    docs = list_documents()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tài liệu đã nạp", len(docs))
    c2.metric("Đoạn kiến thức", sum(d["chunks"] for d in docs))
    c3.metric("LLM", chat_model)

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []
    for msg in st.session_state.ai_messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Nguồn tham khảo"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src['name']}** · {src['location']} · độ liên quan {src['score']:.2f}")

    prompt = st.chat_input("Ví dụ: Giải thích Sharpe Ratio và cho ví dụ tính toán...")
    if prompt:
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            try:
                with st.spinner("Đang tìm kiến thức liên quan và tạo câu trả lời..."):
                    previous = [
                        {"role": old["role"], "content": old["content"]}
                        for old in st.session_state.ai_messages[-7:-1]
                    ]
                    messages, matches = build_messages(
                        prompt, previous, ollama_url, embed_model, top_k=4,
                        portfolio_context=st.session_state.get("portfolio_ai_context"),
                    )
                    result = st.write_stream(stream_chat(
                        base_url=ollama_url, model=chat_model, messages=messages,
                        num_ctx=4096, num_predict=512, temperature=0.3,
                    ))
                    if not result:
                        raise OllamaError("Mô hình trả về nội dung trống.")
                    if matches:
                        with st.expander("Nguồn tham khảo được truy xuất"):
                            for item in matches:
                                st.markdown(f"- **{item['name']}** · {item['location']} · độ liên quan {item['score']:.2f}")
                    st.session_state.ai_messages.append({
                        "role": "assistant", "content": result,
                        "sources": [{k: item[k] for k in ("name", "location", "score")} for item in matches],
                    })
            except (OllamaError, requests.RequestException, ValueError) as exc:
                error = f"Chưa thể tạo câu trả lời: {exc}\n\nKiểm tra Ollama đang chạy, đúng URL và đã tải đủ mô hình."
                st.error(error)
                st.session_state.ai_messages.append({"role": "assistant", "content": error})
    if st.session_state.ai_messages and st.button("Xóa lịch sử hội thoại"):
        st.session_state.ai_messages = []
        st.rerun()

with library_tab:
    st.subheader("Nạp tài liệu vào bộ nhớ kiến thức")
    st.write("Tải tài liệu môn học, ghi chú, slide hoặc tài liệu dự án. Hệ thống trích xuất văn bản, chia thành đoạn và tạo embedding local.")
    uploads = st.file_uploader(
        "Chọn tài liệu", type=["pdf", "docx", "pptx", "txt", "md", "csv"],
        accept_multiple_files=True, help="Khuyến nghị mỗi file dưới 30 MB. File scan cần OCR trước khi nạp.",
    )
    if st.button("Phân tích và nạp vào Knowledge Base", type="primary", disabled=not uploads):
        progress = st.progress(0)
        status = st.empty()
        for index, uploaded in enumerate(uploads):
            try:
                status.info(f"Đang xử lý: {uploaded.name}")
                count, message = add_document(uploaded.name, uploaded.getvalue(), ollama_url, embed_model)
                st.success(f"{uploaded.name}: {message} ({count} đoạn)")
            except Exception as exc:
                st.error(f"{uploaded.name}: {exc}")
            progress.progress((index + 1) / len(uploads))
        status.success("Hoàn tất quá trình nạp tài liệu.")
        st.rerun()

    st.divider()
    st.subheader("Tài liệu trong Knowledge Base")
    docs = list_documents()
    if not docs:
        st.info("Chưa có tài liệu. Hãy tải PDF, DOCX, PPTX, TXT, MD hoặc CSV ở phía trên.")
    else:
        for doc in docs:
            left, right = st.columns([5, 1])
            left.markdown(f"**{doc['name']}**  \n<small>{doc['chunks']} đoạn · nạp lúc {doc['created_at']}</small>", unsafe_allow_html=True)
            if right.button("Xóa", key=f"delete_{doc['id']}"):
                delete_document(doc["id"])
                st.rerun()

with guide_tab:
    st.subheader("Cài đặt lần đầu")
    st.markdown("""
1. Cài **Ollama** từ [ollama.com/download](https://ollama.com/download), cập nhật driver GPU (NVIDIA/AMD tương thích).
2. Mở Terminal / PowerShell và tải mô hình:
   ```bash
   ollama pull qwen3:1.7b
   ollama pull nomic-embed-text
   ```
   Mô hình mặc định là `qwen3:1.7b` (ưu tiên tốc độ).
3. Khởi động Ollama (thường tự chạy nền), sau đó chạy ứng dụng:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Mở trang **Portfolio AI Assistant** trên thanh điều hướng.
5. Vào tab **Kho kiến thức**, tải tài liệu rồi bắt đầu hỏi đáp.

**Chọn model:** mặc định là `qwen3:1.7b`; có thể chọn thêm `qwen3:4b` hoặc `qwen3:8b` trong giao diện. Model cần được tải trước bằng `ollama pull <tên-model>`.

**Tăng tốc:** Ollama tự động dùng GPU tương thích khi driver đã cài đúng. Chat hiển thị từng token, tắt chế độ suy luận dài của Qwen3 và giữ model nạp 30 phút. Nếu GPU không tương thích hoặc VRAM không đủ, Ollama có thể chạy một phần/toàn bộ trên CPU; không thể ép GPU bằng thay đổi Python đơn thuần.

**Lưu ý:** PDF scan dạng ảnh chưa được OCR tự động. Ollama chạy local, nhưng việc tải mô hình ban đầu cần Internet. Không gửi tài liệu lên dịch vụ AI bên ngoài.
    """)
    st.caption(f"Cơ sở dữ liệu local: {DB_PATH}")
