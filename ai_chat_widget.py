from __future__ import annotations

import streamlit as st

from ai_chat_logic import build_messages
from ai_runtime import DEFAULT_OLLAMA_URL, stream_chat
from knowledge_base import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, OllamaError

MODEL_OPTIONS = ["qwen3:1.7b", "qwen3:4b", "qwen3:8b"]


def render_floating_chat(portfolio_context: dict | None = None) -> None:
    if portfolio_context is not None:
        st.session_state["portfolio_ai_context"] = portfolio_context
    st.markdown(
        """
        <style>
        .st-key-ai_floating_launcher {
            position:fixed!important;right:1.15rem!important;bottom:1.05rem!important;
            z-index:99999!important;width:auto!important;max-width:min(94vw,480px)!important;
        }
        .st-key-ai_floating_launcher [data-testid="stPopover"]>button {
            border-radius:999px!important;min-height:3.05rem!important;padding:.55rem 1.1rem!important;
            border:1px solid rgba(56,189,248,.7)!important;
            background:linear-gradient(135deg,#075985,#1d4ed8)!important;color:white!important;
            box-shadow:0 9px 28px rgba(0,0,0,.38)!important;font-weight:700!important;
        }
        .st-key-ai_floating_launcher [data-testid="stPopoverBody"] {
            width:min(480px,calc(100vw - 1.25rem))!important;
            max-height:76vh!important;overflow-y:auto!important;
            border:1px solid rgba(100,180,230,.25)!important;border-radius:18px!important;
            padding:.85rem!important;
        }
        .ai-panel-title {font-size:1.18rem;font-weight:750;color:#f8fafc;letter-spacing:-.02em;}
        .ai-panel-sub {font-size:.70rem;color:#94a3b8;margin-top:.08rem;margin-bottom:.35rem;}
        .st-key-ai_floating_launcher [data-testid="stChatMessage"] {
            border-radius:17px!important;padding:.65rem .8rem!important;margin:.38rem 0!important;
            border:1px solid rgba(148,163,184,.14)!important;
        }
        .st-key-ai_floating_launcher [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background:rgba(29,78,216,.20)!important;margin-left:1.5rem!important;
        }
        .st-key-ai_floating_launcher [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background:rgba(51,65,85,.30)!important;margin-right:.65rem!important;
        }
        .st-key-ai_floating_launcher [data-testid="stTextInput"] input {
            min-height:2.8rem!important;border-radius:14px!important;
        }
        .st-key-ai_floating_launcher [data-testid="stSelectbox"] [data-baseweb="select"] {
            min-height:2rem!important;font-size:.78rem!important;
        }
        .ai-model-caption {font-size:.68rem;color:#8293a8;padding-top:.45rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="ai_floating_launcher"):
        with st.popover("✦  Hỏi Portfolio AI", help="Mở trợ lý AI"):
            title_l, title_r = st.columns([4, 1])
            with title_l:
                st.markdown('<div class="ai-panel-title">Portfolio AI</div>', unsafe_allow_html=True)
                st.markdown('<div class="ai-panel-sub">Trợ lý phân tích tài chính · Knowledge Base</div>',
                            unsafe_allow_html=True)
            with title_r:
                if st.button("↺", key="floating_ai_clear_top", help="Xóa hội thoại"):
                    st.session_state["floating_ai_messages"] = []
                    st.rerun()

            base_url = st.session_state.get("floating_ollama_url", DEFAULT_OLLAMA_URL)
            embed_model = st.session_state.get("floating_embed_model", DEFAULT_EMBED_MODEL)
            if "floating_ai_messages" not in st.session_state:
                st.session_state.floating_ai_messages = []
            history = st.session_state.floating_ai_messages


            if history:
                with st.container(height=340, border=False):
                    for message in history[-24:]:
                        with st.chat_message(
                            message["role"],
                            avatar="👤" if message["role"] == "user" else "🤖",
                        ):
                            st.markdown(message["content"])
                            if message.get("sources"):
                                with st.expander("Nguồn tham khảo"):
                                    for source in message["sources"]:
                                        st.caption(
                                            f"{source['name']} · {source['location']} · "
                                            f"độ liên quan {source['score']:.2f}"
                                        )


            if not history:
                st.caption("Hỏi về CAPM, Sharpe, VaR, công thức hoặc tài liệu đã tải lên.")

            pending = st.session_state.pop("floating_ai_pending_prompt", "")
            with st.form("floating_ai_chat_form", clear_on_submit=True):
                prompt = st.text_input(
                    "Tin nhắn", value=pending,
                    placeholder="Nhắn tin cho Portfolio AI…",
                    label_visibility="collapsed",
                )
                send = st.form_submit_button("Gửi  ↗", type="primary", use_container_width=True)


            model_row_l, model_row_r = st.columns([1.25, 2.75])
            with model_row_l:
                st.markdown('<div class="ai-model-caption">MÔ HÌNH TRẢ LỜI</div>', unsafe_allow_html=True)
            with model_row_r:
                selected = st.session_state.get("floating_chat_model", DEFAULT_CHAT_MODEL)
                if selected not in MODEL_OPTIONS:
                    selected = MODEL_OPTIONS[0]
                model = st.selectbox(
                    "Phiên bản Qwen", MODEL_OPTIONS,
                    index=MODEL_OPTIONS.index(selected),
                    key="floating_chat_model_select",
                    label_visibility="collapsed",
                    help="1.7B nhanh · 4B cân bằng · 8B cần VRAM lớn hơn.",
                )
                st.session_state["floating_chat_model"] = model

            if send and prompt.strip():
                previous = [
                    {"role": item["role"], "content": item["content"]}
                    for item in st.session_state.floating_ai_messages[-8:]
                ]
                st.session_state.floating_ai_messages.append(
                    {"role": "user", "content": prompt.strip()}
                )
                try:
                    messages, matches = build_messages(
                        prompt.strip(), previous, base_url, embed_model, top_k=4,
                        portfolio_context=st.session_state.get("portfolio_ai_context"),
                    )
                    with st.chat_message("assistant", avatar="🤖"):
                        answer = st.write_stream(stream_chat(
                            base_url=base_url,
                            model=model,
                            messages=messages,
                            num_ctx=4096,
                            num_predict=512,
                            temperature=0.3,
                        ))
                    if not answer:
                        answer = "Mô hình chưa trả về nội dung. Hãy kiểm tra model Ollama."
                    st.session_state.floating_ai_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [
                            {key: item[key] for key in ("name", "location", "score")}
                            for item in matches
                        ],
                    })
                    st.rerun()
                except (OllamaError, ValueError) as exc:
                    error = f"Chưa thể trả lời: {exc}\n\nNếu model chưa được tải, chạy: ollama pull {model}"
                    st.session_state.floating_ai_messages.append(
                        {"role": "assistant", "content": error}
                    )
                    st.error(error)
