from __future__ import annotations

from knowledge_base import retrieve


def build_messages(prompt: str, history: list[dict], base_url: str, embed_model: str, top_k: int = 4, portfolio_context: dict | None = None):
    live_context = _format_portfolio_context(portfolio_context)

    context_chars = len(live_context)
    effective_top_k = min(top_k, 1 if context_chars > 6500 else 2 if context_chars > 4000 else 4)
    matches = retrieve(prompt, base_url, embed_model, top_k=effective_top_k)
    if matches:
        context = "\n\n".join(
            f"[Nguồn {i}: {item['name']} | {item['location']}]\n{item['text']}"
            for i, item in enumerate(matches, 1)
        )
        source_instruction = (
            "Chỉ dùng NGUỒN để khẳng định nội dung thuộc tài liệu. Có thể giải thích thêm bằng kiến thức nền, "
            "nhưng phải phân biệt rõ. Nếu nguồn không đủ, hãy nói rõ."
        )
    else:
        context = "Không tìm thấy đoạn tài liệu phù hợp trong kho kiến thức."
        source_instruction = (
            "Nói rõ kho tài liệu hiện chưa có nguồn phù hợp. Có thể giải thích bằng kiến thức tài chính phổ thông, "
            "nhưng không được giả vờ trích dẫn tài liệu đã tải lên."
        )
    system_prompt = f"""Bạn là Portfolio AI, trợ lý phân tích đầu tư. Trả lời tiếng Việt, ngắn gọn và chính xác.
Khi hỏi về danh mục hiện tại, dùng đúng LIVE PORTFOLIO CONTEXT; không tự tính lại hay bịa số.
Các trường phần trăm đã được ghi kèm dấu %; không nhân/chia 100 lần nữa. Nếu thiếu trường, nói rõ chưa có dữ liệu.
Phân biệt số liệu Terminal với kiến thức tài liệu. Với khái niệm/công thức, giải thích biến và đơn vị.
{source_instruction}

LIVE PORTFOLIO CONTEXT:
{live_context}

TÀI LIỆU RAG:
{context}"""
    messages = [{"role": "system", "content": system_prompt}]
    history_limit = 2 if context_chars > 4000 else 4
    messages.extend(history[-history_limit:])
    messages.append({"role": "user", "content": prompt})
    return messages, matches


def _format_portfolio_context(data: dict | None) -> str:
    if not data:
        return ("DỮ LIỆU DANH MỤC HIỆN KHÔNG KHẢ DỤNG: lần chạy Terminal hiện tại chưa hoàn tất. "
                "Không sử dụng hoặc suy đoán từ số liệu của lần chạy trước; hãy yêu cầu người dùng "
                "khắc phục dữ liệu Terminal rồi thử lại.")

    pct_fields = {
        "total_return", "annual_return", "annual_volatility", "max_drawdown",
        "alpha_annual", "var_95_daily", "cvar_95_daily",
        "last_portfolio_daily_return", "last_benchmark_daily_return",
        "risk_free_rate_annual",
    }
    ratio_fields = {
        "sharpe", "sortino", "beta", "r_squared", "information_ratio", "treynor",
    }
    metadata_fields = ("updated_at", "data_source", "period", "benchmark", "market_regime")
    lines = ["LIVE PORTFOLIO METRICS (authoritative Terminal snapshot):"]
    for key in metadata_fields:
        value = data.get(key)
        if value is not None:
            lines.append(f"- {key}: {value}")

    for key in pct_fields:
        value = data.get(key)
        if value is not None:
            try:
                lines.append(f"- {key}: {float(value) * 100:.2f}%")
            except (TypeError, ValueError):
                continue

    for key in ratio_fields:
        value = data.get(key)
        if value is not None:
            try:
                lines.append(f"- {key}: {float(value):.4f}")
            except (TypeError, ValueError):
                continue

    tickers = data.get("tickers") or []
    if tickers:
        lines.append("- tickers: " + ", ".join(str(t) for t in tickers))

    holdings = data.get("holdings") or []
    if holdings:
        lines.append("- holdings (weight/price/returns):")
        for item in holdings[:15]:
            try:
                weight = f"{float(item.get('weight')) * 100:.2f}%" if item.get("weight") is not None else "n/a"
            except (TypeError, ValueError):
                weight = "n/a"
            try:
                price = f"{float(item.get('last_price')):,.2f}" if item.get("last_price") is not None else "n/a"
            except (TypeError, ValueError):
                price = "n/a"
            try:
                period_return = f"{float(item.get('period_return')) * 100:.2f}%" if item.get("period_return") is not None else "n/a"
            except (TypeError, ValueError):
                period_return = "n/a"
            try:
                daily_return = f"{float(item.get('daily_return')) * 100:.2f}%" if item.get("daily_return") is not None else "n/a"
            except (TypeError, ValueError):
                daily_return = "n/a"
            lines.append(
                f"  • {item.get('ticker', '?')}: weight={weight}; price={price}; "
                f"period_return={period_return}; daily_return={daily_return}"
            )

    lines.append(
        "Unit rule: all fields ending in return/volatility/drawdown/alpha/var/cvar/rate above "
        "are already percentages. Ratios (Sharpe, Sortino, Beta, R², Information Ratio, Treynor) "
        "are unitless. Never convert percentages again."
    )
    return "\n".join(lines)
