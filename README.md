# Portfolio Analytics Terminal

Ứng dụng Streamlit phân tích danh mục đầu tư: chỉ số rủi ro/lợi suất, CAPM và yếu tố, tối ưu hóa, backtest, quét quy mô lớn và trợ lý AI chạy cục bộ.

## Yêu cầu

- Python 3.11 trở lên
- Git
- Kết nối Internet (tải giá từ Yahoo Finance)
- Tùy chọn: [Ollama](https://ollama.com/download) cho trang Portfolio AI Assistant

## Cài đặt

### 1. Tải mã nguồn

```bash
git clone https://github.com/lhhoang24/Portfolio-terminal.git
cd Portfolio-terminal
```

### 2. Tạo môi trường ảo

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Nếu PowerShell chặn script, chạy `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi kích hoạt lại.

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng

```bash
python -m streamlit run app.py
```

Mở trình duyệt tại http://localhost:8501.

## Cài đặt trợ lý AI (tùy chọn)

1. Cài Ollama từ https://ollama.com/download và đảm bảo dịch vụ đang chạy.
2. Tải mô hình:

   ```bash
   ollama pull qwen3:1.7b
   ollama pull nomic-embed-text
   ```

   Có thể tải thêm `qwen3:4b` hoặc `qwen3:8b` để chọn trong giao diện.
3. Mở trang **Portfolio AI Assistant** → **Kho kiến thức** và tải tài liệu PDF, DOCX, PPTX, TXT, MD hoặc CSV.

Kho kiến thức được lưu tại `data/knowledge_base.sqlite3` và đã được đưa vào `.gitignore`.

## Dữ liệu giá dự phòng

Khi Yahoo Finance không phản hồi, ứng dụng dùng file `data/prices_snapshot.csv`. Để cập nhật file này (chạy trên máy có Internet):

```bash
python scripts/build_snapshot.py
python scripts/build_snapshot.py --start 2021-01-01 --extra VNM,USDC-USD
```

## Cấu trúc thư mục

```
app.py                  Điểm vào, điều hướng giữa các trang
terminal.py             Giao diện phân tích chính
assistant.py            Trang trợ lý AI và kho kiến thức
ai_chat_widget.py       Khung chat nổi trên trang chính
ai_chat_logic.py        Dựng prompt và ngữ cảnh danh mục
ai_runtime.py           Kết nối Ollama
knowledge_base.py       Lưu trữ và truy xuất tài liệu (SQLite)
analytics.py            Hàm tính chỉ số, tối ưu hóa, backtest
charts.py               Biểu đồ Plotly
theme.py                Giao diện tối
data.py                 Tải dữ liệu và cache
scripts/                Tạo snapshot giá
data/                   Snapshot giá và dữ liệu cục bộ
```

