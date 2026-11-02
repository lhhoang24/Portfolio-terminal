from __future__ import annotations

import json
import shutil
import subprocess
from typing import Iterator

import requests

from knowledge_base import OllamaError

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def gpu_diagnostics(base_url: str = DEFAULT_OLLAMA_URL) -> dict:
    result = {
        "hardware": None,
        "ollama_loaded": False,
        "vram_bytes": 0,
        "model": None,
        "status": "Chưa xác định",
        "detail": "Ollama sẽ tự động chọn GPU tương thích nếu có driver phù hợp.",
    }


    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4, check=True,
            )
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                result["hardware"] = "NVIDIA · " + " | ".join(lines)
        except (subprocess.SubprocessError, OSError):
            pass

    try:
        response = requests.get(base_url.rstrip("/") + "/api/ps", timeout=3)
        response.raise_for_status()
        models = response.json().get("models", [])
        result["ollama_loaded"] = bool(models)
        if models:
            active = max(models, key=lambda item: int(item.get("size_vram", 0) or 0))
            result["model"] = active.get("name") or active.get("model")
            result["vram_bytes"] = int(active.get("size_vram", 0) or 0)
            if result["vram_bytes"] > 0:
                result["status"] = "GPU đang được sử dụng"
                result["detail"] = f"Ollama đang đặt khoảng {result['vram_bytes'] / (1024**3):.1f} GB model trong VRAM."
            else:
                result["status"] = "Model đang chạy trên CPU"
                result["detail"] = "Ollama chưa báo dung lượng VRAM được sử dụng cho model đang nạp."
        elif result["hardware"]:
            result["status"] = "Đã phát hiện GPU · chưa nạp model"
            result["detail"] = "Gửi câu hỏi đầu tiên để Ollama nạp model; sau đó kiểm tra lại trạng thái GPU."
        else:
            result["status"] = "Chưa phát hiện GPU hoạt động"
            result["detail"] = "Nếu máy có GPU rời, hãy cập nhật driver và cài Ollama bản Windows mới nhất."
    except requests.RequestException:
        result["status"] = "Ollama chưa kết nối"
        result["detail"] = "Mở Ollama trước khi kiểm tra tăng tốc GPU."
    return result


def stream_chat(
    *, base_url: str, model: str, messages: list[dict], num_ctx: int = 4096,
    num_predict: int = 512, temperature: float = 0.2,
) -> Iterator[str]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": "30m",

        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    try:
        with requests.post(
            base_url.rstrip("/") + "/api/chat", json=payload,
            stream=True, timeout=(10, 600),
        ) as response:
            if response.status_code == 404:
                raise OllamaError(f"Không tìm thấy model/API. Hãy kiểm tra đã tải model '{model}' bằng ollama pull.")
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("error"):
                    raise OllamaError(item["error"])
                token = item.get("message", {}).get("content", "")
                if token:
                    yield token
    except requests.RequestException as exc:
        raise OllamaError(f"Không gọi được Ollama: {exc}") from exc
