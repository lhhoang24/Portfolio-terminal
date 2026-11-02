from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--extra", default="")
    args = ap.parse_args()

    preset_syms: list[str] = []

    for grp in ("AAPL,MSFT,NVDA", "AAPL,JPM,XOM,JNJ,WMT", "JPM,BAC,WFC,GS",
                "NVDA,AMD,AVGO,QCOM,TXN", "JNJ,WMT,XOM,JPM"):
        preset_syms += data.clean_tickers(grp)
    syms = list(dict.fromkeys(preset_syms + data.PRESET_UNIVERSE + ["SPY", "QQQ", "IWM", "^VIX"] + data.clean_tickers(args.extra)))
    print(f"Đang tải {len(syms)} mã từ {args.start} ...")

    frames = []
    for i in range(0, len(syms), 20):
        part = syms[i : i + 20]
        close = data._flatten_close(data._download(part, args.start), part)
        print(f"  lô {i // 20 + 1}: {close.shape[1]}/{len(part)} mã")
        frames.append(close)
    prices = pd.concat([f for f in frames if not f.empty], axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    data.SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(data.SNAPSHOT_PATH)
    print(f"Đã lưu {prices.shape[1]} mã × {len(prices)} phiên -> {data.SNAPSHOT_PATH}")
    miss = [s for s in syms if s not in prices.columns]
    if miss:
        print("Không tải được:", ", ".join(miss))


if __name__ == "__main__":
    main()
