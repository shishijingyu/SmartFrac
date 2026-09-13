"""HTML 验证报告（FR-4.8）。"""
from __future__ import annotations

import json
from pathlib import Path


def make_report(metrics: dict, figures: list[str], out: str | Path = "report.html") -> Path:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())
    imgs = "".join(f'<img src="{Path(f).name}" style="max-width:600px;margin:10px;"><br>'
                   for f in figures)
    html = f"""<html><head><meta charset="utf-8"><title>SmartFrac Report</title></head>
<body><h1>SmartFrac MVP Verification Report</h1>
<h2>Metrics</h2><table border="1">{rows}</table>
<h2>Figures</h2>{imgs}</body></html>"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out
