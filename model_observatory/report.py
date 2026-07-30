from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_report(output_dir: str | Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "summary.json"
    html_path = destination / "report.html"
    index_path = destination / "index.html"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rendered = render_report(summary)
    html_path.write_text(rendered, encoding="utf-8")
    index_path.write_text(rendered, encoding="utf-8")
    return html_path, json_path


def render_report(summary: dict[str, Any]) -> str:
    current = summary["current"]
    reference = summary.get("reference")
    drift = summary.get("drift")
    slices = summary.get("slices", [])
    metric_rows = [
        ("Accuracy", current["accuracy"], _delta(current, reference, "accuracy")),
        ("Macro F1", current["macro_f1"], _delta(current, reference, "macro_f1")),
        ("Log loss", current["log_loss"], _delta(current, reference, "log_loss")),
        ("ECE", current["ece"], _delta(current, reference, "ece")),
    ]
    cards = "".join(
        f"<div><span>{html.escape(name)}</span><strong>{value:.3f}</strong>"
        f"<small>{_format_delta(delta)}</small></div>"
        for name, value, delta in metric_rows
    )
    slice_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['slice']))}</td>"
        f"<td>{html.escape(str(row['value']))}</td>"
        f"<td>{int(row['count'])}</td>"
        f"<td>{float(row['accuracy']):.3f}</td>"
        f"<td class='{_gap_class(float(row['accuracy_gap']))}'>{float(row['accuracy_gap']):+.3f}</td>"
        f"<td>{float(row['ece']):.3f}</td>"
        "</tr>"
        for row in slices
    )
    class_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(name))}</td>"
        f"<td>{int(values['support'])}</td>"
        f"<td>{float(values['precision']):.3f}</td>"
        f"<td>{float(values['recall']):.3f}</td>"
        f"<td>{float(values['f1']):.3f}</td>"
        "</tr>"
        for name, values in current["per_class"].items()
    )
    risk_rows = "".join(
        f"<div class='risk-row'><span>{float(row['coverage']):.0%}</span>"
        f"<i style='width:{min(100.0, float(row['risk']) * 400):.1f}%'></i>"
        f"<strong>{float(row['risk']):.1%} risk</strong></div>"
        for row in current["risk_coverage"]
    )
    drift_section = _render_drift(drift)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Observatory Report</title>
  <link rel="icon" href="data:,">
  <style>
    :root{{--ink:#17202a;--muted:#65717d;--line:#d7dde2;--paper:#f4f6f7;--blue:#165dcc;--teal:#0d766e;--red:#b42318}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.5}}
    header,main,footer{{width:min(1080px,calc(100vw - 40px));margin:auto}} header{{padding:64px 0 30px;border-bottom:1px solid var(--line)}}
    header p{{margin:0;color:var(--teal);font:700 11px ui-monospace,monospace}} h1{{margin:9px 0 0;font-size:38px}} header span{{display:block;margin-top:10px;color:var(--muted)}}
    section{{padding:36px 0;border-bottom:1px solid var(--line)}} h2{{margin:0 0 20px;font-size:20px}} .metrics{{display:grid;grid-template-columns:repeat(4,1fr);background:white;border-top:1px solid var(--line);border-left:1px solid var(--line)}}
    .metrics div{{padding:20px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}} .metrics span,.metrics small{{display:block;color:var(--muted);font-size:11px}} .metrics strong{{display:block;margin:8px 0 5px;font:700 25px ui-monospace,monospace}}
    table{{width:100%;border-collapse:collapse;background:white;font-size:12px}} th,td{{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--muted);font-size:10px;text-transform:uppercase}} .negative{{color:var(--red);font-weight:700}} .positive{{color:var(--teal);font-weight:700}}
    .drift{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}} .drift div{{padding:18px;background:white}} .drift span{{display:block;color:var(--muted);font-size:11px}} .drift strong{{display:block;margin-top:7px;font:700 20px ui-monospace,monospace}}
    .risk-row{{display:grid;grid-template-columns:54px 1fr 80px;gap:12px;align-items:center;margin:9px 0;font-size:11px}} .risk-row i{{display:block;height:9px;background:var(--blue)}} .risk-row strong{{text-align:right}}
    footer{{padding:24px 0 48px;color:var(--muted);font-size:11px}} @media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.drift{{grid-template-columns:1fr}}h1{{font-size:30px}}table{{display:block;overflow-x:auto}}}}
  </style>
</head>
<body>
  <header><p>MODEL OBSERVATORY / CLASSIFICATION DIAGNOSTICS</p><h1>Evaluation report</h1><span>{int(current['count'])} current records analyzed with explicit calibration and slice checks.</span></header>
  <main>
    <section><h2>Current performance</h2><div class="metrics">{cards}</div></section>
    {drift_section}
    <section><h2>Worst-first slice analysis</h2><table><thead><tr><th>Slice</th><th>Value</th><th>Count</th><th>Accuracy</th><th>Gap</th><th>ECE</th></tr></thead><tbody>{slice_rows}</tbody></table></section>
    <section><h2>Class-level performance</h2><table><thead><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{class_rows}</tbody></table></section>
    <section><h2>Selective prediction</h2><div>{risk_rows}</div></section>
  </main>
  <footer>Generated deterministically by Model Observatory. Review summary.json for machine-readable evidence.</footer>
</body>
</html>
"""


def _render_drift(drift: dict[str, Any] | None) -> str:
    if not drift:
        return ""
    return (
        '<section><h2>Distribution shift</h2><div class="drift">'
        f"<div><span>Prediction JS divergence</span><strong>{drift['prediction_js_divergence']:.4f}</strong></div>"
        f"<div><span>Label JS divergence</span><strong>{drift['label_js_divergence']:.4f}</strong></div>"
        f"<div><span>Confidence PSI</span><strong>{drift['confidence_psi']:.4f}</strong></div>"
        "</div></section>"
    )


def _delta(current: dict[str, Any], reference: dict[str, Any] | None, key: str) -> float | None:
    return None if reference is None else float(current[key]) - float(reference[key])


def _format_delta(delta: float | None) -> str:
    return "single dataset" if delta is None else f"{delta:+.3f} vs reference"


def _gap_class(value: float) -> str:
    return "negative" if value < -0.02 else "positive" if value > 0.02 else ""
