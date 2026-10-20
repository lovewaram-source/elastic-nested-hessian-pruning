"""Build editable README SVGs with stdlib only; numbers come from docs/RESULTS.md.

Layout: small-multiple line grammar adapted from the CCFA plotting recipe.
The method graph reuses the existing docs/METHOD.md and local preview topology.
No generated measurements, fitted curves, error bars or inferred missing data.
"""
from html import escape
from pathlib import Path
import argparse
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FONT = "Times New Roman, Microsoft YaHei, SimSun, serif"
# Same method order and visual encoding in every panel. Dashes supplement color.
METHODS = (
    ("历史 common dense-reference H", "Historical common H", "#667085", "7 5"),
    ("历史各率完整对齐 H", "Historical aligned H", "#0072B2", "10 3 2 3"),
    ("全位置平均激活→H", "All-position mean activation", "#009E73", ""),
    ("50%位置平均激活→H", "50% mean activation", "#E69F00", "9 4"),
    ("10% Token 私有（checkpoint）", "10% private tokens", "#CC79A7", "3 4"),
    ("50% Token 私有（checkpoint）", "50% private tokens", "#D55E00", "12 4"),
    ("全位置平均各分支 H", "Mean branch Hessian (2:4 only)", "#0B7285", "2 3 8 3"),
)


def read_results(path):
    """Read the canonical Markdown tables, leaving missing values as None."""
    tables, current = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = {"非结构化 PPL": "unstructured", "2:4 PPL": "2of4"}.get(line[3:])
            if current:
                tables[current] = {}
        elif current and line.startswith("|"):
            fields = [part.strip() for part in line.strip("|").split("|")]
            if fields[0] in {m[0] for m in METHODS}:
                if len(fields) != 5:
                    raise ValueError(f"Expected four sparsity rates: {fields[0]}")
                values = [None if v == "—" else float(v) for v in fields[1:]]
                if any(v is not None and (not math.isfinite(v) or v <= 0) for v in values):
                    raise ValueError("PPL must be positive and finite")
                tables[current][fields[0]] = values
    for key in ("unstructured", "2of4"):
        if METHODS[0][0] not in tables.get(key, {}):
            raise ValueError(f"Missing historical baseline: {key}")
    return tables


class SVG:
    def __init__(self, width, height, title, description):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                      f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
                      f'<style>text{{font-family:{FONT};fill:#17212b}}</style>',
                      '<rect width="100%" height="100%" fill="white"/>']

    def text(self, x, y, value, size=20, anchor="start", bold=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
                          f'font-weight="{700 if bold else 400}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color="#d8dee4", width=1, dash=""):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                          f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def render(self):
        result = "\n".join(self.parts + ["</svg>\n"])
        ET.fromstring(result)
        return result


def result_figure(tables):
    s = SVG(1200, 940, "Historical Hessian PPL comparison",
            "Absolute PPL and PPL minus historical common H at 20, 30, 40 and 50 percent sparsity. "
            "Historical and checkpoint reconstruction protocols differ. Missing results are not plotted.")
    s.text(36, 38, "Hessian statistics: historical PPL comparison", 28, bold=True)
    s.text(36, 70, "Llama-2-7B · C4 64 × 4096 · seed 0 · full WikiText2 · lower PPL is better", 20)
    for i, (_, label, color, dash) in enumerate(METHODS):
        col, row = i % 3, i // 3
        x, y = 36 + col * 395, 110 + row * 32
        s.line(x, y - 6, x + 32, y - 6, color, 3, dash)
        s.text(x + 43, y, label, 18)
    configs = (
        ("unstructured", 100, "Unstructured · α = 1", (5.4, 6.8), [5.4, 5.8, 6.2, 6.6],
         (-0.04, 0.01), [-0.04, -0.02, 0, 0.01]),
        ("2of4", 695, "2:4 · α = 0.25", (5.6, 9.0), [6, 7, 8, 9],
         (-0.125, 0.025), [-0.12, -0.08, -0.04, 0]),
    )
    for key, left, label, absolute, abs_ticks, delta, delta_ticks in configs:
        for is_delta, top, limits, ticks in ((False, 240, absolute, abs_ticks),
                                             (True, 570, delta, delta_ticks)):
            width, height = 440, 208
            lo, hi = limits
            sx = lambda i: left + 15 + i * (width - 30) / 3
            sy = lambda v: top + height - (v - lo) / (hi - lo) * height
            s.text(left, top - 24, label + (" · ΔPPL" if is_delta else " · PPL"), 22, bold=True)
            for tick in ticks:
                y = sy(tick)
                s.line(left, y, left + width, y, "#a6adb7" if tick == 0 else "#e4e7ec",
                       1.5 if tick == 0 else 1)
                s.text(left - 12, y + 6, f"{tick:.2f}" if is_delta else f"{tick:g}", 18, "end")
            s.line(left, top, left, top + height, "#344054")
            s.line(left, top + height, left + width, top + height, "#344054")
            for i, rate in enumerate((20, 30, 40, 50)):
                s.text(sx(i), top + height + 27, f"{rate}%", 19, "middle")
            s.text(left + width / 2, top + height + 55, "Weight sparsity", 19, "middle")
            baseline = tables[key][METHODS[0][0]]
            for method_index, (name, _, color, dash) in enumerate(METHODS):
                values = tables[key].get(name)
                if values is None:
                    continue
                previous = None
                for i, value in enumerate(values):
                    if value is None:
                        previous = None
                        continue
                    v = value - baseline[i] if is_delta else value
                    if not lo <= v <= hi:
                        raise ValueError(f"Point outside plotting limits: {key}, {name}, {v}")
                    x, y = sx(i), sy(v)
                    if previous:
                        s.line(*previous, x, y, color, 2.5, dash)
                    # Different marker silhouettes survive grayscale reproduction.
                    if method_index % 3 == 0:
                        s.parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="white" stroke="{color}" stroke-width="2"/>')
                    elif method_index % 3 == 1:
                        s.parts.append(f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="white" stroke="{color}" stroke-width="2"/>')
                    else:
                        s.parts.append(f'<path d="M{x},{y-5} L{x+5},{y} L{x},{y+5} L{x-5},{y} Z" fill="white" stroke="{color}" stroke-width="2"/>')
                    previous = x, y
    s.text(36, 862, "ΔPPL = method − historical common H; negative values indicate lower recorded PPL.", 20)
    s.text(36, 894, "Descriptive comparison: historical and checkpoint reconstruction protocols differ; no uncertainty estimates.", 19)
    s.text(36, 922, "Source: docs/RESULTS.md · Missing nonstructured 50% mean results are omitted, not interpolated.", 18)
    return s.render()


def method_figure():
    """Export the already documented computation graph, not a new algorithm."""
    s = SVG(1000, 750, "Reference forward and nested reconstruction",
            "Four fixed sparse reference branches propagate independently. Selected token statistics "
            "form Hessians for a separate reconstruction from original dense weights. No private weight residual is fitted.")
    s.parts.append('<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#667085"/></marker></defs>')
    nodes = {
        "d": (370, 18, ["原始稠密模型", "固定校准输入"]),
        "r": (370, 112, ["四个固定稀疏参考分支", "20% · 30% · 40% · 50%"]),
        "f": (370, 206, ["各参考分支独立前向", "各自权重与真实激活"]),
        "z": (36, 316, ["block入口对齐Token", "按差异选位置集合 M"]),
        "x": (370, 316, ["Hook收集投影输入 X", "各线性层的真实激活"]),
        "n": (704, 316, ["参考输出各自传播", "进入下一block"]),
        "h": (204, 432, ["按策略构造Hessian", "隔离 / 平均表示 / 平均H"]),
        "w": (650, 432, ["原始dense层权重", "目标重建的起点"]),
        "c": (400, 548, ["嵌套剪枝与OBS补偿", "冻结并继承已处理前缀"]),
        "e": (400, 652, ["各档目标快照", "完整WikiText2 PPL"]),
    }
    edges = [("d", "r"), ("r", "f"), ("f", "z"), ("f", "x"), ("f", "n"),
             ("z", "h"), ("x", "h"), ("h", "c"), ("w", "c"), ("c", "e")]
    for a, b in edges:
        x1, y1, _ = nodes[a]
        x2, y2, _ = nodes[b]
        s.parts.append(f'<path d="M{x1+130},{y1+64} C{x1+130},{y1+86} {x2+130},{y2-24} {x2+130},{y2}" fill="none" stroke="#667085" stroke-width="2" marker-end="url(#arrow)"/>')
    s.parts.append('<path d="M630,50 L980,50 L980,408 L780,408 L780,432" fill="none" stroke="#667085" stroke-dasharray="6 5" stroke-width="2" marker-end="url(#arrow)"/>')
    for key, (x, y, lines) in nodes.items():
        fill = "#eaf4fb" if key in ("h", "c") else "#f6f8fa"
        s.parts.append(f'<g id="node-{key}"><rect x="{x}" y="{y}" width="260" height="64" rx="8" fill="{fill}" stroke="#bac8d4"/>')
        for i, line in enumerate(lines):
            s.text(x + 130, y + 26 + i * 23, line, 18, "middle", i == 0)
        s.parts.append('</g>')
    s.text(36, 742, "统计融合不覆盖参考前向；“私有Token”不等于私有参数 ΔW。", 18)
    return s.render()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()
    tables = read_results(ROOT / "docs/RESULTS.md")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, svg in (("method-overview.svg", method_figure()),
                      ("ppl-comparison.svg", result_figure(tables))):
        path = args.output / name
        path.write_text(svg, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
