"""Report completed PPL and exact measured replacement-error coverage, never estimates."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--threshold", type=float, choices=(.05, .1), default=.05)
    parser.add_argument("--all-mean", action="store_true")
    args = parser.parse_args()
    policy = f"true_checkpoint_relative_error_le_{args.threshold:g}_isolated_mean_anchor20"
    if args.all_mean:
        policy = "true_checkpoint_all_mean_isolated_mean_anchor20"
    title = "# 纯全平均：全部位置四路算术平均，无阈值" if args.all_mean else f"# 替换误差≤{args.threshold:.0%}用20%表示，其余四路平均"
    text = [title, "",
            "PPL越低越好；共用率以全部测量位置为分母，零范数位置不允许共用。", "",
            "| 类型 | PPL@20 | PPL@30 | PPL@40 | PPL@50 | 共用率(全部入口) | 共用率(排除首层embedding) | 不含PPL总秒数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    cdf_rows, bin_rows = [], []
    completed = 0
    for mode, title in (("unstructured", "非结构化"), ("2of4", "2:4")):
        folder = args.root / mode
        table = {}
        if (folder / "comparison.tsv").exists():
            for row in csv.DictReader((folder / "comparison.tsv").open(encoding="utf-8"), delimiter="\t"):
                routing_valid = (row["selection_rule"] == "all_mean"
                                 and float(row["anchor_share_fraction_all_inputs"]) == 0.) if args.all_mean else (
                                     float(row["share_error_threshold"]) == args.threshold)
                if (row["hessian_policy"] != policy or not routing_valid
                        or row["status"] != "completed" or int(row["nested_mask_violations"])):
                    raise ValueError("incorrect experiment or invalid result")
                if row["sparsity"] in table:
                    raise ValueError("duplicate rate")
                table[row["sparsity"]] = float(row["ppl_wikitext2"])
        completed += len(table)
        scopes = []
        share = ["0%", "0%"] if args.all_mean else ["待完成", "待完成"]
        path = folder / "replacement_error_summary.json"
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary["threshold"] != args.threshold:
                raise ValueError("wrong threshold")
            for index, name in enumerate(("all_layer_inputs", "excluding_embedding_input")):
                scope = summary[name]
                if scope is not None:
                    share[index] = f"{scope['anchor_share_fraction']:.6%}"
                    scopes.append((name, None, scope))
        trace = folder / "execution_trace.jsonl"
        if trace.exists():
            for line in trace.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                scopes.append(("layer_input", row["layer"], row["mask"]))
        for scope, layer, values in scopes:
            for field, dest in (("cdf", cdf_rows), ("bins", bin_rows)):
                for item in values.get(field, []):
                    dest.append(dict(mode=mode, scope=scope, layer=layer,
                                     total_tokens=values["total_tokens"],
                                     invalid_norm_tokens=values["invalid_norm_tokens"], **item))
        seconds = "待完成"
        if (folder / "timings.json").exists():
            seconds = f"{json.loads((folder / 'timings.json').read_text())['total_wall_excluding_ppl_seconds']:.2f}"
        ppls = [f"{table[r]:.8f}" if r in table else "待完成" for r in ("0.20", "0.30", "0.40", "0.50")]
        text.append("| " + " | ".join([title] + ppls + share + [seconds]) + " |")
    if args.all_mean:
        text += ["", "所有位置均值融合，不计算差异指标，无CDF/区间统计。",
                 f"已完成 {completed}/8 个PPL。", ""]
    else:
        text += ["", "CDF记录≤5%、≤10%、≤15%……≤100%，以及150/200/500/1000%；区间表另列>1000%尾部。",
                 "分位数见各模式replacement_error_summary.json，仅统计有效范数位置。末层未用于H的输出不进入整体统计。",
                 f"已完成 {completed}/8 个PPL；详细统计见replacement_error_cdf.tsv和replacement_error_bins.tsv。", ""]
    for filename, rows in (("replacement_error_cdf.tsv", cdf_rows), ("replacement_error_bins.tsv", bin_rows)):
        if rows:
            with (args.root / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
    (args.root / "summary.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
