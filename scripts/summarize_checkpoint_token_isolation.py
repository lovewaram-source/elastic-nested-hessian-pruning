"""Summarize completed real-checkpoint isolation measurements."""
import argparse
import csv
import json
from pathlib import Path


HISTORY = {
    "unstructured": {
        .2: (5.56156042, 5.55390626), .3: (5.90267799, 5.87659452),
        .4: (6.22839916, 6.20337076), .5: (6.60510868, 6.58272300),
    },
    "2of4": {
        .2: (5.91052147, 5.90984287), .3: (6.52338581, 6.50378078),
        .4: (7.29507053, 7.24646935), .5: (8.72178931, 8.61114292),
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--private-fraction", type=float, choices=(.1, .5), default=.1)
    parser.add_argument("--shared-fusion", choices=("anchor20", "mean_token", "isolated_mean_anchor20"), default="anchor20")
    parser.add_argument("--reference-root", type=Path, help="completed anchor-fusion run for direct PPL comparison")
    parser.add_argument("--reference-fraction", type=float, choices=(.1, .5), default=.1)
    args = parser.parse_args()
    prefix = "true_checkpoint_block_cosine" + (f"_{args.shared_fusion}" if args.shared_fusion != "anchor20" else "")
    policy = f"{prefix}_private_{round(args.private_fraction * 100):03d}"
    reference_policy = f"true_checkpoint_block_cosine_private_{round(args.reference_fraction * 100):03d}"
    lines = [f"# 真实 checkpoint 输出余弦筛选：{args.private_fraction:.0%}隔离，{args.shared_fusion}融合", "",
             "非结构化 α=1；2:4 α=0.25；Llama-2-7B；C4 64×4096，seed=0；完整 WikiText-2。",
             "历史 dense-H 与完整对齐为先前实验参照；本轮补偿执行策略不同，不计算同流程0%/100%收益保留率。", "",
             f"| 类型 | 稀疏率 | 历史dense-H | 历史完整对齐 | 原anchor20/{args.reference_fraction:.0%} | 本轮{args.shared_fusion}/{args.private_fraction:.0%} | 相对原anchor20的PPL降低 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    records = []
    completed = 0
    for mode, title in (("unstructured", "非结构化"), ("2of4", "2:4")):
        table = {}
        reference = {}
        if args.reference_root is not None:
            with (args.reference_root / mode / "comparison.tsv").open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if (row["hessian_policy"] != reference_policy
                            or float(row["private_fraction"]) != args.reference_fraction or row["status"] != "completed"
                            or int(row["nested_mask_violations"])):
                        raise ValueError("invalid anchor-fusion reference result")
                    rate = float(row["sparsity"])
                    if rate in reference:
                        raise ValueError("duplicate reference rate")
                    reference[rate] = float(row["ppl_wikitext2"])
        path = args.root / mode / "comparison.tsv"
        if path.exists():
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if (row["hessian_policy"] != policy or float(row["private_fraction"]) != args.private_fraction
                            or row["status"] != "completed" or int(row["nested_mask_violations"])):
                        raise ValueError("wrong experiment identity or invalid result")
                    rate = float(row["sparsity"])
                    if rate in table:
                        raise ValueError("duplicate rate result")
                    table[rate] = row
        for rate, (baseline, aligned) in HISTORY[mode].items():
            ppl = float(table[rate]["ppl_wikitext2"]) if rate in table else None
            old = reference.get(rate)
            delta = old - ppl if old is not None and ppl is not None else None
            completed += ppl is not None
            lines.append(f"| {title} | {rate:.0%} | {baseline:.8f} | {aligned:.8f} | "
                         + ("未提供" if old is None else f"{old:.8f}") + " | "
                         + ("待完成" if ppl is None else f"{ppl:.8f}") + " | "
                         + ("—" if delta is None else f"{delta:+.8f}") + " |")
            records.append(dict(pruning_type=mode, sparsity=rate, historical_dense_H=baseline,
                                historical_aligned_H=aligned, reference_ppl=old,
                                reference_fraction=args.reference_fraction,
                                shared_fusion=args.shared_fusion,
                                private_fraction=args.private_fraction, current_ppl=ppl,
                                ppl_reduction_vs_reference=delta))
    lines += ["", "| 类型 | 全部准备、重建与重载时间（秒，不含PPL） |", "|---|---:|"]
    for mode, title in (("unstructured", "非结构化"), ("2of4", "2:4")):
        path = args.root / mode / "timings.json"
        value = "待完成"
        if path.exists():
            value = f"{json.loads(path.read_text(encoding='utf-8'))['total_wall_excluding_ppl_seconds']:.2f}"
        lines.append(f"| {title} | {value} |")
    lines += ["", "耗时比较需核对各自protocol的执行布局；不同GPU卡数或存储布局不视为纯比例耗时对照。",
              f"已完成 {completed}/8 个 PPL 结果。", ""]
    markdown = "\n".join(lines)
    (args.root / "summary.md").write_text(markdown, encoding="utf-8")
    with (args.root / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(markdown)
    print(f"ALL_COMPLETE={completed == 8}")


if __name__ == "__main__":
    main()
