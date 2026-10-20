"""Aggregate local diagnostics; no PPL or statistical-significance claims."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def summarize(output):
    output = Path(output)
    cells = defaultdict(list)
    events = 0
    with (output/"events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            events += 1
            for horizon in event["horizons"]:
                for row in horizon["losses"]:
                    if row["donor"] == "no_compensation":
                        continue
                    key = (event["layer"], event["module"], event["stage"], horizon["groups"],
                           row["split"], row["donor"], row["evaluator"],
                           row["donor_uses_current_slice"], row["evaluator_uses_current_slice"])
                    cells[key].append(row)
    names = ("layer", "module", "stage", "window_groups", "split", "donor", "evaluator",
             "donor_uses_current_slice", "evaluator_uses_current_slice")
    rows = []
    for key, values in sorted(cells.items()):
        harmed = helped = 0
        for value in values:
            baseline = value["mse"]-value["delta_mse_vs_no_compensation"]
            tolerance = max(1e-12, 1e-6*baseline)
            harmed += value["delta_mse_vs_no_compensation"] > tolerance
            helped += value["delta_mse_vs_no_compensation"] < -tolerance
        rows.append(dict(zip(names, key), events=len(values), harmed_events=harmed,
            helped_events=helped, neutral_events=len(values)-harmed-helped,
            mean_delta_mse=sum(v["delta_mse_vs_no_compensation"] for v in values)/len(values)))
    with (output/"cross_rate_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else list(names))
        writer.writeheader()
        writer.writerows(rows)
    active = [r for r in rows if r["split"] == "heldout" and r["window_groups"] == 1
              and r["donor_uses_current_slice"] and r["evaluator_uses_current_slice"]
              and int(r["donor"][1:]) != round(r["evaluator"]*100)]
    count = sum(r["events"] for r in active)
    harmed = sum(r["harmed_events"] for r in active)
    report = dict(events=events, active_cross_rate_heldout_one_group_comparisons=count,
                  harmed_comparisons=harmed, harmed_fraction=harmed/count if count else None,
                  diagnostic="sampled rows/tokens; local WX_r target; no PPL",
                  tolerance="max(1e-12,1e-6 * no-compensation MSE), not a significance test")
    (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(summarize(parser.parse_args().output), indent=2))
