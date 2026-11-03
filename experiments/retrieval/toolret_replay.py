"""Sample replay queries from a prepared ToolRet pool, excluding evaluation ids.

Inputs are the rows.jsonl and tools.jsonl produced by toolret_loso. Additional
id filters are explicit files, not assumed parts of a local directory layout.
"""
import argparse
import collections
import json
from pathlib import Path
import random


def ids_of(path):
    path = Path(path)
    if path.suffix == ".txt":
        return set(path.read_text().split())
    with path.open() as source:
        return {json.loads(line)["qid"] for line in source}


def select_training_rows(rows, n, seed=0, include=None, exclusions=None):
    if n < 1:
        raise ValueError("sample size must be positive")
    by_source = collections.defaultdict(list)
    reserved = {row["qid"] for row in rows if row["pool"] != "train"}
    for row in rows:
        if row["pool"] == "train" and (include is None or row["qid"] in include):
            by_source[row["source"]].append(row)
    if not by_source:
        raise ValueError("no eligible training rows")
    excluded = exclusions or {}
    bad = reserved.union(*excluded.values())
    count_ids = lambda ids: sum(row["qid"] in ids for pool in by_source.values() for row in pool)
    counts = dict(reserved_pools=count_ids(reserved),
                  external={name: count_ids(ids) for name, ids in excluded.items()},
                  union=count_ids(bad))
    eligible_count = sum(map(len, by_source.values()))
    available = {name: [row for row in pool if row["qid"] not in bad]
                 for name, pool in by_source.items()}
    available = {name: pool for name, pool in available.items() if pool}
    if not available:
        raise ValueError("all eligible training rows were excluded")
    rng = random.Random(seed)
    quota = n // len(available)
    selected = []
    for index, name in enumerate(sorted(available)):
        pool = available[name]
        size = quota + (n - quota * len(available) if index == len(available) - 1 else 0)
        selected.extend(rng.sample(pool, min(size, len(pool))))
    report = dict(n=len(selected), per_source=collections.Counter(row["source"] for row in selected),
                  eligible_train_rows=eligible_count, excluded_among_eligible_train_rows=counts)
    return selected, report


def write_outputs(path, rows, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")
    report_path = path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="LOSO rows.jsonl with source and pool fields")
    parser.add_argument("--tools", required=True, help="LOSO tools.jsonl")
    parser.add_argument("--include-ids", help="optional accepted ids: text lines or JSONL with qid")
    parser.add_argument("--exclude", nargs="*", default=[], help="extra evaluation ids to exclude")
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with Path(args.rows).open() as source:
        rows = [json.loads(line) for line in source]
    selected, report = select_training_rows(
        rows, args.n, args.seed, ids_of(args.include_ids) if args.include_ids else None,
        {path: ids_of(path) for path in args.exclude})

    from experiments.retrieval import toolret_prep
    with Path(args.tools).open() as source:
        tools = [json.loads(line) for line in source]
    vocab, weights = toolret_prep.bm25_index([tool["doc"] for tool in tools])
    top = []
    for start in range(0, len(selected), 500):
        top.extend(toolret_prep.bm25_top(vocab, weights, [row["query"] for row in selected[start:start + 500]]))
    output = [dict(qid=row["qid"], source=row["source"], query=row["query"], task=row["task"], gold=row["gold"],
                   candidates=[dict(tool_id=tools[index]["id"], doc=tools[index]["doc"]) for index in indices])
              for row, indices in zip(selected, top)]
    report["gold_in_bm25_32"] = round(sum(any(tools[index]["id"] in row["gold"] for index in indices)
                                        for row, indices in zip(selected, top)) / len(selected), 4)
    write_outputs(args.out, output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
