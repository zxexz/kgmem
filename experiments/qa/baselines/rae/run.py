"""Prepare and score a sampled run of RAE's original pipeline."""

import argparse
import collections
import json
import os
from pathlib import Path
import pickle
import random
import re
import subprocess
import sys


CASE = re.compile(r"\+\+(\d+)-th case\+\+")
IMPORT = "from model import Extract, Prune \n"
SKIP = "        if i < args.starting_line:\n            continue\n"

REQUEST_SETUP = '''import os, json, time, urllib.request, urllib.error, requests
_UA = "kgmem-rae-reproduction/1.0 (academic reproduction of RAE, CIKM 2024; low volume)"
_op = urllib.request.build_opener(); _op.addheaders = [("User-Agent", _UA)]; urllib.request.install_opener(_op)
_orig_req = requests.Session.request
def _req(self, *a, **k):
    k.setdefault("headers", {}); k["headers"].setdefault("User-Agent", _UA); return _orig_req(self, *a, **k)
requests.Session.request = _req
RAE_SAMPLE = set(json.load(open(os.environ["RAE_SAMPLE"])))
'''

LABEL_SETUP = '''
import model as _rae_model, utils_func as _rae_utils
from wiki_api import wikidata as _rae_wd
_LABEL = {}
with open(os.environ["WD5M_ENTITY_ALIASES"], encoding="utf-8") as _aliases:
    for _line in _aliases:
        _parts = _line.rstrip("\\n").split("\\t")
        if len(_parts) > 1:
            _LABEL[_parts[0]] = _parts[1]
def _id2entity(q):
    return _LABEL.get(q, q)
_live_entity2id = _rae_wd.entity2id
def _entity2id(q):
    for _attempt in range(7):
        try:
            return _live_entity2id(q)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(2 ** _attempt)
    return _live_entity2id(q)
for _module in (_rae_wd, _rae_model, _rae_utils):
    _module.id2entity = _id2entity
    if hasattr(_module, "entity2id"):
        _module.entity2id = _entity2id
id2entity = _id2entity
'''


def patch_main(source):
    if not source.startswith("import torch\n") or source.count(IMPORT) != 1 or source.count(SKIP) != 1:
        raise ValueError("RAE main.py does not match the expected patch points; check the upstream revision")
    return REQUEST_SETUP + source.replace(IMPORT, IMPORT + LABEL_SETUP).replace(
        SKIP, SKIP + "        if i not in RAE_SAMPLE:\n            continue\n")


def read_results(paths):
    results = {}
    for path in paths:
        case, current = None, None

        def finish():
            if case is None:
                return
            previous = results.get(case)
            if current["complete"]:
                if previous and previous["complete"]:
                    current["raw"] |= previous["raw"]
                    current["pruned"] |= previous["pruned"]
                results[case] = current
            elif previous is None:
                results[case] = current

        for line in Path(path).read_text().splitlines():
            match = CASE.search(line)
            if match:
                finish()
                case = int(match.group(1)) - 1
                current = dict(raw=0, pruned=0, complete=False)
            elif case is not None and line.startswith(("Raw:", "Pruned:")):
                name, value = line.split(":", 1)
                current[name.lower()] |= int(value.split("-->", 1)[0])
            if case is not None and "End++" in line:
                current["complete"] = True
        finish()
    return results


def summary(paths, sample=None):
    results = {case: result for case, result in read_results(paths).items() if result["complete"]}
    if sample is not None:
        results = {case: result for case, result in results.items() if case in sample}
    count = len(results)
    result = dict(cases=count, raw=round(100 * sum(r["raw"] for r in results.values()) / max(1, count), 2),
                  pruned=round(100 * sum(r["pruned"] for r in results.values()) / max(1, count), 2))
    if sample is not None:
        result.update(expected=len(sample), missing=sorted(set(sample) - results.keys()))
    return result


def build_graph(args):
    triples = collections.defaultdict(list)
    for path in args.triples:
        with Path(path).open() as source:
            for line in source:
                head, relation, tail = line.rstrip("\n").split("\t")
                triples[head].append(f"{head}\t{relation}\t{tail}")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        pickle.dump(dict(triples), output, protocol=4)
    print(json.dumps(dict(heads=len(triples), triples=sum(map(len, triples.values())))))


def run(args):
    rae, aliases, output = (Path(p).expanduser().resolve() for p in (args.rae, args.aliases, args.out))
    source = rae / "main.py"
    dataset = rae / "data" / "MQuAKE-CF-3k.json"
    for path in (source, dataset, aliases):
        if not path.is_file():
            raise FileNotFoundError(path)
    count = len(json.loads(dataset.read_text()))
    if not 1 <= args.sample_size <= count or args.shards < 1:
        raise ValueError("sample size must fit the dataset and shards must be positive")
    sample = sorted(random.Random(args.seed).sample(range(count), args.sample_size))
    output.mkdir(parents=True, exist_ok=True)
    sample_file = output / "sample.json"
    if sample_file.exists():
        if json.loads(sample_file.read_text()) != sample:
            raise ValueError("output contains a different sample; use another output directory")
        if not args.resume:
            raise ValueError("output already contains a run; pass --resume or use another directory")
    patched = output / "main_sample.py"
    patched.write_text(patch_main(source.read_text()))
    sample_file.write_text(json.dumps(sample) + "\n")
    completed = {case for case, row in read_results(output.glob("shard_*.log")).items() if row["complete"]}
    todo = [case for case in sample if case not in completed]
    env = os.environ.copy()
    env["WD5M_ENTITY_ALIASES"] = str(aliases)
    env["PYTHONPATH"] = str(rae) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    executable = str(Path(args.python).expanduser().resolve()) if os.sep in args.python else args.python
    processes, logs = [], []
    try:
        for shard in range(args.shards):
            cases = todo[shard::args.shards]
            if not cases:
                continue
            path = output / f"sample_{shard}.json"
            path.write_text(json.dumps(cases) + "\n")
            log = (output / f"shard_{shard}.log").open("a")
            logs.append(log)
            command = [executable, "-u", str(patched), "--model", args.model, "--mode", args.mode,
                       "--dataset", "MQuAKE-CF-3k"]
            processes.append(subprocess.Popen(command, cwd=rae, env={**env, "RAE_SAMPLE": str(path)},
                                               stdout=log, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    finally:
        for log in logs:
            log.close()
    result = summary(output.glob("shard_*.log"), sample)
    (output / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if any(codes) or result["missing"]:
        raise SystemExit("Some cases did not finish; inspect the shard logs and retry with --resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-graph", help="build RAE's Wikidata5m head dictionary")
    build.add_argument("--triples", nargs="+", required=True)
    build.add_argument("--out", required=True)
    execute = commands.add_parser("run", help="run sampled cases with RAE's own model and pruning code")
    execute.add_argument("--rae", required=True, help="RAE checkout with its data and edited graph prepared")
    execute.add_argument("--aliases", required=True, help="Wikidata5m entity alias TSV")
    execute.add_argument("--out", required=True)
    execute.add_argument("--sample-size", type=int, default=300)
    execute.add_argument("--seed", type=int, default=0)
    execute.add_argument("--shards", type=int, default=3)
    execute.add_argument("--model", default="gpt2", help="RAE model alias; gpt2 selects GPT-2 XL")
    execute.add_argument("--mode", default="beam")
    execute.add_argument("--python", default=sys.executable, help="Python executable for RAE's environment")
    execute.add_argument("--resume", action="store_true")
    score = commands.add_parser("score", help="score completed cases in existing logs")
    score.add_argument("logs", nargs="+")
    score.add_argument("--sample", help="sample.json for coverage checking")
    args = parser.parse_args()
    if args.command == "build-graph":
        build_graph(args)
    elif args.command == "run":
        run(args)
    else:
        sample = json.loads(Path(args.sample).read_text()) if args.sample else None
        print(json.dumps(summary(args.logs, sample)))


if __name__ == "__main__":
    main()
