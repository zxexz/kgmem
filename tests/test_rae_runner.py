import argparse
import json
from pathlib import Path
import pickle
import random
import sys

import pytest

from experiments.qa.baselines.rae.run import IMPORT, SKIP, build_graph, patch_main, run, summary


def test_graph_dictionary_preserves_triples(tmp_path):
    triples = tmp_path / "triples.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ1\tP2\tQ3\nQ2\tP1\tQ3\n")
    output = tmp_path / "nested" / "graph.pkl"
    build_graph(argparse.Namespace(triples=[str(triples)], out=str(output)))
    with output.open("rb") as source:
        graph = pickle.load(source)
    assert graph == {"Q1": ["Q1\tP1\tQ2", "Q1\tP2\tQ3"], "Q2": ["Q2\tP1\tQ3"]}


def test_score_excludes_incomplete_attempts_and_unselected_cases(tmp_path):
    log = tmp_path / "shard.log"
    log.write_text("++1-th case++\nRaw:1-->partial\n"
                   "++1-th case++\nRaw:0-->done\nPruned:1-->done\n++End++\n"
                   "++2-th case++\nRaw:1-->other\nPruned:1-->other\n++End++\n")
    result = summary([log], [0, 2])
    assert result == dict(cases=1, raw=0.0, pruned=100.0, expected=2, missing=[2])


def test_patch_rejects_an_unknown_upstream_layout():
    with pytest.raises(ValueError, match="patch points"):
        patch_main("print('unrecognized main')\n")


def test_runner_samples_and_resumes_without_modifying_upstream(tmp_path):
    rae = tmp_path / "rae"
    (rae / "data").mkdir(parents=True)
    original = "import torch\n" + IMPORT + "def main():\n    for i in range(12):\n" + SKIP
    (rae / "main.py").write_text(original)
    (rae / "data" / "MQuAKE-CF-3k.json").write_text(json.dumps([{}] * 12))
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text("Q1\tExample\n")
    interpreter = tmp_path / "fake_python"
    interpreter.write_text(f"#!{sys.executable}\n" + '''import json, os
for case in json.load(open(os.environ["RAE_SAMPLE"])):
    print(f"++{case + 1}-th case++")
    print("Raw:1-->answer")
    print("Pruned:1-->answer")
    print("++End++")
''')
    interpreter.chmod(0o755)
    output = tmp_path / "results"
    args = argparse.Namespace(rae=str(rae), aliases=str(aliases), out=str(output), sample_size=6,
                              seed=0, shards=3, model="gpt2", mode="beam", python=str(interpreter), resume=False)
    run(args)
    expected = sorted(random.Random(0).sample(range(12), 6))
    assert json.loads((output / "sample.json").read_text()) == expected
    result = json.loads((output / "score.json").read_text())
    assert result == dict(cases=6, raw=100.0, pruned=100.0, expected=6, missing=[])
    assert (rae / "main.py").read_text() == original
    assert "WD5M_ENTITY_ALIASES" in (output / "main_sample.py").read_text()
    logs = {p.name: p.read_text() for p in output.glob("shard_*.log")}
    args.resume = True
    run(args)
    assert {p.name: p.read_text() for p in output.glob("shard_*.log")} == logs
