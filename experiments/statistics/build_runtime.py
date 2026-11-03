"""Package the statistics-store read path under the runtime's xa/xm module names.

  python -m experiments.statistics.build_runtime --store store.json --adapter owid_v2_null_clean.pt --licence clean --out runs/statistics/runtime_build

Write pkg/ with the runtime/xa*.py read path and stripped kgmem modules under xm/, xa_tab.json
with the graph and aliases, and xa_blk.pt with the adapter. The AST pass removes
docstrings, comments and the PyKEEN scorer. Generated CUDA kernels default to off.
Fetching and experiment code stay in this repository. --out must be missing or empty;
--tgz adds a reproducible archive."""
import argparse
import ast
import json
import os
import shutil
from pathlib import Path

from experiments.statistics.owid_mem import Store
from experiments.statistics.owid_store import place
from kgmem.paths import ensure_parent, REPO_ROOT

DROP = {"PyKEENScorer", "kg_from_pykeen"}


class Strip(ast.NodeTransformer):
    def _body(self, node):
        b = node.body
        if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                and isinstance(b[0].value.value, str):
            b = b[1:]
        node.body = [n for n in b if not (isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in DROP)] \
            or [ast.Pass()]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._body(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._body(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._body(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def strip(src):
    out = ast.unparse(Strip().visit(ast.parse(src)))
    return out.replace("kgmem", "xm").replace("KGMEM_KERNELS", "XM_K") + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--licence", default="clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tgz", default="", help="write a reproducible archive of pkg/ at this path")
    ap.add_argument("--model_files", action="store_true",
                    help="also write xa_blk.safetensors + xa_blk.json (the block for a model folder; graphs off)")
    a = ap.parse_args()
    output_dir = Path(a.out)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        ap.error("--out must name a missing or empty directory")
    if a.tgz and Path(a.tgz).resolve() in {Path(a.store).resolve(), Path(a.adapter).resolve()}:
        ap.error("--tgz must not overwrite --store or --adapter")
    pkg = os.path.join(a.out, "pkg")
    os.makedirs(os.path.join(pkg, "xm"))
    for f in ("graph.py", "modules.py", "memlm.py", "kernels.py"):
        src = strip((REPO_ROOT / "kgmem" / f).read_text())
        if f == "kernels.py":                          # Default to the torch implementation in the engine.
            src = src.replace("os.environ.get('XM_K', '1')", "os.environ.get('XM_K', '0')")
            assert "os.environ.get('XM_K', '0')" in src
        open(ensure_parent(os.path.join(pkg, "xm", f)), "w").write(src)
    open(ensure_parent(os.path.join(pkg, "xm", "__init__.py")), "w").write(
        "from .graph import KnowledgeGraph, ExactScorer, GraphTensors\nfrom .modules import MemConfig, MemoryBlock\n"
        "from .memlm import MemoryLM\n")
    for f in ("xa.py", "xa_core.py", "xa_num.py"):
        src = (REPO_ROOT / "experiments" / "statistics" / "runtime" / f).read_text()
        src = src.replace("from .xa_core import", "from xa_core import").replace("from .xa_num import", "from xa_num import")
        src = src.replace("from kgmem", "from xm")
        open(ensure_parent(os.path.join(pkg, f)), "w").write(src)
    st = Store(a.store, a.licence)
    kg = st.kg
    alias = sorted({(n, i) for i, h in enumerate(st.heads) for n in {h["name"], place(h["name"])[0], *h["aliases"]}})
    json.dump(dict(e=kg.entities, r=kg.relations, t=[x for tr in kg.triples for x in tr],
                   v={str(k): v for k, v in st.value_of.items()}, a=alias),
              open(ensure_parent(os.path.join(a.out, "xa_tab.json")), "w"))
    shutil.copy(a.adapter, ensure_parent(os.path.join(a.out, "xa_blk.pt")))
    if a.model_files:                                  # The block as a model-folder file: tensors + plain metadata.
        import torch
        from safetensors.torch import save_file
        sd = torch.load(a.adapter, map_location="cpu", weights_only=True)
        save_file({k: v.contiguous() for k, v in sd["block"].items()}, os.path.join(a.out, "xa_blk.safetensors"),
                  metadata={"format": "pt"})
        json.dump(dict(cfg=sd["cfg"], layer_idx=sd["layer_idx"], graphs=False),
                  open(os.path.join(a.out, "xa_blk.json"), "w"), indent=1, sort_keys=True)
    banned = ("owid", "OWID", "store", "Store", "statistic", "kgmem", "zxexz", "Eric", "eric", "Marriott",
              "marriott", "anyscale", "/Users/", "/mnt/", "jev", "esbar", "strandedcpu")
    files = [f for f in os.listdir(pkg) if f.endswith(".py")] + \
        [os.path.join("xm", g) for g in os.listdir(os.path.join(pkg, "xm")) if g.endswith(".py")]
    texts = {f: open(os.path.join(pkg, f)).read() for f in files}
    if a.model_files:
        texts["xa_blk.json"] = open(os.path.join(a.out, "xa_blk.json")).read()
    words = sorted({w for w in banned for t in texts.values() if w in t})
    print("runtime written:", sorted(os.listdir(pkg)), "| names left:", words or "none")
    if words:
        raise SystemExit(f"names left in the runtime: {words}")
    if a.tgz:
        pack(pkg, a.tgz)


def pack(src, path):
    """Archive sorted entries with fixed metadata and a timestamp-free gzip header."""
    import gzip
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for root, dirs, files in sorted(os.walk(src)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for f in sorted(files):
                full = os.path.join(root, f)
                info = tar.gettarinfo(full, arcname=os.path.relpath(full, src))
                info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
                with open(full, "rb") as fh:
                    tar.addfile(info, fh)
    with open(ensure_parent(path), "wb") as out, gzip.GzipFile(fileobj=out, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue())


if __name__ == "__main__":
    main()
