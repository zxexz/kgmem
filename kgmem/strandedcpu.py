"""Run graph relations as C programs with strandedcpu.

  from kgmem.strandedcpu import strandedcpu_graph
  kg, scorer = strandedcpu_graph(range(121), ["square", "successor", "digit sum"])
  mlm.load_graph(kg, scorer=scorer)
  mlm.generate("Q: What is the square of 7?\\nA:")

A program reads global x and writes global y. Entity names are integers such as
"37". The scorer caches each (head, relation) result. Checked faults and results
outside the graph return no tail. Unchecked execution may wrap or raise.

Checked runs use the CPU interpreter or CUDA ISA kernel. The analytical
transformer is a separate reference backend, selected with backend="transformer".

DOMAINS bounds the bundled programs for 16-bit signed arithmetic. PROGRAMS2
contains two-input programs for direct callers; graph relations take one input.

Install strandedcpu or set KGMEM_STRANDEDCPU to a source checkout root.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time

import torch

from .graph import KnowledgeGraph

# Programs may modify x; both x and y are globals in the generated C source.
PROGRAMS: dict[str, str] = {
    "successor": "y = x + 1;",
    "predecessor": "y = x - 1;",
    "double": "y = x + x;",
    "half": "y = x / 2;",
    "square": "y = x * x;",
    "cube": "y = x * x * x;",
    "digit sum": "y = 0; while (x > 0) { y = y + x % 10; x = x / 10; }",
    "reverse": "y = 0; while (x > 0) { y = y * 10 + x % 10; x = x / 10; }",
    # Count divisor pairs in O(sqrt(x)); scanning to x took over a million instructions near 2^15.
    "number of divisors": ("int d; y = 0; d = 1; while (d <= x / d) { if (x % d == 0) { y = y + 2; "
                           "if (d == x / d) { y = y - 1; } } d = d + 1; }"),
    # Avoid d*d: setting d=x to end the search overflowed the next loop test for x >= 181.
    "smallest prime factor": "int d; d = 2; y = x; while (d <= x / d) { if (x % d == 0) { y = d; d = x; } d = d + 1; }",
    "collatz steps": "y = 0; while (x > 1) { if (x % 2 == 0) { x = x / 2; } else { x = 3 * x + 1; } y = y + 1; }",
}


LIMIT = 32767  # Largest signed 16-bit integer.


def _collatz_peak_ok(x: int) -> bool:
    while x > 1:
        x = x // 2 if x % 2 == 0 else 3 * x + 1
        if x > LIMIT:
            return False
    return True


# These bounds apply at N=16. Checked execution also catches faults inside a program.
DOMAINS = {
    "successor": lambda x: x < LIMIT,
    "predecessor": lambda x: x > -LIMIT - 1,
    "double": lambda x: abs(x) <= LIMIT // 2,
    "half": lambda x: True,
    "square": lambda x: abs(x) <= 181,
    "cube": lambda x: abs(x) <= 31,
    "digit sum": lambda x: x >= 0,
    "reverse": lambda x: x >= 0 and int(str(x)[::-1]) <= LIMIT,
    "number of divisors": lambda x: x >= 1,
    "smallest prime factor": lambda x: 2 <= x < LIMIT,
    "collatz steps": lambda x: 1 <= x and _collatz_peak_ok(x),
}

# Two-input programs for direct callers, with 16-bit domain bounds.
PROGRAMS2: dict[str, str] = {
    "gcd": "int t; while (z != 0) { t = z; z = x % z; x = t; } y = x;",
    "lcm": "int a; int b; int t; a = x; b = z; while (b != 0) { t = b; b = a % b; a = t; } y = x / a * z;",
    "power": "int i; y = 1; i = 0; while (i < z) { y = y * x; i = i + 1; }",
    "remainder": "y = x % z;",
}
DOMAINS2 = {
    "gcd": lambda x, z: x >= 0 and z >= 0,
    "lcm": lambda x, z: x >= 1 and z >= 1 and x * z // math.gcd(x, z) <= LIMIT,
    "power": lambda x, z: z >= 0 and all(abs(x) ** i <= LIMIT for i in range(z + 1)),
    "remainder": lambda x, z: x >= 0 and z > 0,
}


def source(body: str, inputs=("x",)) -> str:
    """Wrap a body in main(), moving local declarations before statements for the C89 compiler."""
    decls = " ".join(re.findall(r"int \w+;", body))
    rest = re.sub(r"int \w+;\s*", "", body)
    globs = " ".join(f"int {v};" for v in (*inputs, "y"))
    return f"{globs} int main() {{ {decls} {rest} return y; }}"


def coerce(name: str) -> int | None:
    """Parse a signed integer entity name, or return None."""
    s = name.strip()
    return int(s) if re.fullmatch(r"[+-]?\d+", s) else None


def name_of(value: int) -> str:
    """Format an integer as an entity name."""
    return str(value)


def _strandedcpu_api():
    path = os.environ.get("KGMEM_STRANDEDCPU")
    if path:
        path = os.path.abspath(os.path.expanduser(path))
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        from strandedcpu import api
    except ModuleNotFoundError as e:
        if e.name not in ("strandedcpu", "strandedcpu.api"):
            raise
        raise ImportError("Install strandedcpu or set KGMEM_STRANDEDCPU to a source checkout root") from e
    return api


def run_program(body: str, inputs: dict, backend: str = "interpreter", N: int = 16, trace: bool = False) -> dict:
    """Return y, instruction count, elapsed seconds and optional trace for an unchecked run."""
    if backend == "auto":
        backend = "interpreter"
    if backend not in ("interpreter", "transformer"):
        raise ValueError("Unchecked execution supports interpreter or transformer; use checked execution for CUDA")
    api = _strandedcpu_api()
    t0 = time.perf_counter()
    out = api.run_function(source(body, tuple(inputs)), dict(inputs), N=N, backend=backend, trace=trace)
    res = dict(y=int(out["y"]), steps=int(out["__steps"]), seconds=time.perf_counter() - t0, backend=backend)
    if trace:
        res["trace"] = out.get("__trace", [])
    return res


def _checked(out: dict, seconds: float) -> dict:
    # Dummy-option mismatches do not invalidate the computed value.
    ok = out["abstain_reason"] in (None, "no_match", "multi_match")
    return dict(y=out["value"] if ok else None, steps=int(out["steps"]), seconds=seconds,
                backend=f"checked {out['backend']}", fault=None if ok else out["abstain_reason"],
                detail=None if ok else out.get("fault_detail"))


def run_checked(body: str, inputs: dict, N: int = 16, backend: str = "interpreter") -> dict:
    """Run checked arithmetic; faults return y=None with fault and detail fields.

    Backends: interpreter, cuda, or auto (CUDA when available, interpreter otherwise).
    """
    api = _strandedcpu_api()
    t0 = time.perf_counter()
    out = api.score_options(source(body, tuple(inputs)), dict(inputs), {"_": "0"}, N=N, result="y", backend=backend)
    return _checked(out, time.perf_counter() - t0)


def run_checked_batch(jobs: list[tuple[str, dict]], N: int = 16, backend: str = "auto") -> list[dict]:
    """Run a batch with arithmetic checks. Split elapsed time evenly across returned records."""
    api = _strandedcpu_api()
    t0 = time.perf_counter()
    outs = api.score_options_batch([dict(program=source(b, tuple(i)), inputs=dict(i), option_texts={"_": "0"})
                                    for b, i in jobs], N=N, result="y", backend=backend)
    each = (time.perf_counter() - t0) / max(1, len(jobs))
    return [_checked(o, each) for o in outs]


def run_relation(body: str, x: int, backend: str = "interpreter", N: int = 16, trace: bool = False) -> dict:
    """Run a relation with x as its only input."""
    return run_program(body, {"x": x}, backend, N, trace)


class StrandedCPUScorer:
    """Compute tails from programs and cache each (head, relation) result.

    Relations without programs use fallback when supplied, usually an ExactScorer.
    """

    def __init__(self, kg: KnowledgeGraph, programs: dict[str, str] | None = None, backend: str = "auto",
                 N: int = 16, max_tails: int = 4, fallback=None, checked: bool = True):
        self.kg, self.programs = kg, dict(programs or PROGRAMS)
        self.backend, self.N, self.M, self.fallback = backend, N, max_tails, fallback
        self.checked = checked and backend in ("auto", "interpreter", "cuda")
        self.cache: dict[tuple[int, int], dict | None] = {}
        self.runs = self.hits = 0

    def compute(self, h: int, r: int) -> dict | None:
        """Return a cached result, computing it on a miss. Unhandled pairs return None."""
        key = (h, r)
        if key in self.cache:
            self.hits += 1
        else:
            self.compute_many([key])
        return self.cache[key]

    def compute_many(self, pairs) -> None:
        """Compute uncached pairs in one batch."""
        todo, jobs = [], []
        for key in dict.fromkeys(pairs):
            if key in self.cache:
                continue
            name = self.kg.relations[key[1]]
            body = self.programs.get(name)
            x = coerce(self.kg.entities[key[0]])
            res = None
            if body is not None and x is not None:
                if DOMAINS.get(name, lambda v: True)(x):
                    todo.append(key); jobs.append((body, x))
                    continue
                res = dict(y=None, steps=0, seconds=0.0, backend=self.backend, fault="outside_domain", tail=None)
            self.cache[key] = res
        if not jobs:
            return
        if self.checked:
            outs = run_checked_batch([(b, {"x": x}) for b, x in jobs], self.N, self.backend)
        else:
            outs = [run_relation(b, x, self.backend, self.N) for b, x in jobs]
        self.runs += len(jobs)
        for key, res in zip(todo, outs):
            res["tail"] = self.kg.ent_index.get(name_of(res["y"])) if res["y"] is not None else None
            self.cache[key] = res

    @torch.no_grad()
    def __call__(self, h: torch.Tensor, r: torch.Tensor):
        P = h.numel()
        tails = torch.zeros(P, self.M, dtype=torch.long)
        probs = torch.zeros(P, self.M)
        pairs = list(zip(h.tolist(), r.tolist()))
        prog = [i for i, (_, ri) in enumerate(pairs) if self.kg.relations[ri] in self.programs]
        rest = [i for i, (_, ri) in enumerate(pairs) if self.kg.relations[ri] not in self.programs]
        new = {pairs[i] for i in prog} - self.cache.keys()
        self.compute_many(new)
        self.hits += len(prog) - len(new)
        for i in prog:
            res = self.cache[pairs[i]]
            if res is not None and res["tail"] is not None:
                tails[i, 0], probs[i, 0] = res["tail"], 1.0
        if rest and self.fallback is not None:
            idx = torch.tensor(rest)
            ft, fp = self.fallback(h[idx], r[idx])
            tails[idx], probs[idx] = ft.cpu(), fp.cpu()
        return tails.to(h.device), probs.to(h.device)


def strandedcpu_graph(values, relations, programs: dict[str, str] | None = None, backend: str = "auto",
               N: int = 16, max_tails: int = 4, checked: bool = True) -> tuple[KnowledgeGraph, StrandedCPUScorer]:
    """Build an integer graph and program scorer; materialize valid edges for listing and drawing."""
    programs = dict(programs or PROGRAMS)
    missing = [r for r in relations if r not in programs]
    if missing:
        raise KeyError(f"no strandedcpu program for {missing}; known: {sorted(programs)}")
    kg = KnowledgeGraph([name_of(v) for v in values], list(relations))
    scorer = StrandedCPUScorer(kg, programs, backend, N, max_tails, checked=checked)
    pairs = [(hi, ri) for ri in range(len(kg.relations)) for hi in range(len(kg.entities))]
    scorer.compute_many(pairs)
    for hi, ri in pairs:
        res = scorer.cache[(hi, ri)]
        if res is not None and res["tail"] is not None:
            kg.triples.append((hi, ri, res["tail"]))
    kg._version += 1
    return kg, scorer
