import sys

import pytest

from kgmem.strandedcpu import _strandedcpu_api, run_program, strandedcpu_graph


def checkout(path, api):
    package = path / "strandedcpu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "api.py").write_text(api)
    return path


@pytest.fixture
def isolated_imports(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    def names():
        return [name for name in sys.modules if name == "strandedcpu" or name.startswith("strandedcpu.")]

    saved = {name: sys.modules.pop(name) for name in names()}
    yield
    for name in names():
        sys.modules.pop(name)
    sys.modules.update(saved)


def test_source_checkout_can_be_selected_explicitly(tmp_path, monkeypatch, isolated_imports):
    current = checkout(tmp_path / "current", 'checkout = "current"\n')
    monkeypatch.setenv("KGMEM_STRANDEDCPU", str(current))

    assert _strandedcpu_api().checkout == "current"


def test_missing_dependency_is_not_reported_as_a_missing_checkout(tmp_path, monkeypatch, isolated_imports):
    current = checkout(tmp_path, "import kgmem_missing_test_dependency\n")
    monkeypatch.setenv("KGMEM_STRANDEDCPU", str(current))

    with pytest.raises(ModuleNotFoundError) as error:
        _strandedcpu_api()

    assert error.value.name == "kgmem_missing_test_dependency"


def test_unchecked_graph_resolves_auto_to_interpreter():
    try:
        _strandedcpu_api()
    except ImportError:
        pytest.skip("strandedcpu is not available")

    graph, scorer = strandedcpu_graph(range(3), ["successor"], checked=False)

    assert graph.tails("0", "successor") == ["1"]
    assert graph.tails("1", "successor") == ["2"]
    assert graph.tails("2", "successor") == []
    assert scorer.compute(0, 0)["backend"] == "interpreter"


def test_unchecked_cuda_requires_checked_execution():
    with pytest.raises(ValueError, match="use checked execution for CUDA"):
        run_program("y = x + 1;", {"x": 1}, backend="cuda")
