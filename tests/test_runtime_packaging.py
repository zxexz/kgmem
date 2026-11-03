import pytest

from experiments.statistics import build_runtime


def test_nonempty_output_directory_preserves_inputs(tmp_path, monkeypatch):
    output = tmp_path / "runtime"
    output.mkdir()
    store = output / "store.json"
    adapter = output / "adapter.pt"
    store.write_bytes(b'{"input": "store"}')
    adapter.write_bytes(b"input adapter")
    monkeypatch.setattr("sys.argv", [
        "build_runtime", "--store", str(store), "--adapter", str(adapter), "--out", str(output),
    ])

    with pytest.raises(SystemExit) as error:
        build_runtime.main()

    assert error.value.code == 2
    assert store.read_bytes() == b'{"input": "store"}'
    assert adapter.read_bytes() == b"input adapter"
    assert sorted(path.name for path in output.iterdir()) == ["adapter.pt", "store.json"]


@pytest.mark.parametrize("archive_name", ["store.json", "adapter.pt"])
def test_archive_cannot_overwrite_an_input(tmp_path, monkeypatch, archive_name):
    store = tmp_path / "store.json"
    adapter = tmp_path / "adapter.pt"
    output = tmp_path / "runtime"
    store.write_bytes(b'{"input": "store"}')
    adapter.write_bytes(b"input adapter")
    monkeypatch.setattr("sys.argv", [
        "build_runtime", "--store", str(store), "--adapter", str(adapter),
        "--out", str(output), "--tgz", str(tmp_path / archive_name),
    ])

    with pytest.raises(SystemExit) as error:
        build_runtime.main()

    assert error.value.code == 2
    assert not output.exists()
    assert store.read_bytes() == b'{"input": "store"}'
    assert adapter.read_bytes() == b"input adapter"
