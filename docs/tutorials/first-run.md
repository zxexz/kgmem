# Run the memory demo

[Documentation](../index.md) / [Tutorials](index.md)

The demo uses the included `checkpoints/smollm2-v9.pt` with SmolLM2-135M. It runs on CPU and downloads the
base model and tokenizer on first use. You need Python 3.10 or later and a terminal in the repository root.

## Install and run

On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
mkdir -p runs
python examples/demo.py --adapter checkpoints/smollm2-v9.pt
```

The script uses CUDA if available. To force CPU, run
`env CUDA_VISIBLE_DEVICES="" python examples/demo.py --adapter checkpoints/smollm2-v9.pt`.
It writes `runs/memory_state.pt`; `--state` selects another output file.

## Read the graph answers

Under `[hand-written graph]`, each row prints the memory's answer beside the base model's answer.
Check these memory answers:

| Question | Expected memory answer |
| --- | --- |
| What is the capital of Velmora? | Drakenhold |
| Where does Talia live? | Pellin |
| In which country does Talia live? | Velmora |

The last question follows two edges: Talia lives in Pellin, and Pellin is in Velmora.
The `base LM:` column runs generation with memory disabled; its exact text can vary.

## Check the edited facts

Under `[after editing two edges at inference time]`, the script has changed Velmora's capital to Pellin
and Talia's home to Sorvik. Sorvik is in Quistan:

| Question | Before the edit | After the edit |
| --- | --- | --- |
| What is the capital of Velmora? | Drakenhold | Pellin |
| Where does Talia live? | Pellin | Sorvik |
| In which country does Talia live? | Velmora | Quistan |

No training runs between these questions. The loaded graph's edges determine the new answers.

## Check the replacement graph

Under `[swapped to a fresh random world]`, the script loads a graph with held-out names and different facts.
Compare each row's `memory:` answer with its `(gold: ...)` value.

## Read saved variables

The final section loads `runs/memory_state.pt` for a new session with no shared text context.
The script writes `kel = fox`, then `vom = tea`, then overwrites `kel` with `rice`.
It saves the buffer and reloads it before each question. The expected values are:

```text
Q: What is kel?   memory: rice   (gold:rice)
Q: What is vom?   memory: tea   (gold:tea)
```

These are separate sessions in one process: the questions contain no assignment text, and only the saved
scratchpad carries the values.

To use your own facts, continue with [the Python guide](../how-to/use-memory.md).
For the model's data flow, read [Architecture](../explanation/architecture.md).
