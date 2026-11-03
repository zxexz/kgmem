# Computed relations with strandedcpu

[Experiments](../../index.md) / [Appendices](../../explanation/index.md)

The [current integration](../../explanation/strandedcpu.md) explains the scorer and execution boundary.
The measurements below use its cached integer-relation graph. Timings are historical observations,
not package performance guarantees.

## Execution measurements

On the 0..199 graph, 2,010 runs took 0.47 s with the CUDA kernel and 1.26 s with the interpreter on an
H100 node, and 0.9 s on a Mac. These timings belong to the recorded runtime and hardware.
`PROGRAMS` contains eleven relations: successor, predecessor, double, half, square, cube, digit sum,
reverse, number of divisors, smallest prime factor and collatz steps. The transformer reference run
of `square(9)` executed 23 instructions in 1.5 s and returned the same answer.

## Selecting a relation by name

The arithmetic relation names do not occur in adapter training. The following evaluation asks
“What is the {relation} of {x}?” for successor, predecessor, double, square, digit sum and smallest prime
factor, 10 questions each on a 0..199 graph.

| adapter | what changed | overall |
|---|---|---|
| v3 | six fixed relations | 0.07 |
| v4 | `RelWorld` training on ~130 sampled relation names | 0.32 |
| v5 | + `RelationMentionLinker` | 0.62 |
| v6 | + contrast pairs, multi-word names, question KL; shipped with `relm_raw_cos=30` | 0.83 |
| v7 | + the changes below | 0.95 |
| v9 | + halting, committing to a name, short numbers, exact names (see [v9](adapter-development.md#v9)) | **1.00** (all 911 one-hop questions) |

Five two-hop compositions, 10 questions each, test digit sum∘square, successor∘double,
square∘predecessor, double∘smallest prime factor and smallest prime factor∘successor.
Accuracy rises from 0.06 in v6 to 0.78 in v7, 0.80 in v8 through `carry_share`, and 1.00 in v9.
`python -m experiments.synthetic.eval_adapters <adapter> --only strandedcpu` produces these evaluations;
`--only strandedcpuwide` asks every one-hop question in the graph.

### Names with similar embeddings

v5 scored “ successor” as successor 0.43 and predecessor 0.44. The frozen LM placed the antonym keys
close together, and training had not required their separation. Sampling contrast pairs such as
older/younger sister and previous/next owner into the same `RelWorld` raised successor accuracy from
0.0 to 0.8.

Most v5 training relations had one word. It linked “digit sum” through “ digit”, making the word after
“the” a workable shortcut. Giving about a third of v6's relations multi-word names raised digit sum
from 0.0 to 0.9. v7's `--relm_first` also matches the first-word key and supervises attention over the
whole relation name.

`relm_raw_cos` adds the frozen LM's token-to-key cosine to the learned relation score, taking the
better of the first- and last-token keys. v7 trains at 30. Its learned score still overweights “digit
sum” and “square” for “ double”, which scores highly against many relation keys. Raising the inference
setting to 60 improves double from 0.5 to 0.9 without changing the standard evaluation scores.

### Hop count and order

v6 assigns 0.97 of its halting weight to one hop on “the digit sum of the square of x”. v7's `--rel_halt`
adds relation confidence and overlap between the hops' mention attention to the halting features.
Different named relations can then support a two-hop decision.

On “the successor of the double of 7”, v7 sometimes makes both hops attend to “ successor”.
`hop_order=0.4` corrects this case when several nearby relation names are present: if all hops choose
the same name, the first hop takes the name nearest the head mention and the second takes the next.
This matches both “the R2 of the R1 of X” and “X's R1's R2”. An early version also used relation words
from earlier questions in the context, reducing graph two-hop accuracy to 0.61. Restricting the search
to 8 tokens around the trained attention positions removes that failure and leaves the standard eval
unchanged.

## Answer completion

A number's first space token could trigger a new lookup whose answer outweighed the carried answer,
for example 21 over 3. `carry_fresh=0.3` limits the fresh candidates' weight while a name is being emitted.

<a id="ending-a-retrieved-name"></a>

### Ending a retrieved name

The early completion rule counted a name as retrieved only when its unnormalized weight exceeded 0.05.
Those weights multiply entity, relation, tail and halting weights. In all 8 run-on answers across
3 seeds, the intended name ranked first but weighed 0.05 or less, so it was never marked complete.
The pointer then began another name, such as “A: Velcor Los”. `carry_share=0.2` instead uses a name's
share of the total candidate weight, both to carry it and to recognize completion.

On v8 over 3 seeds × 8 worlds, that rule changes graph accuracy from 0.99 / 0.97 to 1.00 / 0.99.
Requiring the answer to end changes the corresponding scores from 0.97 / 0.92 to 0.98 / 0.93.
The strandedcpu two-hop score rises from 0.78 to 0.80; the other `experiments/synthetic/eval_adapters.py` measurements do not change.

`checkpoints/archive/kgmem_adapter_v7.pt` records its three inference settings in `meta["inference_changes"]`.
On the standard eval over 6 worlds, those settings give graph accuracy 0.99 / 0.98 and unseen-relation
accuracy 0.95 / 1.00; the training settings alone give 1.00 / 1.00 and 0.91 / 0.93.
`checkpoints/archive/kgmem_adapter_v8.pt` keeps v7's graph path and settings, adding `carry_share=0.2`.
`checkpoints/smollm2-v9.pt` also adds `carry_commit` and
`exact_link=0.9`, with retrained halting and end gates; see [the v9 experiments](adapter-development.md#v9).

The LM still sometimes preferred a period after a completed name. With `carry_share`, requiring a line
end reduces v8's graph scores from 1.00 / 0.99 to 0.98 / 0.93. Across 3 seeds, 18 of the 21 unfinished
answers are followed by “.”; 10 of the first 14 are currency questions such as “Which money do they use
in X's country?” The base model assigns the line end probability 0.01 to 0.18. Graph injection lifts it
on average from about 0.4 to 0.98 for one hop and 0.93 for two hops, but does not settle these cases.

Training only the graph value path did not solve stopping. After 600 steps from v8 with
`--only_kg_value --stop_weight 4` or 10, `--carry_share 0.2` and v8's flags, two-hop accuracy with the line
end stays at 0.929 or reaches 0.933. Both runs add 0.002 nats/token of question-text harm; weight 10
also reduces strandedcpu double from 0.9 to 0.8. The value path does not receive the pointer's
`completed` signal. v9's `--end_gate` uses that signal to propose the line end and raises two-hop
accuracy with the line end from 0.929 to 1.000.

## Still open

Four of about 370 v9 unseen-relation one-hop questions fail on “assistant coach” (three cases) or
“record label”. Exact-name priors fire where a name ends. A hop attending to “ assistant” rather than
“ coach” misses the prior. Propagating it backward would let an earlier state depend on later tokens,
which incremental KV-cached decoding cannot reproduce. Retraining relation-mention attention is another
option, but it also changes the calibration used by `hop_order`.

`hop_order` measures relation-likeness after the learned RMSNorm. In a run that retrained the whole
graph path, “?” and “ of” scored 0.48 and 0.58, compared with 0.24 and 0.37 in v7, and the rule split
one-hop questions. Raw or centered frozen-state cosines also failed to separate relation words from
“of” and “the”. Retraining this path therefore requires recalibrating or replacing the rule.
`experiments/synthetic/calibrate_hop_order.py <adapter>` measures both groups; its threshold is 0.39 on v9, which ships 0.4.
