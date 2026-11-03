# kgmem inspector audit

Audit date 2026-10-29, against the inspector as of that day, running `checkpoints/archive/kgmem_adapter_v2.pt` on SmolLM2-135M.

## Status after the revision (same day)

The revision addressed the findings below except those marked *open*. These are the
2026-10-29 results, not a review of later versions. Verification used headless checks;
no interactive browser was available:
- all 8 examples load with 0 JavaScript errors;
- DOM checks for semantics;
- contrast computed from the CSS tokens;
- screenshots in light and dark, at 1600px and 900px.

Hover and click behaviour was reviewed in source; it was not exercised interactively.

| finding | what changed |
|---|---|
| C1 "frozen LM" | Four stages: base LM from a real memory-off pass, upper layers after injection, + readout, final. Each stage's own top 3 is kept. Log scale. |
| C2 own links | The cosine is shown next to the renormalised weight, with a label saying so. Links are faded unless some position uses the token as a mention. The summary reports the inherited head ("Velmora via `ora`@9"). |
| H1 heatmap scale | Absolute scale (0 to at least 1), an absolute "holds something" threshold of 0.5, and "No slot holds anything (largest norm 0.005)" for an empty memory. Near-empty rows are collapsed. |
| H2 adapter switch | Each adapter keeps its own persistent buffer and write log across switches. An answers-by-adapter strip shows the current prompt's answer from each adapter tried. |
| H3 stale views | Graph edits re-run the last prompt and show "Before the edit: X → after: Y". The highlight checks edges, not just names. Messages appear in the graph card. |
| H4 read/write match | "Which writes this read sees" lists overlap with earlier writes, including remembered ones (the server keeps a write log). The persistent table has a "last strong write" column. |
| H5 orientation | A clickable block diagram, cards in data-flow order numbered 1–4, a glossary with 28 ⓘ definitions, and "Walk me through it" with the numbers filled in. |
| H6 demonstrations | A "Teach it a fact it has never seen" example. Examples generated from the loaded graph. Demo examples reload the demo graph when needed. A warning when the prompt names no graph entity. |
| Medium: encodings | Pointer chart weighted by λ and faded when λ < 0.05. "← answer" only at weight and λ ≥ 0.3. Writes and arcs faded when β < 0.1. Slot tables sorted by norm and labelled as logit bias. Each lane labelled with its scale. |
| Medium: flow | Trace opens on the answer position. A "memory effect" lane (with minus without) at every position. Validation errors for: removing an edge that doesn't exist, Remember with memory off, an empty prompt. Warnings for an empty buffer and for truncated prompts. |
| Medium: scale | An overview strip when the sequence is wider than the view. Jump to memory-effect positions, writes, mentions or generated tokens. Graph search, neighbourhood focus, and labels only where relevant on graphs over 25 entities. |
| Accessibility | Muted text 5.3:1, the Run button 5.4:1, edges and borders 3.2:1, and a selection outline of 4.3:1 / 4.8:1. The timeline is one tab stop with arrow keys and a live summary. No focusable elements inside `role="img"`. Every chart is named. ⓘ buttons use `aria-describedby`. Table views for the timeline and heatmap. Focus moves to the timeline after choosing an example. *Open:* graph nodes are mouse-only; keyboard users have the search box and the edge table. |
| Low | One colour system: graph blue, writes orange, reads aqua, pipeline stages as an ordered blue ramp (validated). A summary bar under the timeline for narrow screens. Stage-1 adapters show their stage-1 metrics. The replay check moved into the scratchpad card. The duplicate message is removed. |
| Model 1: injection everywhere | Graph-gate supervision. `checkpoints/archive/kgmem_adapter_v3.pt` has injection 0.00 on question tokens and `lm_harm` 0.066 (was 3.16). *Open:* plain English is still damaged (+5.3 nats/token). The gate needs negative examples from ordinary text. |
| Model 2: empty-memory bias | Read gate: 3.41 → 0.19 logits (v3), 0.02 (v2). |
| Model 3: diffuse storage | Same-slot term for overwrites: 0.89 → 0.95. *Open:* writes are still soft (weights around 0.3–0.5), so overwrites are partial. v3 answers "fox" on the demo's `kel` overwrite. |
| Model 4 | Still not re-checked. |

## Method

The audit used an independent review and a measurement pass.

The independent reviewer inspected the running app and source for:
- all seven example prompts;
- Remember followed by Ask in a new session;
- graph edits (add an entity, set, remove, random world);
- adapter switching;
- error paths;
- light and dark themes, at 1600px and 900px.

The measurement pass covered:
- WCAG contrast ratios computed from the CSS colour tokens;
- the rendered DOM, for keyboard and screen-reader semantics;
- API latency, payload and render size for prompts up to 461 tokens.

Critical and High findings were re-checked against the live API. One unverified
model-behaviour claim is marked below. The temporary review screenshots are not included.

The review considered two audiences:
1. The author: an ML engineer debugging and explaining this architecture.
2. A technically literate learner who knows transformer basics and wants to understand how a frozen LM can be given a swappable knowledge graph and a writable scratchpad.

## Findings before the revision

For the author, the inspector provided:
- runs come back in 100–430 ms;
- a trace with replayed slot states checked against the block's final state;
- adapter provenance;
- observations of model behaviour (see "What the inspector revealed about the model").

The initial UI had four problems for learners:
- Two quantities have labels that misrepresent their meaning (C1, C2).
- The scratchpad scale makes empty memory look full (H1).
- Card order does not follow data flow, and terms lack definitions (H5).
- Demonstrations of graph edits require several manual steps (H6).

The proposed revision changes labels and scales, adds thresholds, and orders the
inspection steps by data flow.

---

## Critical: misleading labels

### C1. The "frozen LM" bar in the Next-token card is not the frozen LM
**What it shows:** what the frozen *upper* layers produce *after* the memory block has injected into the residual stream, minus the scratchpad readout bias (`Inspector.trace`, `p_raw`).

**Evidence (verified), at ":" in "What is the capital of Velmora?":**

| | top predictions |
|---|---|
| the bar labelled "frozen LM" | ` Dra` 0.0001, ` apple` 0.006, ` black` 0.009 |
| the real base model, memory off | ` Vel` 0.17, ` The` 0.17, ` It` 0.13 |

The graph injection's size relative to the residual (|Δ|/|h|) is 2.86 at this position.

**Why it matters:**
- The label attributes the post-injection distribution to the base model.
- The display hides the size of the injection's effect; the copy head (λ = 0.99) supplies most of the answer distribution.
- The candidate list keeps only the top 8 by the *final* probability, so tokens that ranked high at an earlier stage can be dropped.

**Fix:**
- Show four stages: **base LM (memory off)**, **upper layers after memory injection**, **+ scratchpad readout**, **final**. The base-LM stage needs one extra forward pass with memory off (~100 ms).
- Keep each stage's own top tokens.
- Use log probability or rank, so that small values stay visible.

### C2. Every token's "own link" looks near-certain
**What it shows:** per-token link weights from a softmax over only the top 4 keys, presented as if they were confidence.

**Evidence (verified):** ":" links to Orbek at 0.997. The Position card then reports "linked entity: Orbek" for a question about Velmora. The head inherited through mention attention is Velmora (1.0, via `ora`@9).

**Why it matters:** per-token linking and mention selection are separate steps. The
display conflates them and presents normalized top-k weights as confidence.

**Fix:**
- Show the raw similarity next to the renormalised weight, labelled "renormalised over the top 4".
- Dim a token's own link unless something attends to that token as a mention.
- In the Position card, report the *inherited* head ("Velmora, via `ora`@9").

---

## High: state and interpretation

### H1. The scratchpad heatmap's scale makes an empty memory look full
**What happens:**
- The colour range runs to each run's maximum.
- "Used" means at least 2% of that maximum.

On a graph question with an empty scratchpad, the legend reads "0 → 0.01" and shows "27 of 64 slots used" in fully saturated cells. After a real Remember the maximum is 38, and the chart looks the same.

**Fix:**
- Use an absolute colour range, for example 0 to max(run max, a typical write norm).
- Use an absolute threshold for "used".
- Print the range prominently.

### H2. Switching adapters erases remembered state without notice
**What happens:** switching loads the persistent buffer saved inside the adapter file. Verified: 54 slots → switch → 0 slots → switch back → still 0. Switching also reruns the prompt and overwrites the previous answer, so there is no way to compare adapters on the same memory.

**Fix:**
- Keep a buffer per adapter, or warn and offer to carry it over.
- Keep a strip showing the current prompt's answer from each adapter tried.

### H3. Views go stale after a graph edit
**What happens:**
- Only the graph card re-renders.
- The answer, timeline, traversal and inspector still show the old graph's results, with no stale marker.
- The highlight check compares entity and relation lists but not edges, so after **Set** the old traversal is drawn over the new edges.
- The "Run again" message appears in the Prompt card, about 1,500px off screen.

**Fix:** rerun the last prompt automatically (it takes 100–430 ms) and show the answer before and after the edit.

### H4. You can't see whether a read lands on the slot its variable was written to
**What happens:**
- The Position card shows only argmax slots.
- Verified on v2:
  - after Remember, the ` rice` write goes to slot 24 (weight 0.28);
  - in the new session, the `kel` read goes to slot 20 (0.34);
  - the answer is still right.
- The persistent card has no record of which write went where.
- Slot 21 still holds "fox" at the highest norm (38), even though `kel` was overwritten with rice.

Argmax slot agreement does not explain this recall. The README's "overwrites
exactly one row" does not describe this adapter (see "What the inspector revealed
about the model").

**Fix:**
- Show the overlap between the current read address and every earlier write address, including those from the last Remember.
- Annotate persistent slots: "last strong write: ` fox` via `kel`@1".

### H5. No route through the data flow
**What happens:**
- The page opens fully populated.
- The first thing shown is ten unexplained lanes: λ, pointer mass, 2-hop share, |Δ|/|h|, β and so on.
- The only architecture view is a layer stack at the bottom.
- Nothing shows the data flow linker → hops → halting → value/gates → copy head, or which card shows which step.

**Fix:**
- At the top, add a compact, clickable version of the README's block diagram that links each node to its card.
- Order the cards in data-flow order.
- Give every lane and stat a one-line ⓘ definition (see the glossary below).

### H6. The graph-edit demonstration requires manual steps
**What happens:**
- Teaching the model a fact it has never seen works in under a second. Add `Gwenvencor`, set Velmora's capital to it, ask again, and the answer is ` Gwenvencor` while the base LM's answer doesn't change.
- The demonstration requires four manual steps in the graph editor near the bottom.
- Loading a random world leaves the example prompts asking about Velmora, which no longer exists. The answer is "Quar Book Bomu", with no warning.

**Fix:**
- Add example prompts for "teach an unseen fact" and "edit, then ask again".
- Generate example questions from whatever graph is loaded.
- Warn when the prompt mentions no entity in the graph (a server-side string match).

---

## Medium

### Chart encodings
- **The pointer chart ignores λ.** It's normalised by its own mass, so ` Or` shows as a full bar (0.977) while λ = 0.001. Plot λ·p_pointer, or grey the chart out when λ < 0.05.
- **"← answer" has no threshold.** The graph labels Orbek "← answer" at weight 0.038 on a scratchpad question, and Quiron at 0.001 (also clipped) in a random world. Require a meaningful weight and λ.
- **Write addresses look real when β ≈ 0.** At ":" (β = 0.000) the card shows "write from ` Vel` … slot 2 · 1.00" and the timeline draws an orange write arc. Fade both when β < 0.1.
- **Near-empty slots decode with large logit biases.** Each row is RMS-normalised before decoding, so a slot with norm 0.04 decodes as "rice (10.0)", about as strongly as a slot with norm 38. The tables run 31–54 rows, sorted by slot number. Sort by norm, collapse rows below 5% of the maximum, and label the numbers as logit bias.
- **The next-token bars use one linear scale.** When the final probability is 0.99, the earlier stages render as 0.000 slivers. Use a log scale, or open the table by default.
- **The timeline's subtitle is wrong.** It says "each lane is on its own 0–max scale", but 8 of the 10 lanes are fixed at 0–1. Say which is which.

### Feedback and validation
- **Trace mode opens on the wrong position.** On "write and read in one sequence" the default selection is the last strong write, not the answer position.
- **The best on/off comparison isn't offered in Trace.** With memory off, p(rice) is 0.0001; with it on, 0.48.
- **Messages appear far from the action.**
  - Graph-editor errors ("already exists") and "Persistent scratchpad reset." go to the single status line at the top of the page.
  - The Reset button is at the bottom, far from the Remember/Ask controls.
  - Show feedback next to the control that caused it.
- **Missing input validation.**
  - Removing an edge that doesn't exist reports "Graph updated."
  - Remember with memory off claims "remembered N tokens" but writes nothing.
  - An empty prompt shows a raw torch `RuntimeError`.
  - Asking from an empty persistent scratchpad answers "K Or k" with no hint to run Remember first.
- **Long input is cut silently.** Prompts are truncated to 1,500 characters (a 5,000-character prompt became 375 tokens) with no notice.

### Scale (measured)

| prompt length | payload | round trip |
|---|---|---|
| 45 tokens | 0.12 MB | 0.5 s |
| 141 tokens | 0.37 MB | 0.3 s |
| 461 tokens | 1.2 MB | 1.4 s |

At 461 tokens, the timeline is about 17,000px wide, with about 9,000 heatmap cells
and about 4,600 lane cells. It lacks an overview or search. Add an overview with a
viewport marker and jumps to strong writes, mentions and answer positions.

---

## Accessibility (measured)

**Contrast** (WCAG AA: 4.5:1 for small text, 3:1 for graphical objects and UI components):

| element | light | dark |
|---|---|---|
| muted text (ticks, empty states, dimmed names) | **3.50** | 4.85 |
| white on the Run button | **4.42** | **3.64** |
| selected-position highlight | **1.11** | about 1.1 |
| unhighlighted graph edges | **1.75** | **1.48** |
| frozen-layer bars in the Model card | n/a | **1.24** |
| input and select borders | **1.75** | n/a |
| aqua bars (no visible value labels) | **2.74** (the table view is the relief) | n/a |

**Semantics:**
- **Hidden interactive elements.** The timeline (18 focusable tokens) and graph (9 focusable nodes) are inside `<svg role="img">`, which hides those elements from screen readers.
- **Tooltips not linked.** None use `aria-describedby`, so keyboard focus shows a visual tooltip but announces nothing.
- **Unnamed charts.** 3 of the 7 SVG charts have no accessible name.

**Keyboard:**
- There are 61 tab stops at 18 tokens (every bar, chip and halting segment), and the count grows with the prompt. Make the charts single tab stops with arrow keys inside.
- Enter and Space don't select a focused token.
- After choosing an example, focus stays in the select, so the advertised ← → keys do nothing until you click elsewhere.

**Text alternatives:** the timeline lanes and the slot heatmap have no table view; every other chart does.

---

## Low
- **Colour meanings change between cards.** Blue means frozen LM, hop 1, mention attention, link bars and slot norms; orange means readout, hop 2 and write. Give each memory a consistent hue (graph vs scratchpad) and keep it.
- **At 900px the layout stacks,** putting the selected position's details 1,000–2,500px below the timeline. Make the timeline sticky, or show a mini-summary under it.
- **The stage-1 adapter shows "eval – / – / –"** because its metric names differ.
- **The "slot replay matches the block" pill appears above the trace.** Move this implementation check into the Scratchpad card.
- **"Persistent scratchpad updated" appears twice** after Remember.
- **Random worlds (49 nodes) have overlapping labels.** Add search and neighbourhood focus.
- **The dark heat ramp preserves ordering:** high values are light.

---

## Proposed learning sequence

**1. Order inspection by data flow.**
Use this sequence:
1. the question;
2. which token is the mention, and why;
3. which relation, and which tail;
4. one hop or two;
5. how the answer gets spelled.

For the scratchpad, show write address, read address and decoded value in order.
A step-through mode can select the relevant token and card at each stage.

**2. Add paired comparisons.**
Extend the existing base-model/memory answer comparison to:
- memory on vs off at each position (p of the correct token);
- before vs after a graph edit;
- adapter A vs adapter B on the same prompt;
- remembered vs empty scratchpad.

The measured runs take under half a second.

**3. Correct display errors.** In priority order:
- the upper layers after injection shown as the "frozen LM" (C1);
- renormalised top-4 weights shown as confidence (C2);
- an empty memory looking full (H1);
- write arcs drawn where nothing was written (β ≈ 0);
- "← answer" on noise;
- slots presented as exact RAM when this adapter stores values diffusely (H4).

**4. Define terms beside their values.**

| term | one-line definition |
|---|---|
| mention attention | which earlier token the current position treats as the entity being asked about |
| hop | one relation followed in the graph |
| halting | how much weight the 1-hop and 2-hop answers each get |
| λ (copy gate) | how much of the next-token distribution comes from spelling a retrieved name instead of the LM |
| pointer mass | how much candidate-name weight is available to copy at this position |
| β | how strongly this position writes to the scratchpad |
| read / write address | which slots a read or write touches |
| readout bias | the logits a scratchpad read adds directly |
| \|Δ\|/\|h\| | how large the memory's change to the hidden state is, relative to the hidden state |
| persistent buffer | the scratchpad state that survives between sessions |

**5. Worked examples generated from the loaded graph,** including one with an entity the LM has never seen, and one 2-hop question whose first hop is ambiguous.

---

## What the inspector revealed about the model

These observations apply to the audited adapters and date. The proposed model
changes were investigation leads, not validated fixes.

1. **Large graph injection on question tokens.** |Δ|/|h| is 2.3–2.9, including at ` What`, where no lookup is wanted (verified). The value path substantially changes the LM's distribution, and the copy head has to carry every answer. Gate the injection to positions whose mention attention and linking are confident, or bound the injection's scale.
2. **An empty scratchpad still biases the output** (verified). `read_norm` scales a near-zero read up to unit size, and the readout adds about +2.7 logits to ` apple` / ` pear` on graph questions, raising p(apple) from 0.006 to 0.06. Scale the normalised read by min(1, |read| / τ).
3. **v2 stores scratchpad values diffusely** (verified).
   - `kel`'s two writes go to slots 21 (fox, weight 0.82) and 24 (rice, 0.28).
   - Fox remains in slot 21 with the largest norm.
   - The recall read is soft (0.34 on slot 20), and the answer comes out right from the mixture.
   - Training checks address consistency only against the last write. A follow-up could align earlier writes of the same variable and sharpen addresses with an entropy weight or temperature schedule; the measured recall was 0.79.
4. *(Reported by the reviewer, not re-checked.)* In the single-sequence trace, the answer position's read points at `tea` and `rice` rather than `kel`, with address overlap ≤ 0.022 against any write, yet p(rice) = 0.48.

---

## Retain in the revision
1. The base-LM answer beside the memory answer supports direct comparison.
2. The Graph traversal card and highlighted path (Velmora → leader → Brimwen → lives in → Sorvik, halting 0.99 two-hop) expose the model's top-k weighting.
3. The selected-token arcs and β lane helped identify the scratchpad behaviour above.
4. Keep the measured performance and trace checks visible:
   - 100–430 ms runs;
   - the slot replay checked against the block's own state;
   - adapter eval and provenance shown;
   - table views behind "Table";
   - shareable `#preset=` links.
5. Keep the numbered Remember → Ask-in-a-new-session examples and live persistent-state view.

## Suggested order of work

| # | item | covers |
|---|---|---|
| 1 | **Distribution and address displays:** four distribution stages with log scale; renormalisation labels and inherited head; absolute heatmap scale and thresholds; λ-weighted pointer; β-faded write arcs and address charts; "← answer" threshold; slot tables sorted by norm | C1, C2, H1, Medium encodings |
| 2 | **State handling:** rerun after graph edits; a buffer per adapter; feedback next to the control; validation instead of raw errors; a truncation notice | H2, H3, Medium flow |
| 3 | **Navigation and examples:** a clickable block diagram; cards in data-flow order; a glossary in place; walk-through mode; examples generated from the graph; on/off and before/after comparisons | H5, H6, Educational design |
| 4 | **Accessibility:** contrast tokens; charts as single tab stops with arrow keys; drop `role="img"` where children are interactive; `aria-describedby`; table views for the timeline and heatmap | Accessibility |
| 5 | **Scale:** an overview strip; jump-to navigation; graph search and neighbourhood focus | Medium scale, Low |
