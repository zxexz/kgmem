# Statistics store experiments

[Experiments](../index.md) / [Appendices](index.md)

`experiments/statistics/owid_store.py` tests (place, indicator, year) lookups against a structured store of Our World in Data
series. MMLU and MMLU-Pro `global_facts` items informed the problem type; they were not used for training,
tuning or selection. The series families below were specified before training. Each selected series
includes every country, region and year from 1900 that OWID publishes.

| Family | Why | Series (held out of training whole: *italic*) |
|---|---|---|
| development | access and adoption shares of a population: the most common kind of country-year development statistic | internet use; electricity access; clean cooking fuels; safely managed drinking water; safely managed sanitation; mobile subscriptions per 100; *literacy* |
| health | population health outcomes and coverage | child mortality; life expectancy; HIV prevalence (15-49); adult obesity; tobacco use; vaccination coverage for HepB3, PCV, Pol3, rubella, rotavirus, DTP3, *Hib3*, *IPV1*, *measles (MCV1)* |
| conflict | military spending, conflict deaths and weapons stockpiles | military spending (% of GDP); deaths in one-sided violence, non-state, extrasystemic, intrastate and interstate conflicts; *nuclear warheads* |
| economy | income, poverty, prices, trade and labour | GDP per capita (Maddison; World Bank PPP); extreme poverty ($3 a day); trade (% of GDP); unemployment; agricultural employment share; *inflation* |
| demography | population size, fertility and urbanisation | population; urban share; *fertility rate* |
| governance | survey- and expert-coded governance and attitude measures | Corruption Perceptions Index; V-Dem electoral democracy index; life satisfaction (Cantril ladder); trust in others; *women in parliament* |
| environment | emissions, energy mix and land cover | renewables' share of electricity; forest share of land; *CO2 per capita* |

The full store has 46 series, 252 places and 382,045 values. Places include countries, OWID continents,
World Bank income groups and World. Values are rounded to three significant figures, producing 11,011
distinct value nodes. A seeded fifth of each family is held out from training. IHME causes of death were
excluded because OWID refused the non-redistributable download; Pew Global Attitudes toplines required
an account.

The store records `licences` and `licence_class` from OWID's origin metadata. Of the 46 series,
23 are attribution-only or public domain (`clean`). The other 23 carry NC, ND, producer-specific terms
or a bare copyright line (`restricted`). These include WHO NC-SA series and vaccination series marked
“© WHO”, SIPRI, the Corruption Perceptions Index's ND terms, the World Happiness Report, the Integrated
Values Surveys, and portions of child mortality, life expectancy, agricultural employment, renewables
and forest cover.

OWID's metadata classes a series by OWID's own terms, not by the producer's. Within `clean`, the two V-Dem series are
CC BY-SA 4.0 (derivatives share the licence). The producers' own pages also restrict two more series. UNAIDS's terms
allow personal, non-commercial use only. The Federation of American Scientists' nuclear-forces page states no licence
beyond a copyright line. `--licence clean_no_vdem` drops the V-Dem series (21 series, 15,381 entities, 148 relations).
Dropping a series rebuilds the value table only; the block's weights are unchanged. On the questions both tables share,
the rebuilt table changes no read used at τ=0.18: held keys 973 of 979 used in both, 0 choice changes; unknown series 63
used in both, 0 changes. It fires on 0 of 6,471 out-of-domain questions, passes the isolated release gate and takes
about 30 minutes to build and test on one H100. Held-series reads, all abstained, change their unused choice on 127 of
1,217 questions because the relation softmax loses two entries.

## Questions and abstention

The graph has two hops: `place --series--> place-series node --year--> value`. Generated questions use
three natural phrasings per series and six generic ones; one natural and two generic phrasings are
excluded from training. Each question has four numeric options: the gold value and nearby values from
the series or scaled distractors.

The five initial evaluation sets are dev (trained keys and wording), fresh wording, held-out keys,
held-out series and missing keys. Held-out keys are a tenth of each trained series' (place, year) pairs:
present in the graph, never asked during training. Missing keys ask for a trained series at a place and
year with no stored value, where the store should abstain.

Before training, questions pass a 13-gram check, an exact option-set check against public `global_facts`
items, and an external decontamination gate. That gate flagged 197 of 210,000 questions for word overlap.
Those questions were removed. Templates with three or more flags for a series were also banned for
that series in `owid_store.BANNED`.

`experiments/statistics/owid_mem.py` trains question-only on the evaluated custom backbone: version 2, Gemma-4-31B layer 41
with bidirectional prompt attention. Its external loader and checkpoint are not included. Without an abstention target, the decoder confidently maps an unseen
series to a trained one. A relation named “an indicator this store does not hold”, with no edges,
provides that target. Mass assigned to it retrieves nothing; the abstain score is `1 − retrieved mass`.

`experiments/statistics/owid_store.py negatives` samples other OWID charts from the sitemap. It excludes charts whose titles
overlap a stored series, use a word unique to a held-out series, match a public `global_facts` topic,
or have numbers equal to a stored series. A sixth set, `unknown_series`, evaluates held-out charts
against their own data. After 4,000 steps, the 23-series attribution-only store (`--licence clean`) gives:

| held-out set | exact value | 4-option accuracy | decoded "not held" | abstain AUC vs held-out keys |
|---|---|---|---|---|
| fresh wording | 100.0 | 100.0 | 0.0% | |
| held-out keys (in the graph, never asked) | 100.0 | 100.0 | 0.0% | |
| held-out series (in the graph, never trained) | 0.1 | 20.9 | 98.3% | 1.000 |
| missing keys (no value in the graph) | | | 0.1% | 1.000 |
| unknown series (not in the graph) | 0.2 | 21.7 | 96.4% | 0.999 |

The next comparison pairs reads with the custom backbone's own answers, version 2 alone in the DI multiple-choice
format. It uses the store below abstain score 0.18, fitted on dev and half of `unknown_series`, and the
engine otherwise.

The headline results are in the [main report](../statistics-store.md).

On 6,471 numeric-option questions outside statistics lookup (GSM8K, SVAMP, arithmetic and STEM sets),
the store never overrides the engine; the lowest abstain score is 0.32. On an H100, `experiments/statistics/owid_hook.py` costs
75 ms per question with the block and graph attached to the engine's model and its forward unchanged.
`experiments/statistics/hook_latency.py` captures the layers below the block in one CUDA graph per question length, reducing
the read to 32 ms. Its captured states were bit-identical to eager reads on 300 of 300 questions.

Option matching parses the whole option text using the engine's rules for exact fractions and
well-formed thousands commas, extended to a currency sign, scale word or %. Unsupported text and
bracketed numbers cannot match. Comparisons require the same sign; zero matches only zero.
`owid_mem.choose` sums similarity weighted by the retrieved values and abstains if the best score
is nonpositive or tied. An invalid option does not prevent another valid option from winning.
The selected option is invariant to option order.

## Similar indicators in the full store

The full 46-series store includes closely related indicators absent from the attribution-only subset.
The following runs use three negative sets, 4,000 steps each, paired with the engine at the same 0.18 cutoff.

| negatives for the "not held" relation | held-out keys | held-out series (engine 70.0) | held-out series read as "not held" |
|---|---|---|---|
| far from every store series (title overlap < 0.2) | 99.9 | 65.3 | 68.9% |
| near ones too, unless their numbers equal a store series' | 98.7 | 68.8 | 76.4% |
| the same minus those whose numbers correlate > 0.9 with a store series | 97.7 | 68.8 | 56.2% |

Errors include fertility read as life expectancy, IPV1 as Pol3 and Hib3 as rubella. Negative charts
closer to the stored series teach more refusals but cost about a point on held-out keys. Negatives
whose numbers track a stored series can instead teach the model to refuse that stored series.
The attribution-only store has no such siblings and refuses 98% of its held-out series.

Fitting the threshold on dev and unknown series alone does not fix this failure. Neither set asks about
an indicator present in the store but absent from training. The fit chooses 0.99 and then answers 62%
of held-out-series questions incorrectly.
