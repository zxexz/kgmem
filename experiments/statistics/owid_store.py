"""Build a year-qualified statistics store and question sets from Our World in Data.

  python -m experiments.statistics.owid_store fetch --raw runs/statistics/owid/raw
  python -m experiments.statistics.owid_store build --raw runs/statistics/owid/raw --out runs/statistics/owid/store.json
  python -m experiments.statistics.owid_store questions --store runs/statistics/owid/store.json --out runs/statistics/owid/questions
  python -m experiments.statistics.owid_store decontam --questions runs/statistics/owid/questions --public liquamen_public_slices.jsonl.gz

FAMILIES selects whole series by subject family. Ingest all published places and
years from 1900 onward. Each graph path is place -> series node -> year-qualified
value. Store the float and name its value node to three significant figures.

Hold out a fifth of series from training questions while keeping them in the graph.
For trained series, hold out a tenth of (place, year) keys. Question generation
reserves the last template of each series and the last two generic templates."""
from kgmem.paths import ensure_parent

import argparse
import collections
import csv
import gzip
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
import zlib

UA = "kgmem-research/0.1 (year-qualified indicator store; low volume)"
YEAR0, YEAR1 = 1900, 2025

# Series metadata and question templates. ind is a noun phrase; {col} is a chart-column title.
# Templates use {c} for place and {y} for year; each list's last template is eval-only.
FAMILY_REASONS = {
    "development": "access and adoption shares of a population (internet, electricity, water, sanitation, cooking "
                   "fuel, phones, literacy): the most common kind of country-year development statistic",
    "health": "population health outcomes and coverage (mortality, life expectancy, prevalence, risk factors, "
              "vaccination, causes of death)",
    "conflict": "military spending, conflict deaths and weapons stockpiles",
    "economy": "income, poverty, prices, trade and labour",
    "demography": "population size, fertility and urbanisation",
    "governance": "survey- and expert-coded governance and attitude measures (corruption, democracy, women in "
                  "parliament, life satisfaction, trust)",
    "environment": "emissions, energy mix and land cover",
}

FAMILIES = [
    # development
    dict(slug="share-of-individuals-using-the-internet", family="development",
         ind="share of the population using the Internet",
         phr=["What share of people in {c} used the Internet in {y}?",
              "About what percentage of {c_pos} population were Internet users in {y}?",
              "In {y}, roughly what fraction of the population of {c} was online?"]),
    dict(slug="share-of-the-population-with-access-to-electricity", family="development",
         ind="share of the population with access to electricity",
         phr=["What percentage of people in {c} had access to electricity in {y}?",
              "In {y}, about how much of {c_pos} population had electricity?",
              "As of {y}, what proportion of the population of {c} could get electricity?"]),
    dict(slug="access-to-clean-fuels-and-technologies-for-cooking", family="development",
         ind="share of the population relying mainly on clean fuels for cooking",
         phr=["What share of {c_pos} population relied mainly on clean cooking fuels in {y}?",
              "In {y}, about what percentage of people in {c} cooked with clean fuels and technologies?",
              "As of {y}, how much of the population of {c} used clean fuels for cooking?"]),
    dict(slug="proportion-using-safely-managed-drinking-water", family="development",
         ind="share of the population using safely managed drinking water",
         phr=["What percentage of people in {c} used safely managed drinking water in {y}?",
              "As of {y}, what share of {c_pos} population had safely managed drinking water?",
              "In {y}, how much of the population of {c} had safe drinking water at home?"]),
    dict(slug="share-using-safely-managed-sanitation", family="development",
         ind="share of the population using safely managed sanitation",
         phr=["What percentage of people in {c} used safely managed sanitation in {y}?",
              "As of {y}, what share of {c_pos} population had safely managed sanitation?",
              "In {y}, how much of the population of {c} had a safely managed toilet?"]),
    dict(slug="mobile-cellular-subscriptions-per-100-people", family="development",
         ind="number of mobile phone subscriptions per 100 people",
         phr=["How many mobile phone subscriptions per 100 people did {c} have in {y}?",
              "In {y}, how many mobile cellular subscriptions were there for every 100 people in {c}?",
              "As of {y}, what was the number of mobile subscriptions per hundred inhabitants of {c}?"]),
    dict(slug="cross-country-literacy-rates", family="development", ind="adult literacy rate",
         phr=["What share of adults in {c} could read and write in {y}?",
              "What was the literacy rate in {c} in {y}?",
              "In {y}, what percentage of the adult population of {c} was literate?"]),
    # health
    dict(slug="child-mortality", family="health",
         ind="child mortality rate (deaths before age five per 100 live births)",
         phr=["What share of children born in {c} died before their fifth birthday in {y}?",
              "In {y}, how many children per 100 born in {c} died before age five?",
              "As of {y}, what was the under-five mortality rate in {c}, per 100 live births?"]),
    dict(slug="life-expectancy", family="health", ind="life expectancy at birth",
         phr=["What was life expectancy at birth in {c} in {y}?",
              "How long could a newborn in {c} expect to live, on average, in {y}?",
              "In {y}, what was the average life expectancy of people in {c}?"]),
    dict(slug="share-of-the-population-infected-with-hiv", family="health",
         ind="share of people aged 15-49 living with HIV",
         phr=["What percentage of adults aged 15 to 49 in {c} were living with HIV in {y}?",
              "In {y}, what was HIV prevalence among 15-49 year olds in {c}?",
              "As of {y}, what share of people aged 15-49 in {c} had HIV?"]),
    dict(slug="share-of-adults-defined-as-obese", family="health", ind="share of adults who are obese",
         cols=["obesity_among_adults__bmi__gt__30_kg_m2_"],
         phr=["What share of adults in {c} were obese in {y}?",
              "In {y}, what percentage of {c_pos} adults had a BMI of 30 or more?",
              "As of {y}, how common was adult obesity in {c}, as a share of adults?"]),
    dict(slug="share-of-adults-who-smoke", family="health", ind="share of adults who smoke or use tobacco",
         phr=["What percentage of adults in {c} used tobacco in {y}?",
              "In {y}, what share of {c_pos} adults smoked or used other tobacco?",
              "As of {y}, how many adults in {c}, as a percentage, were tobacco users?"]),
    dict(slug="global-vaccination-coverage", family="health", ind="{col_l}",
         phr=["In {y}, what was the {col_l} in {c}?",
              "As of {y}, what percentage of one-year-olds in {c} were covered: {col_l}?",
              "What did the {col_l} stand at in {c} in {y}?"]),
    dict(slug="share-of-deaths-by-cause", family="health", ind="share of deaths caused by {col_l}",
         phr=["What share of all deaths in {c} in {y} were caused by {col_l}?",
              "In {y}, what percentage of deaths in {c} were due to {col_l}?",
              "About what fraction of deaths in {c} did {col_l} account for in {y}?"]),
    # conflict
    dict(slug="military-spending-as-a-share-of-gdp-sipri", family="conflict",
         ind="military expenditure as a share of GDP", cols=["share_gdp"],
         phr=["What share of its GDP did {c} spend on the military in {y}?",
              "In {y}, how much of {c_pos} GDP went to military spending?",
              "As of {y}, what was military expenditure in {c} as a percentage of GDP?"]),
    dict(slug="deaths-in-armed-conflicts-by-type", family="conflict", ind="number of deaths in {col_l}",
         colmap={"one-sided violence": "one-sided violence against civilians", "non-state": "non-state conflicts",
                 "extrasystemic": "extrasystemic conflicts", "intrastate": "intrastate conflicts",
                 "interstate": "interstate conflicts"},
         phr=["How many people died in {col_l} in {c} in {y}?",
              "In {y}, what was the death toll from {col_l} in {c}?",
              "How many deaths were there in {col_l} in {c} in {y}?"]),
    dict(slug="nuclear-warhead-stockpiles", family="conflict", ind="number of nuclear warheads in its stockpile",
         phr=["How many nuclear warheads did {c} have in {y}?",
              "In {y}, how large was {c_pos} nuclear warhead stockpile?",
              "As of {y}, about how many nuclear weapons did {c} possess?"]),
    # economy
    dict(slug="gdp-per-capita-maddison-project-database", family="economy",
         ind="GDP per capita (Maddison Project, 2011 international dollars)", cols=["gdp_per_capita"],
         phr=["What was {c_pos} GDP per person in {y}, in 2011 international dollars?",
              "In {y}, what was average income per person in {c} according to the Maddison Project?",
              "As of {y}, what was GDP per capita in {c} (Maddison Project estimate)?"]),
    dict(slug="gdp-per-capita-worldbank", family="economy",
         ind="GDP per capita (World Bank, 2021 international dollars, PPP)", cols=["ny_gdp_pcap_pp_kd"],
         phr=["What was {c_pos} GDP per capita in {y}, adjusted for purchasing power?",
              "In {y}, what was GDP per person in {c} in 2021 international dollars?",
              "As of {y}, what did the World Bank put {c_pos} PPP GDP per capita at?"]),
    dict(slug="share-of-population-in-extreme-poverty", family="economy",
         ind="share of the population living on less than $3 a day", cols=["headcount_ratio__ppp_version_2021__pover"],
         phr=["What share of {c_pos} population lived in extreme poverty in {y}?",
              "In {y}, what percentage of people in {c} lived on less than $3 a day?",
              "As of {y}, how much of the population of {c} was below the international poverty line?"]),
    dict(slug="inflation-of-consumer-prices", family="economy", ind="consumer price inflation",
         phr=["What was the inflation rate in {c} in {y}?",
              "By how much did consumer prices rise in {c} in {y}?",
              "In {y}, what was annual consumer price inflation in {c}?"]),
    dict(slug="trade-as-share-of-gdp", family="economy", ind="trade (exports plus imports) as a share of GDP",
         phr=["What were {c_pos} exports plus imports as a share of GDP in {y}?",
              "In {y}, how large was trade relative to GDP in {c}?",
              "As of {y}, what was the trade-to-GDP ratio of {c}?"]),
    dict(slug="unemployment-rate", family="economy", ind="unemployment rate",
         phr=["What was the unemployment rate in {c} in {y}?",
              "In {y}, what share of {c_pos} labour force was unemployed?",
              "As of {y}, what percentage of workers in {c} were out of work?"]),
    dict(slug="share-of-employment-in-agriculture", family="economy", ind="share of employment in agriculture",
         phr=["What share of workers in {c} were employed in agriculture in {y}?",
              "In {y}, what percentage of {c_pos} jobs were in agriculture?",
              "As of {y}, how much of the workforce of {c} worked in farming?"]),
    # demography
    dict(slug="population", family="demography", ind="population",
         phr=["What was the population of {c} in {y}?",
              "How many people lived in {c} in {y}?",
              "In {y}, how many inhabitants did {c} have?"]),
    dict(slug="children-per-woman-un", family="demography", ind="fertility rate (births per woman)",
         phr=["How many children did the average woman in {c} have in {y}?",
              "What was {c_pos} fertility rate in {y}?",
              "In {y}, how many births per woman were there in {c}?"]),
    dict(slug="share-of-population-urban", family="demography", ind="share of the population living in urban areas",
         phr=["What percentage of {c_pos} population lived in cities in {y}?",
              "In {y}, what share of people in {c} lived in urban areas?",
              "As of {y}, how urbanised was {c}, as a share of its population?"]),
    # governance and attitudes
    dict(slug="ti-corruption-perception-index", family="governance", ind="Corruption Perceptions Index score",
         cols=["cpi_score"],
         phr=["What was {c_pos} score on Transparency International's Corruption Perceptions Index in {y}?",
              "In {y}, how did {c} score on the Corruption Perceptions Index (0-100)?",
              "As of {y}, what Corruption Perceptions Index score did {c} receive?"]),
    dict(slug="electoral-democracy-index", family="governance", ind="electoral democracy index (V-Dem, 0-1)",
         cols=["electdem_vdem__estimate_best"],
         phr=["What was {c_pos} score on V-Dem's electoral democracy index in {y}?",
              "In {y}, how democratic was {c} on V-Dem's electoral democracy index?",
              "As of {y}, what electoral democracy score did V-Dem give {c}?"]),
    dict(slug="share-of-women-in-parliament", family="governance", ind="share of women in parliament",
         cols=["wom_parl_vdem__estimate_best"],
         phr=["What share of seats in {c_pos} parliament did women hold in {y}?",
              "In {y}, what percentage of members of parliament in {c} were women?",
              "As of {y}, how much of the parliament of {c} was female?"]),
    dict(slug="happiness-cantril-ladder", family="governance",
         ind="average self-reported life satisfaction (0-10 Cantril ladder)",
         phr=["How did people in {c} rate their lives on a 0 to 10 scale in {y}, on average?",
              "In {y}, what was average life satisfaction in {c} on the Cantril ladder?",
              "As of {y}, what was the mean happiness score in {c} (0-10)?"]),
    dict(slug="self-reported-trust-attitudes", family="governance",
         ind="share of people who agree that most people can be trusted", cols=["trust"],
         phr=["What share of people in {c} said most people can be trusted in {y}?",
              "In {y}, what percentage of respondents in {c} agreed that most people can be trusted?",
              "As of {y}, how many people in {c}, as a share, trusted most others?"]),
    # environment
    dict(slug="co-emissions-per-capita", family="environment", ind="CO2 emissions per person (tonnes)",
         phr=["How many tonnes of CO2 per person did {c} emit in {y}?",
              "In {y}, what were {c_pos} carbon dioxide emissions per capita?",
              "As of {y}, how much CO2 did the average person in {c} emit, in tonnes?"]),
    dict(slug="share-electricity-renewables", family="environment",
         ind="share of electricity generated by renewables",
         phr=["What share of {c_pos} electricity came from renewables in {y}?",
              "In {y}, what percentage of electricity in {c} was generated from renewable sources?",
              "As of {y}, how much of the electricity of {c} was renewable?"]),
    dict(slug="forest-area-as-share-of-land-area", family="environment", ind="share of land covered by forest",
         cols=["forest_share"],
         phr=["What share of {c_pos} land was covered by forest in {y}?",
              "In {y}, what percentage of the land area of {c} was forest?",
              "As of {y}, how much of {c} was forested, as a share of its land?"]),
]

GENERIC = ["What was the {ind} in {c} in {y}?", "In {y}, what was the {ind} in {c}?",
           "As of {y}, what was {c_pos} {ind}?", "According to the data for {y}, what was the {ind} for {c}?",
           "What figure did the {ind} reach in {c} in {y}?", "For {c} in {y}, what was the {ind}?"]
GENERIC_EVAL = 2                       # Reserve the last two generic templates for evaluation.
# Exclude templates flagged by the decontamination gate: content-word Jaccard >= 0.5
# against a public global_facts item on at least 3 generated questions for that series.
# Keys: "phr<i>" indexes the series' own phrasings, "gen<i>" GENERIC.
BANNED = {"share-of-the-population-with-access-to-electricity": {"phr0", "phr1", "gen0", "gen1", "gen2"},
          "military-spending-as-a-share-of-gdp-sipri": {"gen0", "gen1", "gen2"},
          "forest-area-as-share-of-land-area": {"phr1"}, "share-of-adults-defined-as-obese": {"phr0"}}


def template_key(s, question):
    """Return the phr<i> or gen<i> template that generated a question, or None."""
    for i, t in enumerate(s["phr"] + GENERIC):
        pat = re.escape(t.replace("{ind}", s["ind"]))
        for slot in ("{c}", "{c_pos}"):
            pat = pat.replace(re.escape(slot), ".+?")
        if re.fullmatch(pat.replace(re.escape("{y}"), r"\d{4}"), question):
            return f"phr{i}" if i < len(s["phr"]) else f"gen{i - len(s['phr'])}"
    return None

REGIONS = ["World", "Africa", "Asia", "Europe", "North America", "South America", "Oceania",
           "High-income countries", "Upper-middle-income countries", "Lower-middle-income countries",
           "Low-income countries"]
THE = {"United States", "United Kingdom", "Netherlands", "Philippines", "Democratic Republic of Congo",
       "Central African Republic", "Gambia", "Bahamas", "United Arab Emirates", "Dominican Republic", "Maldives",
       "Marshall Islands", "Solomon Islands", "Comoros", "Seychelles", "Czech Republic", "World"}
ALIASES = {"United States": ["the United States", "the US", "the U.S.", "the USA", "America"],
           "United Kingdom": ["the United Kingdom", "the UK", "Britain"], "World": ["the world", "worldwide"],
           "Russia": ["the Russian Federation"], "Democratic Republic of Congo": ["the DRC", "DR Congo"],
           "Congo": ["the Republic of the Congo"], "South Korea": ["the Republic of Korea"],
           "Czechia": ["the Czech Republic"], "Netherlands": ["the Netherlands", "Holland"]}


def place(name):
    """Return a place name as used in questions and its possessive form."""
    p = "the world" if name == "World" else ("the " + name if name in THE else
                                             name.lower() if name.endswith("countries") else name)
    return p, p + ("'" if p.endswith("s") else "'s")


def fmt(v):
    """Format a graph value-node name to three significant figures."""
    if v == 0:
        return "0"
    a = abs(v)
    for div, word in ((1e12, " trillion"), (1e9, " billion"), (1e6, " million")):
        if a >= div:
            return f"{float(f'{v / div:.3g}'):g}{word}"
    s = f"{float(f'{v:.3g}'):.6f}".rstrip("0").rstrip(".")
    return f"{int(float(s)):,}" if a >= 1000 else s


def get(url, path):
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        with open(ensure_parent(path), "wb") as f:
            f.write(data)
        time.sleep(0.5)
    return path


def fetch(a):
    os.makedirs(a.raw, exist_ok=True)
    q = "?v=1&csvType=full&useColumnShortNames=true"
    blocked = {}
    for s in FAMILIES:
        base = f"https://ourworldindata.org/grapher/{s['slug']}"
        try:
            get(base + ".csv" + q, os.path.join(a.raw, s["slug"] + ".csv"))
        except urllib.error.HTTPError as e:            # OWID refuses data whose producer forbids redistribution
            blocked[s["slug"]] = f"HTTP {e.code} on the CSV (OWID: not redistributable)"
            print(s["slug"], blocked[s["slug"]], flush=True)
            continue
        meta = json.load(open(get(base + ".metadata.json" + q, os.path.join(a.raw, s["slug"] + ".meta.json"))))
        for col, c in meta["columns"].items():
            if c.get("owidVariableId"):
                get(f"https://api.ourworldindata.org/v1/indicators/{c['owidVariableId']}.metadata.json",
                    os.path.join(a.raw, f"var_{c['owidVariableId']}.json"))
        print(s["slug"], len(meta["columns"]), "columns", flush=True)
    json.dump(blocked, open(ensure_parent(os.path.join(a.raw, "blocked.json")), "w"), indent=1)


CLEAN = {"CC BY 4.0", "CC BY 3.0", "CC BY 3.0 IGO", "CC0 1.0 Universal", "Creative Commons BY 4.0",
         "Open Government Licence v3.0", "CC0", "Public domain", "Open Government Licence"}


def licence_class(lics):
    """Classify as clean only when all origins use attribution-only or public-domain terms.

    NC, ND, producer-specific terms and bare copyright notices count as restricted."""
    names = {l["licence"].split("#")[0].strip() for l in lics}
    return "clean" if names and names <= CLEAN else "restricted"


def licences(var):
    out = []
    for o in var.get("origins", []):
        lic = o.get("license") or {}
        if isinstance(lic, str):
            try:
                lic = json.loads(lic.replace("'", '"'))
            except ValueError:
                lic = {"name": lic}
        out.append(dict(producer=o.get("producer", ""), licence=lic.get("name", ""), url=lic.get("url", "")))
    return out


def build(a):
    series, values, heads = [], {}, {}
    blocked = json.load(open(os.path.join(a.raw, "blocked.json")))
    for s in FAMILIES:
        if s["slug"] in blocked:
            continue
        meta = json.load(open(os.path.join(a.raw, s["slug"] + ".meta.json")))
        rows = list(csv.DictReader(open(os.path.join(a.raw, s["slug"] + ".csv"))))
        for col, c in meta["columns"].items():
            if col == "owid_region" or col.endswith("annotations") or (s.get("cols") and not any(col.startswith(x) for x in s["cols"])) \
                    or c.get("type") not in (None, "Numeric", "Integer", "Float"):
                continue
            title = c.get("titleShort") or col
            col_l = title[0].lower() + title[1:] if not title[:2].isupper() else title
            col_l = s.get("colmap", {}).get(col_l, col_l)
            sid = f"{s['slug']}:{col}"
            var = json.load(open(os.path.join(a.raw, f"var_{c['owidVariableId']}.json"))) \
                if c.get("owidVariableId") else {}
            series.append(dict(id=sid, family=s["family"], slug=s["slug"], column=col, title=title,
                               ind=s["ind"].format(col_l=col_l), col_l=col_l, unit=c.get("unit", ""),
                               short_unit=c.get("shortUnit", ""), citation=c.get("citationShort", ""),
                               licences=licences(var), licence_class=licence_class(licences(var)), phr=[p.replace("{col_l}", col_l) for p in s["phr"]]))
            vals = collections.defaultdict(dict)
            for r in rows:
                v, y = r.get(col, ""), int(r["year"])
                code = r.get("code", "")
                if v in ("", None) or not YEAR0 <= y <= YEAR1:
                    continue
                if not (re.fullmatch(r"[A-Z]{3}", code) or code == "OWID_WRL" or code.startswith("OWID_K")
                        or r["entity"] in REGIONS):
                    continue
                vals[r["entity"]][y] = float(v)
                heads.setdefault(r["entity"], code)
            values[sid] = vals
    rng = random.Random(a.seed)
    by_fam = collections.defaultdict(list)
    for s in series:
        by_fam[s["family"]].append(s)
    for fam, ss in sorted(by_fam.items()):                 # Hold out a fifth of each family's series from training questions.
        for s in rng.sample(ss, max(1, round(len(ss) * a.held_frac)) if len(ss) > 1 else 0):
            s["heldout"] = True
    for s in series:
        s.setdefault("heldout", False)
    out = dict(series=series, heads=[dict(name=n, code=c, place=place(n)[0], aliases=ALIASES.get(n, []))
                                     for n, c in sorted(heads.items())],
               values={sid: {h: {str(y): v for y, v in sorted(ys.items())} for h, ys in vs.items()}
                       for sid, vs in values.items()},
               years=[YEAR0, YEAR1], reasons=FAMILY_REASONS, key_holdout=a.key_frac, blocked=blocked)
    json.dump(out, open(ensure_parent(a.out), "w"))
    n = sum(len(ys) for vs in values.values() for ys in vs.values())
    print(f"{len(series)} series ({sum(s['heldout'] for s in series)} held out), {len(heads)} places, {n} values; "
          f"{len({fmt(v) for vs in values.values() for ys in vs.values() for v in ys.values()})} distinct value names")


def key_held(sid, h, y, frac):
    return zlib.crc32(f"{sid}|{h}|{y}".encode()) % 1000 < frac * 1000


def options(rng, gold, s, vals):
    """Return four options and the gold index.

    Prefer different values from the same series; fill remaining distractors with
    multiplicative perturbations."""
    pool = sorted({v for ys in vals.values() for v in ys.values()})
    far = lambda x, ys: all(abs(x - y) > 0.12 * max(abs(x), abs(y), 1e-9) for y in ys)
    opts = [gold]
    cand = [v for v in rng.sample(pool, min(60, len(pool)))] + [gold * f for f in (0.4, 0.6, 0.75, 1.35, 1.7, 2.5)]
    rng.shuffle(cand)
    for c in cand:
        if len(opts) == 4:
            break
        if far(c, opts) and (gold <= 0 or c > 0):
            opts.append(c)
    while len(opts) < 4:
        opts.append(opts[-1] * 1.5 + 1)
    order = list(range(4))
    rng.shuffle(order)
    return [fmt(opts[i]) for i in order], order.index(0)


def questions(a):
    st = json.load(open(a.store))
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    heads = {h["name"]: h for h in st["heads"]}
    frac = st["key_holdout"]

    def ask(s, h, y, tset):
        c, c_pos = place(h)
        n, g = len(s["phr"]), len(GENERIC)
        keys = [f"phr{i}" for i in range(n - 1)] + [f"gen{i}" for i in range(g - GENERIC_EVAL)] if tset == "train" \
            else [f"phr{n - 1}"] + [f"gen{i}" for i in range(g - GENERIC_EVAL, g)]
        temps = [(s["phr"] + GENERIC)[int(k[3:]) + (0 if k[:3] == "phr" else n)] for k in keys
                 if k not in BANNED.get(s["slug"], ())]
        alias = rng.choice([None] + heads[h]["aliases"]) if heads[h]["aliases"] and h != "World" else None
        if alias:
            c, c_pos = alias, alias + ("'" if alias.endswith("s") else "'s")
        t = rng.choice(temps)
        return t.format(c=c, c_pos=c_pos, y=y, ind=s["ind"]).replace("  ", " "), c

    def item(s, h, y, tset, kind):
        v = st["values"][s["id"]][h][str(y)]
        q, mention = ask(s, h, y, tset)
        opts, gi = options(rng, v, s, st["values"][s["id"]])
        return dict(qid=f"{kind}-{zlib.crc32(f'{q}|{s['id']}|{h}|{y}'.encode()):08x}", kind=kind, question=q,
                    mention=mention, head=h, series=s["id"], year=y, value=v, answer=fmt(v), options=opts,
                    gold=gi, unit=s["unit"])

    trained = [s for s in st["series"] if not s["heldout"]]
    held = [s for s in st["series"] if s["heldout"]]
    keys = lambda ss, want: [(s, h, int(y)) for s in ss for h, ys in st["values"][s["id"]].items() for y in ys
                             if key_held(s["id"], h, y, frac) == want]
    train_keys, held_keys, series_keys = keys(trained, False), keys(trained, True), keys(held, False) + keys(held, True)
    sets = {"train": [item(*rng.choice(train_keys), "train", "train") for _ in range(a.n_train)],
            "dev": [item(*rng.choice(train_keys), "train", "dev") for _ in range(a.n_eval)],
            "fresh_wording": [item(*rng.choice(train_keys), "eval", "fresh") for _ in range(a.n_eval)],
            "held_keys": [item(*rng.choice(held_keys), "eval", "heldkey") for _ in range(a.n_eval)],
            "held_series": [item(*rng.choice(series_keys), "eval", "heldseries") for _ in range(a.n_eval)]}
    missing = []                                       # Missing-value questions for trained series.
    while len(missing) < a.n_eval:
        s = rng.choice(trained)
        h = rng.choice(list(st["values"][s["id"]]))
        ys = [int(y) for y in st["values"][s["id"]][h]]
        y = rng.randint(max(YEAR0, min(ys) - 10), min(YEAR1, max(ys) + 5))
        if str(y) in st["values"][s["id"]][h]:
            continue
        q, mention = ask(s, h, y, "eval")
        missing.append(dict(qid=f"missing-{zlib.crc32(f'{q}|{s['id']}|{h}|{y}'.encode()):08x}", kind="missing",
                            question=q, mention=mention, head=h, series=s["id"], year=y, value=None))
    sets["missing"] = missing
    for k, rows in sets.items():
        seen = collections.Counter()
        for r in rows:                                 # Keep repeated draws distinguishable by qid.
            seen[r["qid"]] += 1
            if seen[r["qid"]] > 1:
                r["qid"] += f"-{seen[r['qid']]}"
        with open(ensure_parent(os.path.join(a.out, f"{k}.jsonl")), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(k, len(rows))


def gate(a):
    """Apply stored gate verdicts and BANNED template exclusions.

    Write <set>.gated.jsonl as a subset of the previously passed clean items."""
    st = json.load(open(a.store))
    ser = {s["id"]: s for s in st["series"]}
    fail = {json.loads(l)["qid"] for l in open(a.gate_fail)}
    for fn in sorted(os.listdir(a.questions)):
        if not fn.endswith(".clean.jsonl"):
            continue
        rows = [json.loads(l) for l in open(os.path.join(a.questions, fn))]
        keep = [r for r in rows if r["qid"] not in fail and (r["series"] is None or
                template_key(ser[r["series"]], r["question"]) not in BANNED.get(ser[r["series"]]["slug"], ()))]
        with open(ensure_parent(os.path.join(a.questions, fn.replace(".clean.", ".gated."))), "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        print(fn, len(rows), "->", len(keep), f"({sum(r['qid'] in fail for r in rows)} failed the gate)")


def ngrams(text, n=13):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


STOP = set("a an the of in on for to and or by per at as with from who that is are was were their its than more "
           "less how what which share number rate total annual population people percent".split())


def content(text):
    """Tokenize content words and fold plural -s."""
    return {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOP and len(w) > 2}


def negatives(a):
    """Generate NULL-relation questions from seeded OWID chart titles.

    Exclude titles overlapping store indicators (Jaccard >= 0.2), words unique to
    held-out series, and public global_facts topics (at least two words and half
    the title's content words in one item). The 184 public items are a blocklist.
    Fetch candidate data to reject duplicate and correlated store-series variants.
    Reserve 20% of titles for unknown_series with values for native comparisons;
    the other 80% supply training negatives without values."""
    st = json.load(open(a.store))
    rng = random.Random(a.seed)
    os.makedirs(a.raw, exist_ok=True)
    slugs = sorted(set(re.findall(r"https://ourworldindata.org/grapher/([a-z0-9-]+)",
                                  open(os.path.join(a.raw, "sitemap.xml")).read())) - {s["slug"] for s in FAMILIES})
    mine = [content(s["title"] + " " + s["ind"]) for s in st["series"]]
    trained_w = set().union(*(content(s["title"] + " " + s["ind"]) for s in st["series"] if not s["heldout"]))
    held_w = set().union(*(content(s["title"] + " " + s["ind"]) for s in st["series"] if s["heldout"])) - trained_w
    pub = []
    with gzip.open(a.public, "rt") as f:
        for l in f:
            if "global_facts" in l:
                pub += [content(q["instructions"]) for q in json.loads(l)["row"]["questions"].values()]
    keep, why = [], collections.Counter()
    for slug in rng.sample(slugs, a.n_charts):
        try:
            meta = json.load(open(get(f"https://ourworldindata.org/grapher/{slug}.metadata.json"
                                      "?v=1&csvType=full&useColumnShortNames=true",
                                      os.path.join(a.raw, slug + ".meta.json"))))
        except (urllib.error.HTTPError, ValueError):
            why["no metadata"] += 1
            continue
        cols = [(k, c) for k, c in meta.get("columns", {}).items() if k != "owid_region"
                and not k.endswith("annotations") and c.get("type") in (None, "Numeric", "Integer", "Float")]
        if not cols:
            why["no numeric column"] += 1
            continue
        title = (meta.get("chart") or {}).get("title") or cols[0][1].get("titleShort") or ""
        title = re.sub(r",? \d{3,4}(?: to \d{3,4})?$", "", title).strip()
        cw = content(title)
        if len(cw) < 2:
            why["too generic"] += 1
        elif any(len(cw & m) / len(cw | m) >= a.title_jaccard for m in mine):
            why["overlaps a store series"] += 1
        elif cw & held_w:
            why["a held-out series' word"] += 1
        elif any(len(cw & p) >= max(2, len(cw) / 2) for p in pub):
            why["public global_facts topic"] += 1
        else:
            keep.append(dict(slug=slug, column=cols[0][0], title=title, unit=cols[0][1].get("unit", ""),
                             ind=title[0].lower() + title[1:] if not title[:2].isupper() else title))
    heads = st["heads"]
    names = {h["name"] for h in heads}
    for t in keep:                                         # Filter duplicate and correlated series after fetching data.
        try:
            rows = list(csv.DictReader(open(get(f"https://ourworldindata.org/grapher/{t['slug']}.csv"
                                                "?v=1&csvType=full&useColumnShortNames=true",
                                                os.path.join(a.raw, t["slug"] + ".csv")))))
        except urllib.error.HTTPError:
            rows = []
        vals = collections.defaultdict(dict)
        for r in rows:
            v, yr = r.get(t["column"], ""), r.get("year", "")         # Daily charts are outside this annual store.
            if v not in ("", None) and yr.lstrip("-").isdigit() and r["entity"] in names and YEAR0 <= int(yr) <= YEAR1:
                try:
                    vals[r["entity"]][int(yr)] = float(v)
                except ValueError:
                    pass
        t["vals"] = vals
    store_vals = {sid: {(h, int(y)): v for h, ys in vs.items() for y, v in ys.items()} for sid, vs in st["values"].items()}

    def duplicate(t):
        """Detect a store-series duplicate: >= 20 shared keys, at least half within 1%."""
        mine = {(h, y): v for h, ys in t["vals"].items() for y, v in ys.items()}
        for sv in store_vals.values():
            shared = [k for k in mine if k in sv]
            if len(shared) >= 20 and sum(abs(mine[k] - sv[k]) <= 0.01 * max(abs(mine[k]), abs(sv[k]), 1e-9)
                                         for k in shared) >= len(shared) / 2:
                return True
        return False
    dup = [t["slug"] for t in keep if duplicate(t)]
    why["same data as a store series"] = len(dup)
    keep = [t for t in keep if t["slug"] not in dup]

    def correlated(t):
        """Detect Pearson r > --corr_max on >= 20 shared keys.

        Exclude correlated variants so negatives do not train refusal of a stored series."""
        mine = {(h, y): v for h, ys in t["vals"].items() for y, v in ys.items()}
        for sv in store_vals.values():
            shared = [k for k in mine if k in sv]
            if len(shared) < 20:
                continue
            x, y = [mine[k] for k in shared], [sv[k] for k in shared]
            mx, my = sum(x) / len(x), sum(y) / len(y)
            sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
            sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
            if sxx > 0 and syy > 0 and sxy / (sxx * syy) ** 0.5 > a.corr_max:
                return True
        return False
    if a.corr_max is not None:
        cor = [t["slug"] for t in keep if correlated(t)]
        why[f"correlates > {a.corr_max} with a store series"] = len(cor)
        keep = [t for t in keep if t["slug"] not in cor]
    print(f"{len(keep)} negative charts kept of {a.n_charts}; dropped {dict(why)}")
    rng.shuffle(keep)
    cut = int(0.8 * len(keep))
    train_t, eval_t = keep[:cut], keep[cut:]

    def q(t, tset, y):
        h = rng.choice(heads)["name"]
        c, c_pos = place(h)
        temps = GENERIC[:-GENERIC_EVAL] if tset == "train" else GENERIC[-GENERIC_EVAL:]
        return rng.choice(temps).format(c=c, c_pos=c_pos, y=y, ind=t["ind"]), c, h
    out = []
    for _ in range(a.n_neg):
        t = rng.choice(train_t)
        text, mention, h = q(t, "train", rng.randint(1960, 2023))
        out.append(dict(qid=f'''neg-{zlib.crc32(f"{text}|{t['slug']}".encode()):08x}''', kind="neg", question=text,
                        mention=mention, head=h, series=None, year=int(re.findall(r"\d{4}", text)[-1]),
                        value=None, chart=t["slug"]))
    unknown = []
    eval_t = [t for t in eval_t if t.get("vals")]
    print(len(eval_t), "unknown-series charts with yearly data")
    while len(unknown) < a.n_eval and eval_t:
        t = rng.choice(eval_t)
        h = rng.choice(list(t["vals"]))
        y = rng.choice(list(t["vals"][h]))
        c, c_pos = place(h)
        text = rng.choice(GENERIC[-GENERIC_EVAL:]).format(c=c, c_pos=c_pos, y=y, ind=t["ind"])
        v = t["vals"][h][y]
        opts, gi = options(rng, v, None, {k: {str(yy): x for yy, x in ys.items()} for k, ys in t["vals"].items()})
        unknown.append(dict(qid=f'''unknown-{zlib.crc32(f"{text}|{t['slug']}".encode()):08x}''', kind="unknown",
                            question=text, mention=c, head=h, series=None, year=y, value=v, answer=fmt(v),
                            options=opts, gold=gi, unit=t["unit"], chart=t["slug"]))
    for name, rows in (("neg_train", out), ("unknown_series", unknown)):
        with open(ensure_parent(os.path.join(a.out, f"{name}{a.tag}.jsonl")), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(name, len(rows), "questions from", len({r["chart"] for r in rows}), "charts")
    json.dump(dict(train=[t["slug"] for t in train_t], eval=[t["slug"] for t in eval_t], dropped=dict(why),
                   duplicates=dup),
              open(ensure_parent(os.path.join(a.out, f"neg_charts{a.tag}.json")), "w"), indent=1)


def decontam(a):
    """Remove generated questions sharing a public global_facts 13-gram or option-number set.

    Public items are used only for this exclusion check."""
    pub_grams, pub_opts = set(), set()
    num = lambda s: tuple(sorted(re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))))
    n_pub = 0
    with gzip.open(a.public, "rt") as f:                 # DI public-slice fields: questions.q1.{instructions, criteria}.
        for l in f:
            if "global_facts" not in l:
                continue
            r = json.loads(l)["row"]                    # Track names identify MMLU and MMLU-Pro global_facts rows.
            n_pub += 1
            for q in r["questions"].values():
                pub_grams |= ngrams(q["instructions"])
                opts = q.get("criteria") or {}
                o = tuple(sorted(sum((list(num(str(x))) for x in opts.values()), [])))
                if o:
                    pub_opts.add(o)
    print(f"public global_facts: {n_pub} items, {len(pub_grams)} 13-grams, {len(pub_opts)} option sets")
    for fn in sorted(os.listdir(a.questions)):
        if not fn.endswith(".jsonl") or fn.endswith(".clean.jsonl"):
            continue
        rows = [json.loads(l) for l in open(os.path.join(a.questions, fn))]
        keep = [r for r in rows if not (ngrams(r["question"]) & pub_grams) and
                tuple(sorted(sum((list(num(o)) for o in r.get("options") or []), []))) not in pub_opts]
        with open(ensure_parent(os.path.join(a.questions, fn.replace(".jsonl", ".clean.jsonl"))), "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        print(fn, len(rows), "->", len(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "build", "questions", "decontam", "gate", "negatives"])
    ap.add_argument("--n_charts", type=int, default=400)
    ap.add_argument("--n_neg", type=int, default=50000)
    ap.add_argument("--title_jaccard", type=float, default=0.2,
                    help="negatives: drop charts whose title overlaps a store series this much (0.6: near-identical "
                         "only, so same-domain siblings with different data stay as hard negatives)")
    ap.add_argument("--tag", default="", help="suffix for the negatives' file names")
    ap.add_argument("--corr_max", type=float, default=None,
                    help="negatives: also drop charts whose numbers correlate above this with a store series")
    ap.add_argument("--gate_fail", default="")
    ap.add_argument("--raw", default="runs/statistics/owid/raw")
    ap.add_argument("--out", default=None,
                    help="output file for build (default: --store), or directory for questions/negatives (default: --questions)")
    ap.add_argument("--store", default="runs/statistics/owid/store.json")
    ap.add_argument("--questions", default="runs/statistics/owid/questions")
    ap.add_argument("--public", default="")
    ap.add_argument("--held_frac", type=float, default=0.2)
    ap.add_argument("--key_frac", type=float, default=0.1)
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.out is None:
        a.out = a.store if a.cmd == "build" else a.questions
    dict(fetch=fetch, build=build, questions=questions, decontam=decontam, gate=gate, negatives=negatives)[a.cmd](a)


if __name__ == "__main__":
    main()
