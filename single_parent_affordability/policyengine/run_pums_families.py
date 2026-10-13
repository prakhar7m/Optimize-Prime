"""Run a weighted sample of real Virginia single-parent families through PolicyEngine.

Each family keeps its own size, its children's actual ages and its own reported income, so the
cross-check tests the one-parent-one-toddler assumption used everywhere else.
"""
import json, pathlib, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
XW = ("/private/tmp/claude-501/-Users-prakhar7m-Desktop-Datathon/"
      "e1497b8b-b244-4d81-9cad-2b12b977c6d2/scratchpad/xwalk/puma_locality_weights.csv")
OUT_FAMILIES = HERE / "pums_families_2024.csv"
OUT_RESULTS = HERE / "pums_family_results_2024.json"
URL = "https://api.policyengine.org/us/calculate"
YEAR = "2024"
SAMPLE = 2000
OWN_CHILD = [25, 26, 27]        # biological, adopted, stepchild
PARTNER = [22, 24]              # unmarried partner, opposite or same sex

people = pd.read_csv(HERE / "pums_single_parent_2024.csv",
                     dtype={"SERIALNO": str, "PUMA": str})

children = people[people["RELSHIPP"].isin(OWN_CHILD) & (people["AGEP"] < 18)]
by_household = children.groupby("SERIALNO")
reference = people[people["RELSHIPP"] == 20].set_index("SERIALNO")
families = reference.loc[reference.index.isin(by_household.groups)].copy()

families["n_children"] = by_household.size()
families["child_ages"] = by_household["AGEP"].apply(lambda s: sorted(s.tolist()))
families["youngest"] = by_household["AGEP"].min()
families["has_partner"] = families.index.isin(
    set(people.loc[people["RELSHIPP"].isin(PARTNER), "SERIALNO"]))

# Income, inflated to 2024 dollars with the Census Bureau's own adjustment factor.
adjust = families["ADJINC"]
for column in ["WAGP", "SEMP", "INTP", "RETP", "OIP", "PAP", "SSIP", "SSP", "FINCP"]:
    families[column] = families[column].fillna(0) * adjust
families["market_income"] = families[["WAGP", "SEMP", "INTP", "RETP", "OIP"]].sum(axis=1)
families["reported_non_market"] = families[["PAP", "SSIP", "SSP"]].sum(axis=1)

# A microdata area can span several localities. Rather than always picking the largest, each
# family is assigned a locality drawn in proportion to that locality's share of the area's
# single-parent families, so the sample spreads across all 133 the way the population does.
MASTER = ("/Users/prakhar7m/Desktop/Datathon/single_parent_affordability/"
          "va_single_parent_complete_budget_acs2024_5yr.csv")
prices = pd.read_csv(MASTER, dtype={"geoid": str}).set_index("geoid")
PRICE = {"infant": "childcare_infant_annual", "toddler": "childcare_toddler_annual",
         "school": "childcare_school_age_annual"}


def childcare_for(geoid, age):
    """What this locality charges to care for a child of this age, in 2024 dollars.

    Nobody pays for the care of a child of thirteen or more, and the federal childcare
    credit stops there too, so those children carry no childcare cost.
    """
    if age >= 13:
        return 0.0
    band = "infant" if age <= 2 else ("toddler" if age <= 5 else "school")
    if geoid not in prices.index:
        return 0.0
    value = prices.loc[geoid, PRICE[band]]
    return 0.0 if pd.isna(value) else float(value)


weights = pd.read_csv(XW, dtype={"PUMA5CE": str, "geoid": str})
assign = np.random.default_rng(20261012)
lookup = {puma: (group["geoid"].to_numpy(), group["weight"].to_numpy())
          for puma, group in weights.groupby("PUMA5CE")}
families["geoid"] = [assign.choice(lookup[p][0], p=lookup[p][1] / lookup[p][1].sum())
                     for p in families["PUMA"]]
assert families["geoid"].notna().all(), "a microdata area did not map to a locality"

families = families.reset_index()
families.to_csv(OUT_FAMILIES, index=False)
print(f"{len(families):,} families, weighted {int(families['WGTP'].sum()):,}", flush=True)

rng = np.random.default_rng(20261012)
probability = families["WGTP"].clip(lower=0) / families["WGTP"].clip(lower=0).sum()
picked = rng.choice(len(families), size=min(SAMPLE, len(families)), replace=False,
                    p=probability)
sample = families.iloc[sorted(picked)].copy()
print(f"sampled {len(sample):,} families with probability proportional to weight", flush=True)


def household(row):
    ages = row["child_ages"] if isinstance(row["child_ages"], list) else eval(row["child_ages"])
    members = ["parent"] + [f"child{i}" for i in range(len(ages))]
    persons = {"parent": {"age": {YEAR: int(min(max(row["AGEP"], 18), 85))},
                          "employment_income": {YEAR: float(max(row["market_income"], 0))},
                          "medicaid": {YEAR: None}, "chip": {YEAR: None}}}
    marital = {"mu_parent": {"members": ["parent"]}}
    for i, age in enumerate(ages):
        persons[f"child{i}"] = {
            "age": {YEAR: int(age)}, "medicaid": {YEAR: None},
            "chip": {YEAR: None}, "wic": {YEAR: None},
            "pre_subsidy_childcare_expenses": {
                YEAR: childcare_for(row["geoid"], int(age))}}
        marital[f"mu_child{i}"] = {"members": [f"child{i}"]}
    return {"household": {
        "people": persons,
        "families": {"family": {"members": members}},
        "marital_units": marital,
        "tax_units": {"tax_unit": {"members": members, "eitc": {YEAR: None},
                                   "va_eitc": {YEAR: None}, "ctc": {YEAR: None},
                                   "cdcc": {YEAR: None}, "aca_ptc": {YEAR: None},
                                   "income_tax": {YEAR: None}}},
        "spm_units": {"spm": {"members": members, "snap": {YEAR: None},
                              "snap_max_allotment": {YEAR: None}, "va_tanf": {YEAR: None},
                              "free_school_meals": {YEAR: None}}},
        "households": {"hh": {"members": members, "state_name": {YEAR: "VA"},
                              "county_fips": {YEAR: int(row["geoid"])},
                              "household_benefits": {YEAR: None},
                              "household_tax": {YEAR: None},
                              "household_net_income": {YEAR: None}}}}}


def call(payload, tries=4):
    for attempt in range(tries):
        try:
            request = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(request, timeout=300)).get("result")
        except Exception as exc:
            if attempt == tries - 1:
                return {"__error__": f"{type(exc).__name__}: {exc}"}
            time.sleep(3 * (attempt + 1))


lock = threading.Lock()
rows, done = [], 0


def one(record):
    global done
    index, row = record
    result = call(household(row))
    base = {"SERIALNO": row["SERIALNO"], "geoid": row["geoid"], "PUMA": row["PUMA"],
            "WGTP": float(row["WGTP"]), "n_children": int(row["n_children"]),
            "market_income": float(row["market_income"]),
            "childcare_market": sum(
                childcare_for(row["geoid"], int(a))
                for a in (row["child_ages"] if isinstance(row["child_ages"], list)
                          else eval(row["child_ages"]))),
            "reported_non_market": float(row["reported_non_market"]),
            "HHT": int(row["HHT"]), "has_partner": bool(row["has_partner"])}
    if "__error__" in result:
        base["error"] = result["__error__"]
    else:
        unit = result["spm_units"]["spm"]
        tax_unit = result["tax_units"]["tax_unit"]
        household_unit = result["households"]["hh"]
        get = lambda block, key: (block.get(key) or {}).get(YEAR)
        base.update({
            "eitc": get(tax_unit, "eitc"), "va_eitc": get(tax_unit, "va_eitc"),
            "ctc": get(tax_unit, "ctc"), "cdcc": get(tax_unit, "cdcc"),
            "aca_ptc": get(tax_unit, "aca_ptc"),
            "snap": get(unit, "snap"), "snap_max_allotment": get(unit, "snap_max_allotment"),
            "va_tanf": get(unit, "va_tanf"), "school_meals": get(unit, "free_school_meals"),
            "medicaid": sum((result["people"][p].get("medicaid") or {}).get(YEAR, 0.0)
                            for p in result["people"]),
            # FAMIS covers children above the Medicaid limit and is reported separately.
            "chip": sum((result["people"][p].get("chip") or {}).get(YEAR, 0.0)
                        for p in result["people"]),
            "wic": sum((result["people"][p].get("wic") or {}).get(YEAR, 0.0)
                       for p in result["people"] if p != "parent"),
            "tax": get(household_unit, "household_tax"),
            "net_income": get(household_unit, "household_net_income"),
        })
    with lock:
        rows.append(base)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(sample)}", flush=True)


start = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(one, list(sample.iterrows())))
OUT_RESULTS.write_text(json.dumps(rows))
errors = sum(1 for r in rows if "error" in r)
print(f"DONE {len(rows)} families, {errors} errors, {time.time() - start:.0f}s "
      f"-> {OUT_RESULTS}", flush=True)
