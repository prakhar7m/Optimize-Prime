"""Market and non-market resources at every earnings level, for every Virginia locality.

Produces the input to a New York City style True Cost of Living calculation: for each locality,
the full resource curve against earnings, so the break-even earnings level can be solved for and
read off the published income distribution.
"""
import json, pathlib, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = pathlib.Path(__file__).parent
OUT = HERE / "resource_grid_2024.json"
URL = "https://api.policyengine.org/us/calculate"
YEAR = "2024"
EARNINGS = list(range(0, 82500, 2500))

d = pd.read_csv("/Users/prakhar7m/Desktop/Datathon/single_parent_affordability/"
                "va_single_parent_complete_budget_acs2024_5yr.csv", dtype={"geoid": str})
d = d.dropna(subset=["childcare_toddler_annual"])
print(f"{len(d)} localities x {len(EARNINGS)} earnings levels = "
      f"{len(d) * len(EARNINGS)} calls", flush=True)


def household(income, childcare, fips):
    return {"household": {
      "people": {
        "parent": {"age": {YEAR: 32}, "employment_income": {YEAR: float(income)},
                   "medicaid": {YEAR: None}, "chip": {YEAR: None}},
        "child": {"age": {YEAR: 3}, "pre_subsidy_childcare_expenses": {YEAR: float(childcare)},
                  "medicaid": {YEAR: None}, "chip": {YEAR: None},
                  "wic": {YEAR: None}}},
      "families": {"family": {"members": ["parent", "child"]}},
      "marital_units": {"mu": {"members": ["parent"]}, "muc": {"members": ["child"]}},
      "tax_units": {"tax_unit": {"members": ["parent", "child"], "eitc": {YEAR: None},
                                 "va_eitc": {YEAR: None}, "ctc": {YEAR: None},
                                 "cdcc": {YEAR: None}, "aca_ptc": {YEAR: None},
                                 "income_tax": {YEAR: None}}},
      "spm_units": {"spm": {"members": ["parent", "child"], "snap": {YEAR: None},
                            "va_tanf": {YEAR: None},
                            "free_school_meals": {YEAR: None}}},
      "households": {"hh": {"members": ["parent", "child"], "state_name": {YEAR: "VA"},
                            "county_fips": {YEAR: int(fips)},
                            "household_benefits": {YEAR: None},
                            "household_tax": {YEAR: None},
                            "household_net_income": {YEAR: None}}}}}


def call(payload, tries=4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=300)).get("result")
        except Exception as exc:
            if attempt == tries - 1:
                return {"__error__": f"{type(exc).__name__}: {exc}"}
            time.sleep(3 * (attempt + 1))


lock = threading.Lock()
rows, done = [], 0
total = len(d) * len(EARNINGS)


def one(task):
    global done
    geoid, locality, childcare, earnings = task
    res = call(household(earnings, childcare, geoid))
    if "__error__" in res:
        row = {"geoid": geoid, "locality": locality, "earnings": earnings,
               "error": res["__error__"]}
    else:
        def g(path, key):
            return (res[path[0]][path[1]].get(key) or {}).get(YEAR)
        row = {
            "geoid": geoid, "locality": locality, "earnings": earnings,
            "childcare_market": float(childcare),
            "eitc": g(("tax_units", "tax_unit"), "eitc"),
            "va_eitc": g(("tax_units", "tax_unit"), "va_eitc"),
            "ctc": g(("tax_units", "tax_unit"), "ctc"),
            "cdcc": g(("tax_units", "tax_unit"), "cdcc"),
            "aca_ptc": g(("tax_units", "tax_unit"), "aca_ptc"),
            "income_tax": g(("tax_units", "tax_unit"), "income_tax"),
            "snap": g(("spm_units", "spm"), "snap"),
            "va_tanf": g(("spm_units", "spm"), "va_tanf"),
            "school_meals": g(("spm_units", "spm"), "free_school_meals"),
            "wic": (res["people"]["child"].get("wic") or {}).get(YEAR),
            "medicaid_parent": (res["people"]["parent"].get("medicaid") or {}).get(YEAR),
            "medicaid_child": (res["people"]["child"].get("medicaid") or {}).get(YEAR),
            # Children lose Medicaid above 138 percent of the poverty guideline but keep coverage
            # through FAMIS, which the model reports separately. Without this the ledger records
            # nothing for a child who is in fact insured.
            "chip_parent": (res["people"]["parent"].get("chip") or {}).get(YEAR),
            "chip_child": (res["people"]["child"].get("chip") or {}).get(YEAR),
            "benefits": g(("households", "hh"), "household_benefits"),
            "tax": g(("households", "hh"), "household_tax"),
            "net_income": g(("households", "hh"), "household_net_income"),
        }
    with lock:
        rows.append(row)
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{total}", flush=True)
    return None


tasks = [(r.geoid, r.locality, r.childcare_toddler_annual, e)
         for r in d.itertuples() for e in EARNINGS]
start = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(one, tasks))

OUT.write_text(json.dumps(rows))
errors = sum(1 for x in rows if "error" in x)
print(f"DONE {len(rows)} rows, {errors} errors, {time.time() - start:.0f}s -> {OUT}", flush=True)
