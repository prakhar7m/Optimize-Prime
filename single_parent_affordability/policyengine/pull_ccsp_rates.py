"""Pull Virginia's childcare subsidy reimbursement rate and copayment for every locality.

The benefit itself will not populate through the interface, but the two numbers that build it
do: the daily maximum reimbursable rate per child and the family copayment. Both are published
policy parameters, so reading them here keeps the simulation on Virginia's own schedule.
"""
import json, pathlib, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

HERE = pathlib.Path(__file__).parent
MASTER = ("/Users/prakhar7m/Desktop/Datathon/single_parent_affordability/"
          "va_single_parent_complete_budget_acs2024_5yr.csv")
OUT = HERE / "va_ccsp_rates_2024.csv"
URL = "https://api.policyengine.org/us/calculate"
YEAR, MONTH = "2024", "2024-01"
AGES = {"infant": 1, "toddler": 3, "school_age": 8}

localities = pd.read_csv(MASTER, dtype={"geoid": str})[["geoid", "locality"]]
print(f"{len(localities)} localities x {len(AGES)} care ages", flush=True)


def payload(fips, age, income):
    return {"household": {
        "people": {
            "parent": {"age": {YEAR: 32}, "employment_income": {YEAR: float(income)},
                       "is_ccdf_reason_for_care_eligible": {YEAR: True}},
            "child": {"age": {YEAR: age}, "is_ccdf_reason_for_care_eligible": {YEAR: True},
                      "va_ccsp_daily_mrr": {MONTH: None},
                      "va_ccsp_care_age_group": {MONTH: None}}},
        "families": {"family": {"members": ["parent", "child"]}},
        "marital_units": {"mu": {"members": ["parent"]}, "muc": {"members": ["child"]}},
        "tax_units": {"tax_unit": {"members": ["parent", "child"]}},
        "spm_units": {"spm": {"members": ["parent", "child"],
                              "va_ccsp_activity_eligible": {MONTH: True},
                              "va_ccsp_copay": {MONTH: None},
                              "va_ccsp_eligible": {MONTH: None},
                              "va_ccsp_income_eligible": {MONTH: None},
                              "va_ccsp_locality_group": {MONTH: None}}},
        "households": {"hh": {"members": ["parent", "child"],
                              "state_name": {YEAR: "VA"}, "county_fips": {YEAR: int(fips)},
                              "va_ccsp_ready_region": {YEAR: None}}}}}


def call(body, tries=4):
    for attempt in range(tries):
        try:
            request = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(request, timeout=300))["result"]
        except Exception as exc:
            if attempt == tries - 1:
                return {"__error__": f"{type(exc).__name__}: {exc}"}
            time.sleep(3 * (attempt + 1))


lock, rows, done = threading.Lock(), [], 0


def one(record):
    global done
    _, row = record
    out = {"geoid": row["geoid"], "locality": row["locality"]}
    for band, age in AGES.items():
        result = call(payload(row["geoid"], age, 30000))
        if "__error__" in result:
            out[f"error_{band}"] = result["__error__"]
            continue
        spm, child = result["spm_units"]["spm"], result["people"]["child"]
        get = lambda block, key, period: (block.get(key) or {}).get(period)
        out[f"daily_mrr_{band}"] = get(child, "va_ccsp_daily_mrr", MONTH)
        if band == "toddler":
            out["ccsp_group_modelled"] = get(spm, "va_ccsp_locality_group", MONTH)
            out["ready_region"] = get(result["households"]["hh"], "va_ccsp_ready_region", YEAR)
            out["copay_monthly_at_30k"] = get(spm, "va_ccsp_copay", MONTH)
    with lock:
        rows.append(out)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(localities)}", flush=True)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(one, localities.iterrows()))

frame = pd.DataFrame(rows).sort_values("geoid")
frame.to_csv(OUT, index=False)
errors = sum(1 for column in frame.columns if column.startswith("error_")
             for value in frame[column].dropna())
print(f"DONE {len(frame)} localities, {errors} errors -> {OUT}", flush=True)
