"""Sweep earnings for three representative localities to expose the benefit cliff."""
import json, time, urllib.request, pathlib
import pandas as pd

HERE = pathlib.Path(__file__).parent
OUT = HERE / "sweep.json"
URL = "https://api.policyengine.org/us/calculate"
YEAR = "2024"

d = pd.read_csv("/Users/prakhar7m/Desktop/Datathon/single_parent_affordability/"
                "va_single_parent_complete_budget_acs2024_5yr.csv", dtype={"geoid": str})
PICK = ["Arlington County", "Richmond city", "Lee County"]
sel = d[d.locality.isin(PICK)][["geoid", "locality", "childcare_toddler_annual",
                                "median_income_single_mother"]]
print(sel.to_string(index=False), flush=True)

def household(income, childcare, fips):
    return {"household": {
      "people": {"parent": {"age": {YEAR: 32}, "employment_income": {YEAR: float(income)}},
                 "child": {"age": {YEAR: 3},
                           "pre_subsidy_childcare_expenses": {YEAR: float(childcare)}}},
      "families": {"family": {"members": ["parent", "child"]}},
      "marital_units": {"mu": {"members": ["parent"]}, "muc": {"members": ["child"]}},
      "tax_units": {"tax_unit": {"members": ["parent", "child"], "eitc": {YEAR: None},
                                 "va_eitc": {YEAR: None}, "ctc": {YEAR: None},
                                 "cdcc": {YEAR: None}, "income_tax": {YEAR: None}}},
      "spm_units": {"spm": {"members": ["parent", "child"], "snap": {YEAR: None},
                            "va_tanf": {YEAR: None}, "free_school_meals": {YEAR: None}}},
      "households": {"hh": {"members": ["parent", "child"], "state_name": {YEAR: "VA"},
                            "county_fips": {YEAR: int(fips)},
                            "household_benefits": {YEAR: None},
                            "household_tax": {YEAR: None},
                            "household_net_income": {YEAR: None}}}}}

def call(payload, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=240)).get("result")
        except Exception as exc:
            if attempt == tries - 1:
                return {"__error__": f"{type(exc).__name__}: {exc}"}
            time.sleep(4 * (attempt + 1))

rows = []
EARNINGS = list(range(0, 82000, 2000))
total = len(sel) * len(EARNINGS)
done = 0
for _, r in sel.iterrows():
    for earnings in EARNINGS:
        res = call(household(earnings, r.childcare_toddler_annual, r.geoid))
        done += 1
        if "__error__" in res:
            rows.append({"locality": r.locality, "earnings": earnings,
                         "error": res["__error__"]})
        else:
            g = lambda e, k: (res[e[0]][e[1]].get(k) or {}).get(YEAR)
            rows.append({
                "locality": r.locality, "earnings": earnings,
                "childcare_market": float(r.childcare_toddler_annual),
                "eitc": g(("tax_units", "tax_unit"), "eitc"),
                "va_eitc": g(("tax_units", "tax_unit"), "va_eitc"),
                "ctc": g(("tax_units", "tax_unit"), "ctc"),
                "cdcc": g(("tax_units", "tax_unit"), "cdcc"),
                "income_tax": g(("tax_units", "tax_unit"), "income_tax"),
                "snap": g(("spm_units", "spm"), "snap"),
                "va_tanf": g(("spm_units", "spm"), "va_tanf"),
                "school_meals": g(("spm_units", "spm"), "free_school_meals"),
                "benefits": g(("households", "hh"), "household_benefits"),
                "tax": g(("households", "hh"), "household_tax"),
                "net_income": g(("households", "hh"), "household_net_income"),
            })
        if done % 20 == 0:
            print(f"  {done}/{total}", flush=True)
        time.sleep(0.3)

OUT.write_text(json.dumps(rows, indent=1))
print("DONE", len(rows), "rows,", sum(1 for x in rows if "error" in x), "errors ->", OUT, flush=True)
