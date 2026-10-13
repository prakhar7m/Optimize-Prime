"""Run every Virginia locality through PolicyEngine to get taxes and transfers."""
import json, time, urllib.request, urllib.error, sys, pathlib
import pandas as pd

OUT = pathlib.Path("/private/tmp/claude-501/-Users-prakhar7m-Desktop-Datathon/"
                   "e1497b8b-b244-4d81-9cad-2b12b977c6d2/scratchpad/pe/results.json")
URL = "https://api.policyengine.org/us/calculate"
YEAR = "2024"

d = pd.read_csv("/Users/prakhar7m/Desktop/Datathon/single_parent_affordability/"
                "va_ccsp_group_realignment_acs2024_5yr.csv", dtype={"geoid": str})

def household(income, childcare, child_age, fips):
    return {"household": {
      "people": {
        "parent": {"age": {YEAR: 32}, "employment_income": {YEAR: float(income)}},
        "child":  {"age": {YEAR: child_age},
                   "pre_subsidy_childcare_expenses": {YEAR: float(childcare)}}},
      "families":      {"family": {"members": ["parent", "child"]}},
      "marital_units": {"mu": {"members": ["parent"]}, "muc": {"members": ["child"]}},
      "tax_units":     {"tax_unit": {"members": ["parent", "child"],
                          "eitc": {YEAR: None}, "va_eitc": {YEAR: None},
                          "ctc": {YEAR: None}, "cdcc": {YEAR: None},
                          "income_tax": {YEAR: None}}},
      "spm_units":     {"spm": {"members": ["parent", "child"],
                          "snap": {YEAR: None}, "va_tanf": {YEAR: None},
                          "free_school_meals": {YEAR: None}}},
      "households":    {"hh": {"members": ["parent", "child"],
                          "state_name": {YEAR: "VA"}, "county_fips": {YEAR: int(fips)},
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

rows, done = [], 0
sub = d.dropna(subset=["median_income_single_mother", "childcare_toddler_annual"])
print(f"localities to run: {len(sub)}", flush=True)
for _, r in sub.iterrows():
    res = call(household(r.median_income_single_mother, r.childcare_toddler_annual, 3, r.geoid))
    if "__error__" in res:
        rows.append({"geoid": r.geoid, "locality": r.locality, "error": res["__error__"]})
    else:
        g = lambda e, k: (res[e[0]][e[1]].get(k) or {}).get(YEAR)
        rows.append({
            "geoid": r.geoid, "locality": r.locality,
            "income": r.median_income_single_mother,
            "childcare_market": r.childcare_toddler_annual,
            "eitc":       g(("tax_units","tax_unit"), "eitc"),
            "va_eitc":    g(("tax_units","tax_unit"), "va_eitc"),
            "ctc":        g(("tax_units","tax_unit"), "ctc"),
            "cdcc":       g(("tax_units","tax_unit"), "cdcc"),
            "income_tax": g(("tax_units","tax_unit"), "income_tax"),
            "snap":       g(("spm_units","spm"), "snap"),
            "va_tanf":    g(("spm_units","spm"), "va_tanf"),
            "school_meals": g(("spm_units","spm"), "free_school_meals"),
            "benefits":   g(("households","hh"), "household_benefits"),
            "tax":        g(("households","hh"), "household_tax"),
            "net_income": g(("households","hh"), "household_net_income"),
        })
    done += 1
    if done % 10 == 0:
        print(f"  {done}/{len(sub)}", flush=True)
    time.sleep(0.4)

OUT.write_text(json.dumps(rows, indent=1))
errs = sum(1 for x in rows if "error" in x)
print(f"DONE. {len(rows)} localities, {errs} errors -> {OUT}", flush=True)
