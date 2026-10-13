"""Virginia single-parent families from American Community Survey microdata, 2020-2024.

Pulls every person record in a family household headed by someone with no spouse present, for
all sixty Virginia public use microdata areas, and keeps the income components separately so
market and non-market resources can be told apart.
"""
import json, os, pathlib, time, urllib.parse, urllib.request

import pandas as pd
from dotenv import load_dotenv

HERE = pathlib.Path(__file__).parent
OUT = HERE / "pums_single_parent_2024.csv"
BASE = "https://api.census.gov/data/2024/acs/acs5/pums"
XWALK = ("/private/tmp/claude-501/-Users-prakhar7m-Desktop-Datathon/"
         "e1497b8b-b244-4d81-9cad-2b12b977c6d2/scratchpad/xwalk/tract_to_puma.txt")

load_dotenv("/Users/prakhar7m/Desktop/Datathon/.env")
KEY = (os.environ.get("CENSUS_API_KEY") or "").strip()
assert KEY, "CENSUS_API_KEY missing"

HOUSEHOLD = ["PUMA", "HHT", "PAOC", "NOC", "NPF", "NP", "FINCP", "HINCP", "ADJINC",
             "WGTP", "TYPEHUGQ", "BDSP", "TEN", "GRNTP", "FS"]
PERSON = ["SPORDER", "AGEP", "SEX", "RELSHIPP", "PWGTP", "ESR", "WKHP", "HINS4"]
INCOME = ["WAGP", "SEMP", "INTP", "RETP", "OIP", "PAP", "SSIP", "SSP"]
GET = ["SERIALNO"] + PERSON + INCOME + [v for v in HOUSEHOLD if v != "PUMA"]

pumas = sorted(pd.read_csv(XWALK, dtype=str).query("STATEFP == '51'")["PUMA5CE"].unique())
print(f"{len(pumas)} Virginia public use microdata areas", flush=True)

frames = []
for i, puma in enumerate(pumas, 1):
    for hht in ("2", "3"):          # family household, male / female householder, no spouse
        params = {"get": ",".join(GET), "for": f"public use microdata area:{puma}",
                  "in": "state:51", "HHT": hht, "key": KEY}
        url = f"{BASE}?{urllib.parse.urlencode(params)}"
        for attempt in range(4):
            try:
                payload = json.load(urllib.request.urlopen(url, timeout=300))
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"PUMA {puma} HHT {hht}: {exc}")
                time.sleep(3 * (attempt + 1))
        frame = pd.DataFrame(payload[1:], columns=payload[0])
        # The interface echoes each filtered variable back as an extra column of the same name,
        # so duplicates are dropped before anything is added.
        frame = frame.loc[:, ~frame.columns.duplicated()]
        frame["PUMA"] = puma
        frame["HHT"] = hht
        frames.append(frame)
    if i % 10 == 0:
        print(f"  {i}/{len(pumas)} areas", flush=True)
    time.sleep(0.2)

people = pd.concat(frames, ignore_index=True)
people = people.loc[:, ~people.columns.duplicated()]
NUMERIC = [c for c in people.columns if c not in ("SERIALNO", "PUMA", "state",
                                                  "public use microdata area")]
people[NUMERIC] = people[NUMERIC].apply(pd.to_numeric, errors="coerce")
people = people[people["TYPEHUGQ"] == 1]          # housing units only, no group quarters
people.to_csv(OUT, index=False)
print(f"DONE {len(people):,} person records, "
      f"{people['SERIALNO'].nunique():,} households -> {OUT}", flush=True)
