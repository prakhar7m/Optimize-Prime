"""Longitudinal cohort simulation for Virginia single-parent families.

The analysis notebooks answer "how many families fall short this year". This module answers the
question a policymaker actually has to fund: "what happens to these families over the eighteen
years a child is in the household, and which intervention at which age changes the arithmetic".

It takes the two thousand real families drawn from the household microdata, ages each one year by
year, reprices its costs as its children move through the care bands, and runs the result under
competing policy rules. Nothing here is a representative family; every row is a household that
answered the survey.

Structures
----------
cohort  one row per family, the starting state
panel   one row per family-year, the simulated trajectory
summary one row per policy scenario, the comparison a decision-maker reads

Every empirical constant carries its source in the comment beside it. Where a figure could not be
derived from our own data it is a named parameter with a stated default, never a silent guess.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
OUTDIR = ROOT / "single_parent_affordability"
PEDIR = OUTDIR / "policyengine"

HORIZON = 18            # a child is in the household from birth to eighteen
OTHER_NECESSITIES = 0.10        # United For ALICE allowance for everything a budget misses
COPAY_CEILING = 0.05            # Virginia caps the childcare copayment at 5% of income, from
                                # 1 July 2025 (8VAC20-790-40)
CARE_DAYS_PER_YEAR = 255        # weekdays less public holidays, the subsidy attendance basis

# Federal poverty guideline, 2024, contiguous states: base plus an increment per extra person.
FPG_BASE, FPG_STEP = 15060.0, 5380.0

# The three locality tiers Virginia assigns administratively, as a share of the poverty guideline.
# The realignment table labels them "I (150% FPG)" and so on, so the tier is read off the leading
# numeral rather than matched on the whole string.
GROUP_LIMIT = {"I": 1.50, "II": 1.60, "III": 1.85}


def tier_of(label) -> str | None:
    """The tier numeral from a label such as 'II (160% FPG)'."""
    if not isinstance(label, str):
        return None
    head = label.split(" ")[0].strip()
    return head if head in GROUP_LIMIT else None

# Care price bands. The age cuts follow the US Department of Labor childcare price database, whose
# school-age price covers before school, after school and the summer — the "three o'clock gap".
CARE_BAND = [(0, 2, "childcare_infant_annual"),
             (3, 5, "childcare_toddler_annual"),
             (6, 12, "childcare_school_age_annual")]
CARE_FREE_FROM = 13     # no paid care is modelled from a child's thirteenth birthday


def federal_poverty_guideline(people: int | np.ndarray) -> np.ndarray:
    """The 2024 guideline for a household of this size."""
    return FPG_BASE + FPG_STEP * (np.asarray(people) - 1)


@dataclass
class Policy:
    """One policy world to simulate.

    Attributes
    ----------
    subsidy_rule
        ``current``  tiers as Virginia assigns them today, sorted by the price of care.
        ``floored``  the revenue-positive re-sort: a locality takes the better of its current tier
                     and the tier its affordability burden earns it, so nobody is demoted.
        ``none``     no childcare subsidy at all, the counterfactual.
    graduated_taper
        Replace the hard income cutoff with a phase-out, so a raise can never leave a family worse
        off. Without it the cliff measured in chapter 6 applies.
    out_of_school_time
        Carry the subsidy past a child's sixth birthday to cover before school, after school and
        the summer, instead of ending it when the child starts school.
    child_support_modification
        Automatically recalculate orders written before the July 2025 guidelines and file the
        modification, instead of requiring a parent to return to court.
    credential_pipeline
        Enrol the parent in a tuition-free community college credential and step earnings up on
        completion.
    """

    name: str
    subsidy_rule: str = "current"
    graduated_taper: bool = False
    out_of_school_time: bool = False
    child_support_modification: bool = False
    credential_pipeline: bool = False
    notes: str = ""


@dataclass
class Assumptions:
    """Every number the simulation needs that is not read from a data file.

    Each one is derived from our own microdata where that is possible, and flagged where it is not.
    """

    # Real earnings growth. Derived from the age-earnings profile of Virginia single parents in the
    # 2020-2024 microdata: median market income rises from $35,262 at ages 25-29 to $65,032 at
    # 45-49, which is 3.1% a year. A cross-sectional profile mixes cohort effects into individual
    # growth, so it is an upper bound; the default halves it and the sensitivity switch restores it.
    wage_growth_observed: float = 0.031
    wage_growth: float = 0.0155

    # Child support. 14.6% of these families report other income, the field that contains child
    # support and alimony, at a weighted median of $6,703 a year. That share is an upper bound on
    # child support receipt because the field holds more than child support.
    child_support_receipt_rate: float = 0.146
    child_support_median: float = 6703.0

    # Share of existing orders written before the July 2025 guidelines took effect. Orders do not
    # update automatically, so these sit on the 2014 schedule until a parent returns to court.
    # NOT DERIVED FROM OUR DATA — a named parameter awaiting the court caseload file.
    pre_2025_order_share: float = 0.80

    # Uplift from recalculating an order under the raised combined income cap. SB 805 (Chapter 702,
    # 2025) lifted the cap from $35,000 to $42,500 of combined monthly income.
    # NOT DERIVED FROM OUR DATA — awaiting the Virginia child support guideline schedule.
    order_modification_uplift: float = 0.15

    # Community college credential. Earnings step applied on completion, after two years enrolled.
    # NOT DERIVED FROM OUR DATA — awaiting the wage-record outcomes file.
    credential_enrolment_rate: float = 0.10
    credential_completion_rate: float = 0.55
    credential_earnings_step: float = 0.18
    credential_years: int = 2

    sensitivity: dict = field(default_factory=dict)

    def undocumented(self) -> list[str]:
        """The parameters a reader should challenge first, because we could not derive them."""
        return ["pre_2025_order_share", "order_modification_uplift",
                "credential_enrolment_rate", "credential_completion_rate",
                "credential_earnings_step"]


# --------------------------------------------------------------------------------------------
# Cohort
# --------------------------------------------------------------------------------------------

def load_cohort() -> pd.DataFrame:
    """The starting state: two thousand real families with their own children and incomes."""
    families = pd.read_csv(PEDIR / "pums_families_2024.csv",
                           dtype={"SERIALNO": str, "PUMA": str, "geoid": str})
    costs = pd.read_csv(OUTDIR / "va_single_parent_complete_budget_acs2024_5yr.csv",
                        dtype={"geoid": str})
    rents = pd.read_csv(OUTDIR / "va_rent_by_bedrooms_acs2024_5yr.csv", dtype={"geoid": str})
    groups = pd.read_csv(OUTDIR / "va_ccsp_group_realignment_acs2024_5yr.csv",
                         dtype={"geoid": str})

    keep = ["geoid", "locality", "childcare_infant_annual", "childcare_toddler_annual",
            "childcare_school_age_annual", "transport_cost_annual", "food_annual",
            "health_care_annual"]
    cohort = (families
              .merge(costs[keep], on="geoid", how="left")
              .merge(rents[["geoid", "rent_1br", "rent_2br", "rent_3br", "rent_4br", "rent_5br"]],
                     on="geoid", how="left")
              .merge(groups[["geoid", "ccsp_group", "proposed_group", "floored_group"]],
                     on="geoid", how="left"))

    cohort["child_ages_start"] = cohort["child_ages"].apply(
        lambda value: value if isinstance(value, list) else eval(value))
    # This is the full microdata cohort, not the sample drawn for the PolicyEngine cross-check, so
    # each family carries its own survey weight. (The drawn sample is self-weighting because it was
    # taken with probability proportional to that weight; this one is not.)
    cohort["weight"] = cohort["WGTP"].astype(float)
    return cohort


# --------------------------------------------------------------------------------------------
# Cost side
# --------------------------------------------------------------------------------------------

def care_price(row: pd.Series, age: int) -> float:
    """What this locality charges to care for a child of this age, over a year."""
    if age >= CARE_FREE_FROM:
        return 0.0
    for low, high, column in CARE_BAND:
        if low <= age <= high:
            value = row.get(column)
            return 0.0 if pd.isna(value) else float(value)
    return 0.0


def bedrooms_needed(n_children: int) -> int:
    """The parent takes a bedroom and the children share two to a room.

    This is the rule the Self-Sufficiency Standard and the New York True Cost of Living study both
    use. A one-child family needs two bedrooms, so the result reduces exactly to the two-bedroom
    figure the rest of the analysis uses.
    """
    return int(min(1 + np.ceil(n_children / 2), 5))


def annual_rent(row: pd.Series, n_children: int) -> float:
    column = {1: "rent_1br", 2: "rent_2br", 3: "rent_3br", 4: "rent_4br",
              5: "rent_5br"}[bedrooms_needed(n_children)]
    value = row.get(column)
    return np.nan if pd.isna(value) else float(value) * 12


def threshold_for(row: pd.Series, ages: list[int], food_scale: float) -> dict:
    """The cost of a year for this family, in this locality, with these children.

    Seven of the eight categories the New York study prices. Savings is deliberately left out, so
    every shortfall reported here is understated rather than overstated.
    """
    n_children = len(ages)
    housing = annual_rent(row, n_children)
    childcare = sum(care_price(row, age) for age in ages)
    transport = float(row["transport_cost_annual"])          # household level, not per person
    food = float(row["food_annual"]) * food_scale            # scaled on the federal food plan
    health = float(row["health_care_annual"]) / 2.0 * (1 + n_children)   # per person
    basic = housing + childcare + transport + food + health
    return {"housing": housing, "childcare": childcare, "transport": transport,
            "food": food, "health_care": health, "basic_needs": basic,
            "other_necessities": basic * OTHER_NECESSITIES,
            "before_tax_threshold": basic * (1 + OTHER_NECESSITIES)}


# --------------------------------------------------------------------------------------------
# Childcare subsidy
# --------------------------------------------------------------------------------------------

def subsidy_for(row: pd.Series, policy: Policy, income: float, ages: list[int],
                childcare_cost: float, rates: pd.DataFrame | None) -> dict:
    """What Virginia's childcare subsidy pays this family this year, under this policy.

    The benefit is the cost of care up to the state's maximum reimbursable rate, less the family
    copayment. Eligibility is the locality's tier as a multiple of the federal poverty guideline,
    which is the lever the re-sort pulls.
    """
    if policy.subsidy_rule == "none":
        return {"subsidy": 0.0, "copay": 0.0, "eligible": False, "group": None}

    column = {"current": "ccsp_group", "floored": "floored_group"}[policy.subsidy_rule]
    group = tier_of(row.get(column))
    if group is None:
        return {"subsidy": 0.0, "copay": 0.0, "eligible": False, "group": None}

    # Only children young enough to need paid care draw a subsidy. Without the out-of-school-time
    # rule the subsidy stops at a child's sixth birthday, which is the three o'clock gap.
    ceiling = CARE_FREE_FROM if policy.out_of_school_time else 6
    covered = [age for age in ages if age < ceiling]
    if not covered or childcare_cost <= 0:
        return {"subsidy": 0.0, "copay": 0.0, "eligible": False, "group": group}

    limit = GROUP_LIMIT[group] * federal_poverty_guideline(1 + len(ages))
    if income > limit and not policy.graduated_taper:
        # The cliff: one dollar over the line and the whole benefit stops.
        return {"subsidy": 0.0, "copay": 0.0, "eligible": False, "group": group}

    # The maximum the state will reimburse, if the published rate is available for this locality.
    cap = childcare_cost
    if rates is not None and row["geoid"] in rates.index:
        per_day = rates.loc[row["geoid"]]
        daily = sum(float(per_day.get(_rate_column(age), np.nan)) for age in covered)
        if np.isfinite(daily) and daily > 0:
            cap = min(childcare_cost, daily * CARE_DAYS_PER_YEAR)

    copay = min(COPAY_CEILING * income, cap)
    if income <= federal_poverty_guideline(1 + len(ages)):
        copay = 0.0     # the copayment is waived at or below the poverty guidelines

    benefit = max(cap - copay, 0.0)
    if policy.graduated_taper and income > limit:
        # Phase the benefit out over the next half of the eligibility band rather than ending it,
        # so the marginal keep rate can never go negative.
        band = 0.5 * limit
        benefit *= float(np.clip(1 - (income - limit) / band, 0.0, 1.0))
    return {"subsidy": benefit, "copay": copay, "eligible": benefit > 0, "group": group}


def _rate_column(age: int) -> str:
    if age <= 2:
        return "daily_mrr_infant"
    return "daily_mrr_toddler" if age <= 5 else "daily_mrr_school_age"


# --------------------------------------------------------------------------------------------
# Non-market resources that change over the lifecycle
# --------------------------------------------------------------------------------------------

def child_support_for(row: pd.Series, policy: Policy, assume: Assumptions,
                      draw: np.random.Generator) -> float:
    """Child support, and what automatic recalculation would add to it.

    Only a minority of these families receive anything. The intervention does not create new
    orders; it recalculates the ones that already exist under the guidelines in force since
    July 2025, which existing orders do not pick up on their own.
    """
    receives = row["OIP"] > 0
    if not receives:
        return 0.0
    current = float(row["OIP"])
    if not policy.child_support_modification:
        return current
    stale = draw.random() < assume.pre_2025_order_share
    return current * (1 + assume.order_modification_uplift) if stale else current


def earnings_path(start: float, years: int, assume: Assumptions, policy: Policy,
                  draw: np.random.Generator) -> np.ndarray:
    """Market earnings over the horizon, with the credential step where it applies."""
    path = start * (1 + assume.wage_growth) ** np.arange(years)
    if not policy.credential_pipeline:
        return path
    if draw.random() >= assume.credential_enrolment_rate:
        return path
    if draw.random() >= assume.credential_completion_rate:
        return path
    start_year = int(draw.integers(0, max(years - assume.credential_years, 1)))
    finish = start_year + assume.credential_years
    path[finish:] *= 1 + assume.credential_earnings_step
    return path


# --------------------------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------------------------

def simulate(cohort: pd.DataFrame, policy: Policy, assume: Assumptions,
             rates: pd.DataFrame | None = None, seed: int = 20261012,
             horizon: int = HORIZON) -> pd.DataFrame:
    """Run one policy world and return one row per family-year."""
    draw = np.random.default_rng(seed)
    records = []

    for _, row in cohort.iterrows():
        ages_start = row["child_ages_start"]
        earnings = earnings_path(max(float(row["market_income"]), 0.0), horizon,
                                 assume, policy, draw)
        support_base = child_support_for(row, policy, assume, draw)

        for year in range(horizon):
            ages = [age + year for age in ages_start]
            resident = [age for age in ages if age < 18]      # a child who turns 18 leaves the unit
            if not resident:
                break
            # Food scales on the federal food plan, which is set by household size.
            food_scale = _food_scale(1 + len(resident))
            cost = threshold_for(row, resident, food_scale)
            if not np.isfinite(cost["before_tax_threshold"]):
                continue

            income = float(earnings[year])
            help_ = subsidy_for(row, policy, income, resident, cost["childcare"], rates)
            support = support_base if any(age < 18 for age in ages) else 0.0

            # What the family actually pays for care. The copayment is already inside the
            # benefit (the subsidy is the reimbursable cost less the copayment), so adding it
            # again here would charge the family twice.
            net_childcare = max(cost["childcare"] - help_["subsidy"], 0.0)
            # The ten percent allowance for everything a budget misses is taken on what the
            # family actually spends, so a subsidy reduces it too.
            basic = cost["basic_needs"] - cost["childcare"] + net_childcare
            threshold = basic * (1 + OTHER_NECESSITIES)

            resources = income + support
            records.append({
                "SERIALNO": row["SERIALNO"], "geoid": row["geoid"],
                "locality": row["locality"], "weight": row["weight"],
                "policy": policy.name, "year": year,
                "n_children": len(resident), "youngest": min(resident),
                "earnings": income, "child_support": support,
                "childcare_market": cost["childcare"], "childcare_net": net_childcare,
                "subsidy": help_["subsidy"], "copay": help_["copay"],
                "subsidy_group": help_["group"],
                "housing": cost["housing"], "transport": cost["transport"],
                "food": cost["food"], "health_care": cost["health_care"],
                "threshold_before_tax": threshold,
                "resources_before_tax": resources,
                "gap_before_tax": resources - threshold,
            })

    return pd.DataFrame(records)


def _food_scale(people: int) -> float:
    """Household size multiplier on the food line, from the federal Thrifty Food Plan.

    The plan's cost for a two-person household is the base, because the published food figure in
    the cost table covers one adult and one child. The ratios below are the plan's own.
    """
    plan = {1: 3804.0, 2: 6423.0, 3: 9198.0, 4: 11676.0, 5: 13860.0,
            6: 16632.0, 7: 18384.0, 8: 21012.0}
    people = int(np.clip(people, 1, 8))
    return plan[people] / plan[2]


def summarise(panel: pd.DataFrame, baseline: str = "Today") -> pd.DataFrame:
    """One row per policy: what a decision-maker compares.

    The share of families short is the headline, but it is the wrong measure of a programme aimed
    at families who are far below the line: help that closes two thirds of a $19,000 gap moves no
    one across it and still changes a family's year. So the table also reports the dollars of
    shortfall closed, and the shortfall closed per dollar of public spending.
    """
    base = panel[panel["policy"] == baseline]
    base_shortfall = float((base["gap_before_tax"].clip(upper=0).abs()
                            * base["weight"]).sum())
    base_cost = float((base["subsidy"] * base["weight"]).sum())
    rows = []
    for name, group in panel.groupby("policy"):
        short = group["gap_before_tax"] < 0
        weight = group["weight"]
        first = group[group["year"] == 0]
        rows.append({
            "policy": name,
            "family_years": len(group),
            "share_short_pct": float(weight[short].sum() / weight.sum() * 100),
            "share_short_year_one_pct": float(
                first.loc[first["gap_before_tax"] < 0, "weight"].sum()
                / first["weight"].sum() * 100),
            "median_gap": float(group["gap_before_tax"].median()),
            "median_childcare_net": float(group["childcare_net"].median()),
            # Weighted to the state: the mean annual subsidy bill across the horizon.
            "subsidy_cost_per_year": float(
                (group["subsidy"] * group["weight"]).groupby(group["year"]).sum().mean()),
            "families_receiving_pct": float(
                group.loc[group["subsidy"] > 0, "weight"].sum() / group["weight"].sum() * 100),
        })
        shortfall = float((group["gap_before_tax"].clip(upper=0).abs()
                           * group["weight"]).sum())
        cost = float((group["subsidy"] * group["weight"]).sum())
        rows[-1]["shortfall_closed"] = (base_shortfall - shortfall) / HORIZON
        rows[-1]["extra_cost"] = (cost - base_cost) / HORIZON
        rows[-1]["closed_per_dollar"] = (
            (base_shortfall - shortfall) / (cost - base_cost)
            if cost > base_cost else np.nan)
    return pd.DataFrame(rows).sort_values("shortfall_closed", ascending=False)


# --------------------------------------------------------------------------------------------
# The scenarios the submission compares
# --------------------------------------------------------------------------------------------

SCENARIOS = [
    Policy("Today", subsidy_rule="current",
           notes="Tiers sorted by the price of care, benefit ends at the child's sixth birthday, "
                 "a hard income cutoff, orders never recalculated."),
    Policy("Re-sort only", subsidy_rule="floored",
           notes="Tiers pegged to the affordability gap; nobody demoted."),
    Policy("Re-sort plus taper", subsidy_rule="floored", graduated_taper=True,
           notes="Adds a phase-out so a raise can never leave a family worse off."),
    Policy("Ages 0 to 12", subsidy_rule="floored", graduated_taper=True,
           out_of_school_time=True,
           notes="Carries the subsidy through the before-school, after-school and summer gap."),
    Policy("Full lifecycle", subsidy_rule="floored", graduated_taper=True,
           out_of_school_time=True, child_support_modification=True, credential_pipeline=True,
           notes="Adds automatic order recalculation and the credential pipeline."),
]


def run_all(seed: int = 20261012) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = load_cohort()
    assume = Assumptions()
    rates_path = PEDIR / "va_ccsp_rates_2024.csv"
    rates = (pd.read_csv(rates_path, dtype={"geoid": str}).set_index("geoid")
             if rates_path.exists() else None)
    if rates is None:
        print("NOTE: reimbursement rates not found; the subsidy is capped at the market price.")
    panel = pd.concat([simulate(cohort, policy, assume, rates, seed) for policy in SCENARIOS],
                      ignore_index=True)
    return panel, summarise(panel)


if __name__ == "__main__":
    panel, summary = run_all()
    panel.to_csv(OUTDIR / "lifecycle_panel.csv", index=False)
    summary.to_csv(OUTDIR / "lifecycle_summary.csv", index=False)
    print(f"{len(panel):,} family-years across {panel['policy'].nunique()} policy worlds\n")
    show = summary[["policy", "share_short_pct", "median_gap", "families_receiving_pct",
                    "shortfall_closed", "extra_cost", "closed_per_dollar"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nParameters a reader should challenge first, because we could not derive them:")
    for name in Assumptions().undocumented():
        print(f"  {name} = {getattr(Assumptions(), name)}")
