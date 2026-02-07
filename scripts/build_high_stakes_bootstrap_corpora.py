#!/usr/bin/env python3
"""
Build bootstrap high-stakes corpora for health, financial regulation, and disaster risk.

These corpora are synthetic benchmark notes designed for fast retrieval/gating iteration.
They are intentionally source-attributed and conflict-rich so uncertainty behavior can be tested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _doc(
    domain: str,
    source: str,
    topic: str,
    stance: str,
    details: str,
    year: int,
    idx: int,
) -> Dict[str, object]:
    title = f"{source} benchmark note on {topic} ({year})"
    content = (
        f"Domain: {domain}. Source: {source}. Topic: {topic}. "
        f"Stance: {stance}. Details: {details}. "
        "This is a synthetic benchmark note for uncertainty-aware RAG testing. "
        "When evidence quality is low or guidance diverges across sources, "
        "the recommended behavior is to state uncertainty and prefer cautious actions."
    )
    return {
        "id": f"{domain}_{idx:04d}",
        "domain": domain,
        "source_org": source,
        "title": title,
        "year": year,
        "topic": topic,
        "content": content,
        "source_type": "synthetic_benchmark_note",
    }


def build_health() -> List[Dict[str, object]]:
    concepts = [
        ("WHO", "guideline development process", "structured evidence-to-decision workflow", "The process links question framing, systematic evidence review, certainty assessment, and transparent recommendation wording."),
        ("CDC", "MMWR recommendation reports", "operational public health recommendation channel", "Reports translate current evidence into implementation guidance for clinicians, laboratories, and state agencies."),
        ("NICE", "evidence grading", "explicit certainty and recommendation-strength mapping", "Guidance separates certainty of evidence from strength of recommendation to expose benefit-harm trade-offs."),
        ("WHO", "living guideline", "continuously updated recommendation set", "Living guidance is revised as pivotal evidence changes expected benefits, harms, or feasibility assumptions."),
        ("NICE", "strength labels", "decision support under variable certainty", "Strength labels indicate whether recommendations are strong, conditional, or evidence-limited."),
    ]

    conflicts = [
        ("WHO", "booster prioritization", "risk-first strategy", "Highest priority is older adults, immunocompromised groups, and frontline health workers when supply is constrained."),
        ("CDC", "booster prioritization", "risk-first with shorter re-eligibility windows", "Priority groups are similar but revaccination windows can be shorter in seasonal high-transmission periods."),
        ("ECDC", "booster prioritization", "risk-first with country-adaptive timing", "Priority is aligned with local burden and uptake constraints; timing can vary by member-state rollout plans."),
        ("WHO", "isolation duration", "symptom- and risk-based release", "Isolation guidance uses symptom resolution and risk profile rather than a single fixed duration across all cases."),
        ("CDC", "isolation duration", "time-based baseline plus masking period", "Guidance provides fixed minimum isolation with additional masking period, with clinical exceptions for severe cases."),
        ("NICE", "isolation duration", "clinical context plus workplace risk", "Return criteria include symptoms, vulnerability of contacts, and care-setting risk."),
        ("WHO", "masking guidance", "risk-tiered masking", "Healthcare settings generally maintain stricter masking than low-risk community settings."),
        ("CDC", "masking guidance", "setting- and transmission-level masking", "Recommendations scale by care setting and local transmission level."),
        ("ECDC", "masking guidance", "scenario-based masking", "Mask recommendations can tighten during waves and relax when pressure on care systems decreases."),
        ("WHO", "antiviral use in mild outpatient cases", "targeted use for high-risk outpatients", "Broad use is discouraged when benefit is uncertain for low-risk mild cases."),
        ("NICE", "antiviral use in mild outpatient cases", "narrow eligibility by clinical risk", "Eligibility is tied to risk stratification and treatment-timing constraints."),
        ("CDC", "antiviral use in mild outpatient cases", "early use for high-risk subgroups", "Emphasis is on early initiation for patients with elevated complication risk."),
        ("WHO", "asymptomatic testing frequency", "context-dependent cadence", "Frequency depends on exposure intensity, outbreak stage, and population vulnerability."),
        ("CDC", "asymptomatic testing frequency", "programmatic cadence by setting", "Schools, congregate settings, and healthcare programs can use different baseline testing intervals."),
        ("NICE", "asymptomatic testing frequency", "cost-benefit constrained cadence", "Testing cadence should reflect local prevalence and service capacity."),
        ("WHO", "conditional vs abstain", "conditional recommendation when plausible net benefit", "If uncertainty is high but expected benefit remains plausible, conditional language is preferred over no recommendation."),
        ("NICE", "conditional vs abstain", "abstain when uncertainty dominates", "When key outcomes are too uncertain, evidence-gap statements can replace directional guidance."),
        ("CDC", "conditional vs abstain", "interim recommendation under active surveillance", "Interim recommendations may be issued with explicit plan for rapid updates."),
    ]

    records: List[Dict[str, object]] = []
    i = 1
    for source, topic, stance, details in concepts + conflicts:
        year = 2021 + (i % 4)
        records.append(_doc("health", source, topic, stance, details, year, i))
        i += 1
    return records


def build_finreg() -> List[Dict[str, object]]:
    concepts = [
        ("BCBS", "BCBS 239 objective", "accurate and timely risk data aggregation", "Principles target governance, architecture, and controls that support decision-useful risk reporting."),
        ("EBA", "supervisory review", "evaluate governance and risk control effectiveness", "Supervisory review tests whether risk governance is credible under stress and normal operations."),
        ("BCBS", "data lineage", "traceability from source to report", "Lineage enables auditability and faster remediation for reporting defects."),
        ("ECB", "stress testing", "forward-looking capital and risk resilience check", "Stress tests evaluate capital adequacy and risk concentration under adverse scenarios."),
        ("OCC", "model risk management", "independent validation and governance controls", "Model risk requires periodic validation, use constraints, and override governance."),
    ]

    conflicts = [
        ("BCBS", "near-real-time aggregation", "target capability for material risks", "Critical risk views should be available quickly during stress events."),
        ("EBA", "near-real-time aggregation", "proportional capability by institution profile", "Expectations vary by complexity and systemic footprint."),
        ("PRA", "near-real-time aggregation", "incident-driven prioritization", "Institutions should demonstrate escalation-specific timeliness rather than universal real-time delivery."),
        ("BCBS", "board accountability for data quality", "explicit board-level ownership", "Board and senior management are directly accountable for persistent quality failures."),
        ("EBA", "board accountability for data quality", "board oversight with delegated control ownership", "Day-to-day control ownership can be delegated with clear accountability mapping."),
        ("FED", "board accountability for data quality", "governance evidence over formal assignment", "Supervisors assess governance effectiveness, not only formal ownership language."),
        ("BCBS", "manual workarounds", "temporary only with remediation plans", "Manual controls are tolerated short-term but should not become permanent architecture."),
        ("ECB", "manual workarounds", "controlled use with audit traceability", "Workarounds can continue if control evidence and audit trails remain robust."),
        ("PRA", "manual workarounds", "acceptable during transition programs", "Large change programs may use controlled interim workarounds."),
        ("BCBS", "climate risk in ICAAP", "integrated into risk taxonomy", "Material climate risk should be reflected in governance, measurement, and planning."),
        ("ECB", "climate risk in ICAAP", "accelerated integration expectations", "Supervisory letters emphasize faster integration timelines for significant institutions."),
        ("EBA", "climate risk in ICAAP", "phased integration with proportionality", "Smaller institutions may follow phased implementation plans."),
        ("BCBS", "AI model explainability", "decision-relevant explainability and control", "Explainability should be sufficient for challenge, monitoring, and audit."),
        ("OCC", "AI model explainability", "risk-based explainability thresholds", "Higher-impact models require stronger explainability and governance controls."),
        ("EBA", "AI model explainability", "consumer and prudential explainability concerns", "Expectations include prudential control and transparency for affected stakeholders."),
        ("BCBS", "material reporting error", "error tied to risk decision impact", "Materiality depends on whether error changes governance decisions or capital views."),
        ("FED", "material reporting error", "thresholds with supervisory judgment", "Numeric thresholds guide triage but supervisory judgment remains central."),
        ("PRA", "material reporting error", "scenario- and context-dependent materiality", "Materiality can vary by stress context and business concentration."),
    ]

    records: List[Dict[str, object]] = []
    i = 1
    for source, topic, stance, details in concepts + conflicts:
        year = 2020 + (i % 5)
        records.append(_doc("finreg", source, topic, stance, details, year, i))
        i += 1
    return records


def build_disaster() -> List[Dict[str, object]]:
    concepts = [
        ("NOAA", "seasonal climate outlook purpose", "probabilistic planning support", "Outlooks provide probability-weighted scenarios to inform preparedness and resource planning."),
        ("UNDRR", "disaster risk reduction", "reduce risk drivers and vulnerability", "Risk reduction combines prevention, preparedness, and resilience investments."),
        ("WMO", "probability ranges", "communicate forecast uncertainty", "Ranges reflect model spread, scenario variation, and observational uncertainty."),
        ("IPCC", "hazard exposure vulnerability", "distinct components of risk", "Hazard is physical event potential, exposure is assets/people in harm zones, vulnerability is susceptibility."),
        ("UNDRR", "early warning systems", "lead-time for protective action", "Early warning systems reduce mortality and losses through timely action protocols."),
    ]

    conflicts = [
        ("NOAA", "near-term drought risk", "high confidence for specific basins", "Near-term drought risk can be elevated where antecedent moisture deficits and forecast signals align."),
        ("IPCC", "near-term drought risk", "moderate confidence with regional heterogeneity", "Regional confidence varies where observational records are short or model agreement is limited."),
        ("WMO", "near-term drought risk", "watchlist framing with uncertainty caveats", "Operational outlooks emphasize monitoring updates as conditions evolve."),
        ("UNDRR", "resilience investment priority", "protect vulnerable populations first", "Budget-limited plans prioritize interventions with highest risk-reduction equity impact."),
        ("National Agency", "resilience investment priority", "critical infrastructure continuity first", "Priority may be assigned to infrastructure dependencies before household-level measures."),
        ("Development Bank", "resilience investment priority", "portfolio diversification approach", "Balanced portfolios can combine high-certainty and exploratory adaptation projects."),
        ("WMO", "heavy precipitation projections", "intensification signal in many regions", "Projected heavy precipitation intensity increases in multiple regions under warming."),
        ("Regional Study", "heavy precipitation projections", "subregional divergence", "Local topography and convective dynamics can produce mixed directionality in some subregions."),
        ("IPCC", "heavy precipitation projections", "confidence tied to event class and region", "Confidence differs across event definitions and model representation quality."),
        ("NOAA", "false alarm vs missed event", "conservative trigger for life safety", "For high-consequence hazards, policy can prefer false alarms over missed events."),
        ("National Meteorological Service", "false alarm vs missed event", "balanced trigger to preserve trust", "Repeated false alarms can reduce compliance, so trigger policy may be balanced."),
        ("UNDRR", "false alarm vs missed event", "context-specific trade-off framing", "Warning policy should account for social vulnerability and response capacity."),
        ("IPCC", "low-probability high-impact scenarios", "explicit scenario inclusion", "Planning should include tail-risk scenarios even when probability is uncertain."),
        ("Regional Planner", "low-probability high-impact scenarios", "screening then staged integration", "Tail scenarios can be screened first and integrated in phases."),
        ("WMO", "low-probability high-impact scenarios", "communication caution", "Tail-risk messaging should avoid false precision and emphasize decision implications."),
        ("UNDRR", "priority groups under uncertainty", "protect highest-vulnerability populations", "Prioritization should account for poverty, exposure, and limited adaptive capacity."),
        ("National Agency", "priority groups under uncertainty", "critical service continuity groups", "Essential workers and service-dependent groups may be prioritized to avoid cascade failures."),
        ("NGO Consortium", "priority groups under uncertainty", "community-led targeting", "Local governance can improve targeting where centralized data is incomplete."),
    ]

    records: List[Dict[str, object]] = []
    i = 1
    for source, topic, stance, details in concepts + conflicts:
        year = 2019 + (i % 6)
        records.append(_doc("disaster", source, topic, stance, details, year, i))
        i += 1
    return records


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-stakes bootstrap corpora")
    parser.add_argument("--out-dir", default="data/corpora", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    health = build_health()
    finreg = build_finreg()
    disaster = build_disaster()

    write_jsonl(out_dir / "health_corpus.jsonl", health)
    write_jsonl(out_dir / "finreg_corpus.jsonl", finreg)
    write_jsonl(out_dir / "disaster_corpus.jsonl", disaster)

    print(f"health_corpus.jsonl: {len(health)} docs")
    print(f"finreg_corpus.jsonl: {len(finreg)} docs")
    print(f"disaster_corpus.jsonl: {len(disaster)} docs")


if __name__ == "__main__":
    main()

