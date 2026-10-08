#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import socket
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_M1 = "Baseline+LCI"
METHOD_PHASE_A = "Baseline+LCI+CED"
METHOD_FULL = "Baseline+LCI+CED+TMR"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fmt(value, digits=4):
    value = number(value)
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def fmt_p(value):
    value = number(value)
    if not math.isfinite(value):
        return "NA"
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_signflip_p(values, seed=42, simulations=100_000):
    values = np.asarray([x for x in values if math.isfinite(x)], dtype=np.float64)
    observed = float(values.mean())
    n = len(values)
    if n <= 20:
        means = []
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            means.append(float(np.mean(values * np.asarray(signs))))
        p = sum(x >= observed - 1e-15 for x in means) / len(means)
        return p, "exact", len(means)
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    block = 10_000
    while done < simulations:
        current = min(block, simulations - done)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(current, n))
        means = (signs * values[None, :]).mean(axis=1)
        exceed += int(np.sum(means >= observed - 1e-15))
        done += current
    return (exceed + 1) / (simulations + 1), "Monte-Carlo", simulations


def paired_stats(values, seed=42, bootstrap=10_000):
    values = np.asarray([x for x in values if math.isfinite(x)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(bootstrap, len(values)), replace=True).mean(axis=1)
    p, p_kind, permutations = paired_signflip_p(values, seed=seed)
    return {
        "tail_class_count": int(len(values)),
        "mean_difference": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "one_sided_signflip_p": float(p),
        "signflip_kind": p_kind,
        "signflip_samples": permutations,
        "positive_class_fraction": float(np.mean(values > 0)),
    }


def group_metric(rows, method, metric, *, constraint=None):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("method") != method:
            continue
        if row.get("tail_name") == "__MACRO__" or row.get("row_type"):
            continue
        if str(row.get("active_pair", "")).lower() != "true":
            continue
        if constraint is not None and row.get("constraint") != constraint:
            continue
        value = number(row.get(metric))
        if math.isfinite(value):
            grouped[int(row["tail_index"])].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def compare_grouped(first, second, direction):
    keys = sorted(set(first) & set(second))
    if direction == "second_minus_first":
        diffs = [second[key] - first[key] for key in keys]
    elif direction == "first_minus_second":
        diffs = [first[key] - second[key] for key in keys]
    else:
        raise ValueError(direction)
    return paired_stats(diffs)


def macro_row(rows, method):
    for row in rows:
        if row.get("method") == method and row.get("row_type") == "active_pair_macro":
            return row
    raise KeyError(method)


def operating_macro(rows, method, constraint):
    for row in rows:
        if row.get("method") == method and row.get("row_type") == "active_pair_macro" and row.get("constraint") == constraint:
            return row
    raise KeyError((method, constraint))


def lci_aggregate(rows, method, intervention, mask_type, ratio):
    for row in rows:
        if (
            row.get("method") == method
            and row.get("row_type") == "aggregate"
            and row.get("intervention") == intervention
            and row.get("mask_type") == mask_type
            and abs(number(row.get("top_ratio")) - ratio) < 1e-9
        ):
            return row
    raise KeyError((method, intervention, mask_type, ratio))


def method_label(method):
    return {
        "Baseline": "Baseline",
        METHOD_M1: "M1 (LCI)",
        METHOD_PHASE_A: "M2-Phase A（证据解耦）",
        METHOD_FULL: "M2 完整 CEDR",
    }.get(method, method)


def build_dataset_bundle(root: Path, dataset: str):
    base = root / "runs/evidence_validation_v2" / dataset
    return {
        "overall": read_csv(base / "overall_metrics.csv"),
        "subset": read_csv(base / "subset_metrics.csv"),
        "lci": read_csv(base / "lci_intervention_metrics.csv"),
        "overlap": read_csv(base / "ced_overlap_summary.csv"),
        "routing": read_csv(base / "ced_routing_control_metrics.csv"),
        "branch": read_csv(base / "ced_branch_metrics.csv"),
        "margin": read_csv(base / "tmr_margin_summary.csv"),
        "ablation": read_csv(base / "tmr_ablation.csv"),
        "operating": read_csv(base / "operating_point_metrics.csv"),
        "calibration": read_csv(base / "calibration_metrics.csv"),
        "correlation": read_csv(base / "correlation_metrics.csv"),
        "manifest": read_json(base / "run_manifest.json"),
        "base": base,
    }


def make_paired_stats(bundle):
    subset = bundle["subset"]
    operating = bundle["operating"]
    definitions = {
        "de_cooccur_ap_gain": (subset, "de_cooccur_tail_ap", None, "second_minus_first"),
        "hard_negative_auprc_gain": (subset, "hard_negative_auprc_head_present_cohort", None, "second_minus_first"),
        "fixed_0_5_fpr_reduction": (subset, "hard_negative_tail_fpr", None, "first_minus_second"),
        "fpr_at_recall_60_reduction": (operating, "fpr", "fixed_recall", "first_minus_second"),
        "recall_at_fpr_0_5_gain": (operating, "recall", "fixed_fpr", "second_minus_first"),
    }
    output = {}
    for name, (rows, metric, constraint, direction) in definitions.items():
        first = group_metric(rows, METHOD_M1, metric, constraint=constraint)
        second = group_metric(rows, METHOD_FULL, metric, constraint=constraint)
        output[name] = compare_grouped(first, second, direction)
    return output


def group_lci_metric(rows, method, intervention, mask_type, metric, ratio=0.15):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("row_type") != "sample_tail" or row.get("method") != method:
            continue
        if row.get("intervention") != intervention or row.get("mask_type") != mask_type:
            continue
        if abs(number(row.get("top_ratio")) - ratio) > 1e-9:
            continue
        value = number(row.get(metric))
        if math.isfinite(value):
            grouped[int(row["tail_index"])].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def make_m1_stats(bundle):
    rows = bundle["lci"]
    b_context = group_lci_metric(rows, "Baseline", "context_swap", "attention_top", "absolute_change")
    m_context = group_lci_metric(rows, METHOD_M1, "context_swap", "attention_top", "absolute_change")
    b_target = group_lci_metric(rows, "Baseline", "target_weaken", "attention_top", "probability_drop")
    m_target = group_lci_metric(rows, METHOD_M1, "target_weaken", "attention_top", "probability_drop")
    m_random = group_lci_metric(rows, METHOD_M1, "target_weaken", "random_same_area", "probability_drop")
    return {
        "context_change_reduction": compare_grouped(b_context, m_context, "first_minus_second"),
        "target_necessity_gain": compare_grouped(b_target, m_target, "second_minus_first"),
        "m1_target_minus_random_specificity": compare_grouped(m_random, m_target, "second_minus_first"),
    }


def main_table_voc(two_module):
    rows = []
    for item in two_module["rows"]:
        rows.append([
            item["method"],
            "✓" if item.get("M1_LCI") else "",
            "✓" if item.get("M2_CEDR") else "",
            fmt(item["overall_map"]), fmt(item["head_map"]), fmt(item["middle_map"]), fmt(item["tail_map"]),
        ])
    return rows


def main_table_coco(formal):
    ds = formal["coco_lt"]
    mapping = [("Baseline", "baseline", "", ""), ("+ M1", "module1", "✓", ""), ("+ M1 + M2", "module3", "✓", "✓")]
    return [[name, m1, m2, fmt(ds[key]["overall_map"]), fmt(ds[key]["head_map"]), fmt(ds[key]["middle_map"]), fmt(ds[key]["tail_map"])] for name, key, m1, m2 in mapping]


def append_dataset_sections(lines, dataset_label, bundle, stats):
    lci = bundle["lci"]
    subset = bundle["subset"]
    operating = bundle["operating"]

    lines += [f"### {dataset_label}", ""]
    top_rows = []
    for method in ("Baseline", METHOD_M1):
        context = lci_aggregate(lci, method, "context_swap", "attention_top", 0.15)
        target = lci_aggregate(lci, method, "target_weaken", "attention_top", 0.15)
        random = lci_aggregate(lci, method, "target_weaken", "random_same_area", 0.15)
        anti = lci_aggregate(lci, method, "target_weaken", "anti_attention", 0.15)
        top_rows.append([method_label(method), fmt(context["absolute_change"]), fmt(target["probability_drop"]), fmt(random["probability_drop"]), fmt(anti["probability_drop"]), context["sample_count"]])
    lines += [md_table(["模型", "场景替换绝对变化↓", "目标核心削弱下降↑", "随机同面积下降", "反注意力下降", "场景替换事件数"], top_rows), ""]

    scale_rows = []
    for ratio in (0.10, 0.15, 0.20):
        bctx = lci_aggregate(lci, "Baseline", "context_swap", "attention_top", ratio)
        mctx = lci_aggregate(lci, METHOD_M1, "context_swap", "attention_top", ratio)
        btgt = lci_aggregate(lci, "Baseline", "target_weaken", "attention_top", ratio)
        mtgt = lci_aggregate(lci, METHOD_M1, "target_weaken", "attention_top", ratio)
        scale_rows.append([f"{int(ratio*100)}%", fmt(bctx["absolute_change"]), fmt(mctx["absolute_change"]), fmt(number(bctx["absolute_change"]) - number(mctx["absolute_change"])), fmt(btgt["probability_drop"]), fmt(mtgt["probability_drop"]), fmt(number(mtgt["probability_drop"]) - number(btgt["probability_drop"]))])
    lines += [md_table(["保留/削弱面积", "Baseline 场景变化", "M1 场景变化", "波动降低", "Baseline 核心下降", "M1 核心下降", "核心依赖增量"], scale_rows), ""]

    b = lci_aggregate(lci, "Baseline", "context_swap", "attention_top", 0.15)
    m = lci_aggregate(lci, METHOD_M1, "context_swap", "attention_top", 0.15)
    lines += [f"解释：M1 将 15% 核心保留条件下的外围场景替换波动从 {fmt(b['absolute_change'])} 降至 {fmt(m['absolute_change'])} 个概率百分点；同时真实注意力核心的削弱效应远大于随机同面积与反注意力区域。这分别对应“外围场景不应替代目标”和“预测仍应依赖目标核心”两个互补条件。", ""]

    m1_overall = next(row for row in bundle["overall"] if row["method"] == METHOD_M1)
    full_overall = next(row for row in bundle["overall"] if row["method"] == METHOD_FULL)
    m1_sub = macro_row(subset, METHOD_M1)
    full_sub = macro_row(subset, METHOD_FULL)
    m1_r60 = operating_macro(operating, METHOD_M1, "fixed_recall")
    full_r60 = operating_macro(operating, METHOD_FULL, "fixed_recall")
    m1_fpr = operating_macro(operating, METHOD_M1, "fixed_fpr")
    full_fpr = operating_macro(operating, METHOD_FULL, "fixed_fpr")
    behavior_rows = [
        ["M1", fmt(m1_overall["tail_map"]), fmt(m1_sub["de_cooccur_tail_ap"]), fmt(m1_sub["hard_negative_auprc_head_present_cohort"]), fmt(m1_sub["hard_negative_auroc_head_present_cohort"]), fmt(m1_sub["hard_negative_tail_fpr"]), fmt(m1_r60["fpr"]), fmt(m1_fpr["recall"])],
        ["M2 完整 CEDR", fmt(full_overall["tail_map"]), fmt(full_sub["de_cooccur_tail_ap"]), fmt(full_sub["hard_negative_auprc_head_present_cohort"]), fmt(full_sub["hard_negative_auroc_head_present_cohort"]), fmt(full_sub["hard_negative_tail_fpr"]), fmt(full_r60["fpr"]), fmt(full_fpr["recall"])],
    ]
    lines += [md_table(["模型", "Tail mAP↑", "去共现 AP↑", "Head在场 AUPRC↑", "Head在场 AUROC↑", "固定0.5困难负FPR↓", "FPR@R60↓", "R@FPR0.5↑"], behavior_rows), ""]

    stat_labels = {
        "de_cooccur_ap_gain": "去共现 AP 增益",
        "hard_negative_auprc_gain": "Head在场 AUPRC 增益",
        "fixed_0_5_fpr_reduction": "固定0.5 FPR 降幅",
        "fpr_at_recall_60_reduction": "FPR@R60 降幅",
        "recall_at_fpr_0_5_gain": "R@FPR0.5 增益",
    }
    stat_rows = []
    for key, label in stat_labels.items():
        item = stats[key]
        stat_rows.append([label, fmt(item["mean_difference"]), f"[{fmt(item['ci95'][0])}, {fmt(item['ci95'][1])}]", fmt_p(item["one_sided_signflip_p"]), item["signflip_kind"], item["tail_class_count"], fmt(100 * item["positive_class_fraction"], 1) + "%"])
    lines += [md_table(["类别配对量", "M2 改善均值", "95% class-bootstrap CI", "单侧符号翻转 p", "检验", "Tail类数", "改善类比例"], stat_rows), ""]

    overlap_rows = [[row["reference_type"], fmt(row["overlap_reduction_mean"]), fmt(row["overlap_reduction_std"]), f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]", fmt_p(row["paired_p_vs_real"]), row["tail_class_count"]] for row in bundle["overlap"]]
    lines += [md_table(["Head参照", "raw→private overlap降幅↑", "标准差", "95% CI", "与真实配对比较p", "Tail类数"], overlap_rows), ""]

    routing_rows = [[row["reference_type"], fmt(row["tail_map"]), fmt(row["de_cooccur_ap"]), fmt(row["hard_negative_fpr_threshold_0_5"]), fmt(row["hard_negative_fpr_at_recall_60"]), fmt(row["head_present_auprc"])] for row in bundle["routing"]]
    lines += [md_table(["推理路由", "Tail mAP↑", "去共现 AP↑", "固定0.5 FPR↓", "FPR@R60↓", "Head在场 AUPRC↑"], routing_rows), ""]

    branch_rows = [[row["branch"], fmt(row["de_cooccur_ap"]), fmt(row["hard_negative_fpr_threshold_0_5"]), fmt(row["hard_negative_fpr_at_recall_60"]), fmt(row["head_present_auprc"])] for row in bundle["branch"]]
    lines += [md_table(["证据分支", "去共现 AP↑", "固定0.5 FPR↓", "FPR@R60↓", "Head在场 AUPRC↑"], branch_rows), ""]

    margin = bundle["margin"][0]
    lines += [md_table(["正样本margin", "困难负margin", "margin AUROC↑", "正样本margin>0", "困难负margin<0", "Cohen's d"], [[fmt(margin["positive_margin_mean"]), fmt(margin["hard_negative_margin_mean"]), fmt(margin["margin_auroc"]), fmt(margin["positive_margin_gt_zero_percent"]), fmt(margin["hard_negative_margin_lt_zero_percent"]), fmt(margin["cohens_d_positive_minus_hard_negative"])] ]), ""]

    ablation_rows = [[row["method"], fmt(row["overall_map"]), fmt(row["tail_map"]), fmt(row["de_cooccur_ap"]), fmt(row["hard_negative_fpr_at_recall_60"]), fmt(row["precision_at_recall_60"])] for row in bundle["ablation"]]
    lines += [md_table(["M2内部推理配置", "Overall mAP↑", "Tail mAP↑", "去共现 AP↑", "FPR@R60↓", "P@R60↑"], ablation_rows), ""]

    cal_rows = [[method_label(row["method"]), fmt(row["ece"]), fmt(row["brier_score"]), fmt(row["best_f1_threshold"]), fmt(row["best_f1"]), fmt(row["auroc"]), fmt(row["auprc"])] for row in bundle["calibration"]]
    lines += [md_table(["模型/阶段", "ECE↓", "Brier↓", "事后最佳F1阈值", "最佳F1", "AUROC↑", "AUPRC↑"], cal_rows), ""]

    corr_rows = [[row["analysis"], row["tail_class_count"], fmt(row["spearman_rho"]), fmt_p(row["approximate_p_value"])] for row in bundle["correlation"]]
    lines += [md_table(["类别级一致性分析", "Tail类数", "Spearman ρ", "近似p"], corr_rows), ""]


def append_m1_tables(lines, dataset_label, bundle, stats):
    lci = bundle["lci"]
    lines += [f"### {dataset_label}", ""]
    rows = []
    for method in ("Baseline", METHOD_M1):
        context = lci_aggregate(lci, method, "context_swap", "attention_top", 0.15)
        target = lci_aggregate(lci, method, "target_weaken", "attention_top", 0.15)
        random = lci_aggregate(lci, method, "target_weaken", "random_same_area", 0.15)
        anti = lci_aggregate(lci, method, "target_weaken", "anti_attention", 0.15)
        rows.append([method_label(method), fmt(context["absolute_change"]), fmt(target["probability_drop"]), fmt(random["probability_drop"]), fmt(anti["probability_drop"]), context["sample_count"]])
    lines += [md_table(["模型", "场景替换绝对变化↓", "目标核心削弱下降↑", "随机同面积下降", "反注意力下降", "场景替换事件数"], rows), ""]
    scale_rows = []
    for ratio in (0.10, 0.15, 0.20):
        bctx = lci_aggregate(lci, "Baseline", "context_swap", "attention_top", ratio)
        mctx = lci_aggregate(lci, METHOD_M1, "context_swap", "attention_top", ratio)
        btgt = lci_aggregate(lci, "Baseline", "target_weaken", "attention_top", ratio)
        mtgt = lci_aggregate(lci, METHOD_M1, "target_weaken", "attention_top", ratio)
        scale_rows.append([f"{int(ratio*100)}%", fmt(bctx["absolute_change"]), fmt(mctx["absolute_change"]), fmt(number(bctx["absolute_change"]) - number(mctx["absolute_change"])), fmt(btgt["probability_drop"]), fmt(mtgt["probability_drop"]), fmt(number(mtgt["probability_drop"]) - number(btgt["probability_drop"]))])
    lines += [md_table(["核心面积", "Baseline 场景变化", "M1 场景变化", "波动降低", "Baseline 核心下降", "M1 核心下降", "核心依赖增量"], scale_rows), ""]
    labels = {
        "context_change_reduction": "场景替换波动降低",
        "target_necessity_gain": "目标核心必要性增益",
        "m1_target_minus_random_specificity": "M1真实核心−随机同面积",
    }
    stat_rows = []
    for key, label in labels.items():
        item = stats[key]
        stat_rows.append([label, fmt(item["mean_difference"]), f"[{fmt(item['ci95'][0])}, {fmt(item['ci95'][1])}]", fmt_p(item["one_sided_signflip_p"]), item["signflip_kind"], item["tail_class_count"], fmt(100 * item["positive_class_fraction"], 1) + "%"])
    lines += [md_table(["类别配对量", "M1改善均值", "95% class-bootstrap CI", "单侧p", "符号翻转", "Tail类数", "改善类比例"], stat_rows), ""]


def append_behavior_tables(lines, dataset_label, bundle, stats):
    subset = bundle["subset"]
    operating = bundle["operating"]
    m1_overall = next(row for row in bundle["overall"] if row["method"] == METHOD_M1)
    full_overall = next(row for row in bundle["overall"] if row["method"] == METHOD_FULL)
    m1_sub, full_sub = macro_row(subset, METHOD_M1), macro_row(subset, METHOD_FULL)
    m1_r60, full_r60 = operating_macro(operating, METHOD_M1, "fixed_recall"), operating_macro(operating, METHOD_FULL, "fixed_recall")
    m1_fpr, full_fpr = operating_macro(operating, METHOD_M1, "fixed_fpr"), operating_macro(operating, METHOD_FULL, "fixed_fpr")
    rows = [
        ["M1", fmt(m1_overall["tail_map"]), fmt(m1_sub["de_cooccur_tail_ap"]), fmt(m1_sub["hard_negative_auprc_head_present_cohort"]), fmt(m1_sub["hard_negative_auroc_head_present_cohort"]), fmt(m1_sub["hard_negative_tail_fpr"]), fmt(m1_r60["fpr"]), fmt(m1_fpr["recall"])],
        ["M2 完整 CEDR", fmt(full_overall["tail_map"]), fmt(full_sub["de_cooccur_tail_ap"]), fmt(full_sub["hard_negative_auprc_head_present_cohort"]), fmt(full_sub["hard_negative_auroc_head_present_cohort"]), fmt(full_sub["hard_negative_tail_fpr"]), fmt(full_r60["fpr"]), fmt(full_fpr["recall"])],
    ]
    lines += [f"### {dataset_label}", "", md_table(["模型", "Tail mAP↑", "去共现 AP↑", "Head在场 AUPRC↑", "Head在场 AUROC↑", "固定0.5困难负FPR↓", "FPR@R60↓", "R@FPR0.5↑"], rows), ""]
    labels = {
        "de_cooccur_ap_gain": "去共现 AP 增益",
        "hard_negative_auprc_gain": "Head在场 AUPRC 增益",
        "fixed_0_5_fpr_reduction": "固定0.5 FPR 降幅",
        "fpr_at_recall_60_reduction": "FPR@R60 降幅",
        "recall_at_fpr_0_5_gain": "R@FPR0.5 增益",
    }
    stat_rows = []
    for key, label in labels.items():
        item = stats[key]
        stat_rows.append([label, fmt(item["mean_difference"]), f"[{fmt(item['ci95'][0])}, {fmt(item['ci95'][1])}]", fmt_p(item["one_sided_signflip_p"]), item["signflip_kind"], item["tail_class_count"], fmt(100 * item["positive_class_fraction"], 1) + "%"])
    lines += [md_table(["类别配对量", "M2改善均值", "95% class-bootstrap CI", "单侧p", "符号翻转", "Tail类数", "改善类比例"], stat_rows), ""]


def append_overlap_tables(lines, dataset_label, bundle):
    overlap_rows = [[row["reference_type"], fmt(row["overlap_reduction_mean"]), fmt(row["overlap_reduction_std"]), f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]", fmt_p(row["paired_p_vs_real"]), row["tail_class_count"]] for row in bundle["overlap"]]
    routing_rows = [[row["reference_type"], fmt(row["tail_map"]), fmt(row["de_cooccur_ap"]), fmt(row["hard_negative_fpr_threshold_0_5"]), fmt(row["hard_negative_fpr_at_recall_60"]), fmt(row["head_present_auprc"])] for row in bundle["routing"]]
    lines += [f"### {dataset_label}", "", md_table(["Head参照", "raw→private overlap降幅↑", "标准差", "95% CI", "与真实配对比较p", "Tail类数"], overlap_rows), "", md_table(["推理路由", "Tail mAP↑", "去共现 AP↑", "固定0.5 FPR↓", "FPR@R60↓", "Head在场 AUPRC↑"], routing_rows), ""]


def append_branch_tables(lines, dataset_label, bundle):
    rows = [[row["branch"], fmt(row["de_cooccur_ap"]), fmt(row["hard_negative_fpr_threshold_0_5"]), fmt(row["hard_negative_fpr_at_recall_60"]), fmt(row["head_present_auprc"])] for row in bundle["branch"]]
    lines += [f"### {dataset_label}", "", md_table(["证据分支", "去共现 AP↑", "固定0.5 FPR↓", "FPR@R60↓", "Head在场 AUPRC↑"], rows), ""]


def append_margin_tables(lines, dataset_label, bundle):
    row = bundle["margin"][0]
    values = [[fmt(row["positive_margin_mean"]), fmt(row["hard_negative_margin_mean"]), fmt(row["margin_auroc"]), fmt(row["positive_margin_gt_zero_percent"]), fmt(row["hard_negative_margin_lt_zero_percent"]), fmt(row["cohens_d_positive_minus_hard_negative"]), row["positive_count"], row["hard_negative_count"]]]
    lines += [f"### {dataset_label}", "", md_table(["正样本margin", "困难负margin", "margin AUROC↑", "正样本>0", "困难负<0", "Cohen's d", "正样本数", "困难负数"], values), ""]


def append_ablation_tables(lines, dataset_label, bundle):
    rows = [[row["method"], fmt(row["overall_map"]), fmt(row["tail_map"]), fmt(row["de_cooccur_ap"]), fmt(row["hard_negative_fpr_at_recall_60"]), fmt(row["precision_at_recall_60"])] for row in bundle["ablation"]]
    lines += [f"### {dataset_label}", "", md_table(["M2内部推理配置", "Overall mAP↑", "Tail mAP↑", "去共现 AP↑", "FPR@R60↓", "P@R60↑"], rows), ""]


def append_calibration_tables(lines, dataset_label, bundle):
    rows = [[method_label(row["method"]), fmt(row["ece"]), fmt(row["brier_score"]), fmt(row["best_f1_threshold"]), fmt(row["best_f1"]), fmt(row["auroc"]), fmt(row["auprc"])] for row in bundle["calibration"]]
    lines += [f"### {dataset_label}", "", md_table(["模型/阶段", "ECE↓", "Brier↓", "事后最佳F1阈值", "最佳F1", "AUROC↑", "AUPRC↑"], rows), ""]


def append_correlation_tables(lines, dataset_label, bundle):
    rows = [[row["analysis"], row["tail_class_count"], fmt(row["spearman_rho"]), fmt_p(row["approximate_p_value"])] for row in bundle["correlation"]]
    lines += [f"### {dataset_label}", "", md_table(["类别级一致性分析", "Tail类数", "Spearman ρ", "近似p"], rows), ""]


def run(args):
    root = Path(args.project_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    two_root = Path(args.two_module_root).resolve()
    two_ablation = read_json(two_root / "ablation/ablation_two_module.json")
    formal = read_json(root / "final_results/summary.json")
    bundles = {name: build_dataset_bundle(root, name) for name in ("voc", "coco")}
    m1_stats = {name: make_m1_stats(bundle) for name, bundle in bundles.items()}
    stats = {name: make_paired_stats(bundle) for name, bundle in bundles.items()}

    lines = [
        "# CIDER 两模块全部实验表与有效性验证总汇总",
        "",
        "> 本文档将每个实验独立成表，并将 VOC-LT、COCO-LT 分开报告。所有数值来自服务器既有锁定 checkpoint 与只读逐事件/逐类结果；本轮没有训练、调参、换 checkpoint、改阈值或占用 GPU 重新筛选。",
        "",
        "## 1. 统一问题与两模块分工",
        "",
        "研究问题是：在 Tail 类视觉监督不足时，如何避免高频 Head 共现对象和共享场景替代 Tail 目标自身成为主要判别依据，并提高真实 Tail 在非常见共现场景下的稳定性。M1/LCI 专门处理候选区域外的共享场景替代；M2/CEDR 专门处理候选区域内 Tail–Head 混合证据及 Head-only 困难负样本。",
        "",
        "所有实验使用 ResNet-50、448×448、ASL、官方 class_split.pkl、seed=42、单视图评估。Tail–Head Top-3 关系只从官方训练标签统计。由于只有一组训练 seed，本报告的 bootstrap/符号翻转是类别级不确定性，不能冒充多训练种子。",
        "",
        "## 2. 实验一：两模块主消融",
        "",
        "该实验只回答整体分类性能是否提升；条件 AP、FPR、Recall 不混入 mAP 表。旧 CED 中间阶段不作为论文顶层模块单列。",
        "",
        "### 2.1 VOC-LT",
        "",
        md_table(["方法", "M1", "M2", "Overall mAP", "Head mAP", "Middle mAP", "Tail mAP"], main_table_voc(two_ablation)),
        "",
        "VOC-LT 中完整两模块相对 Baseline 的 Overall/Tail mAP 分别提高 4.2226/7.7581 点；相对 M1，M2 使 Tail mAP 再提高 3.8450 点。",
        "",
        "### 2.2 COCO-LT",
        "",
        md_table(["方法", "M1", "M2", "Overall mAP", "Head mAP", "Middle mAP", "Tail mAP"], main_table_coco(formal)),
        "",
        "COCO-LT 中完整两模块相对 Baseline 的 Overall/Tail mAP 分别提高 2.4512/4.7320 点；相对 M1，M2 使 Tail mAP 再提高 3.9091 点。",
        "",
        "## 3. 实验二：M1 外围场景替代与目标必要性验证",
        "",
        "固定 Top-15% Tail 响应区作为候选目标核心：场景替换保留核心、只替换外围；目标削弱保留外围、只中和核心；随机同面积与反注意力区域排除面积效应。场景变化越小越好，目标核心削弱下降越大越好。第二张表同时报告 10%/15%/20%，避免只展示单一掩码面积。",
        "",
    ]
    append_m1_tables(lines, "3.1 VOC-LT", bundles["voc"], m1_stats["voc"])
    append_m1_tables(lines, "3.2 COCO-LT", bundles["coco"], m1_stats["coco"])
    lines += [
        "结论：两套数据在 10%/15%/20% 三种面积下均表现为 M1 场景波动更小、目标核心削弱下降更大；随机同面积和反注意力区域效应远小于真实核心。因此 M1 的有效性不是单一面积或纯遮挡造成。",
        "",
        "## 4. 实验三：M2 行为结果——去共现真阳性与 Head-only 困难负样本",
        "",
        "这里直接比较 M1 与完整 M2。去共现 AP 衡量 Head 缺席时真实 Tail 的排序；Head在场 AUPRC/AUROC 衡量 Tail 真阳性与 Head-only 困难负样本的分离；固定阈值和两个公平操作点单独列出，防止仅靠整体压低分数制造低误报。第二张表在 Tail 类层面做配对 bootstrap 与单侧符号翻转。正的“改善均值”表示方向有利于 M2。本节使用跨数据集齐全的 evidence-v2 归档 checkpoint；VOC 最新两模块主表使用另一套已锁定最终 T1，因此两节数值不得拼成同一 checkpoint 的结果。",
        "",
    ]
    append_behavior_tables(lines, "4.1 VOC-LT", bundles["voc"], stats["voc"])
    append_behavior_tables(lines, "4.2 COCO-LT", bundles["coco"], stats["coco"])
    lines += [
        "结论：两套数据的 Tail mAP、去共现 AP、Head在场 AUPRC、固定0.5困难负 FPR 和 R@FPR0.5 均改善；但 FPR@R60 与 AUROC 并非一致改善，因此 M2 的优势主要体现在 AP/AUPRC、低误报召回与校准，不能写成整条 ROC 曲线全面支配。",
        "",
        "## 5. 实验四：M2-Phase A 定向空间去重叠与路由对照",
        "",
        "overlap 定义为 `sum(min(A_tail,A_head))/sum(A_tail)`，比较 raw→private 降幅。random Head 和 shuffled Tail–Head 对照不使用测试标签选关系。路由表是同一 checkpoint 的推理时功能性扰动，不是重新训练。",
        "",
    ]
    append_overlap_tables(lines, "5.1 VOC-LT", bundles["voc"])
    append_overlap_tables(lines, "5.2 COCO-LT", bundles["coco"])
    lines += [
        "结论：VOC 对 random Head 的定向优势显著，但对 shuffled Head 不显著；COCO 对两种对照均显著。真实路由在 COCO 的 Tail mAP 最好，但并非每个行为指标都最好；VOC 真实路由也没有稳定支配随机路由。因此空间去重叠证据成立，但 Top-3 路由功能性证据在 VOC 仍有限。",
        "",
        "## 6. 实验五：M2-Phase A private/shared 分支功能",
        "",
        "private-only、shared-only 是同一已训练解耦模块的分支分数，未独立校准；其固定0.5 FPR 只作诊断。完整 CED 分数用于检查两分支组合后能否恢复实际分类能力。",
        "",
    ]
    append_branch_tables(lines, "6.1 VOC-LT", bundles["voc"])
    append_branch_tables(lines, "6.2 COCO-LT", bundles["coco"])
    lines += [
        "结论：两套数据的 private-only 去共现 AP 都远高于 shared-only，说明分离后的 private 分支保留了更多 Tail 证据；完整解耦输出显著优于任一单分支。由于分支未独立校准，不能用其绝对 FPR 证明单分支可直接部署。",
        "",
        "## 7. 实验六：M2-Phase B 正负记忆 margin 可分性",
        "",
        "margin 比较 Tail 正原型相似度与 Head-present/Tail-absent 困难负原型相似度。AUROC 越高、正负均值间隔和 Cohen's d 越大，说明 private 证据越容易被可靠性复核区分。",
        "",
    ]
    append_margin_tables(lines, "7.1 VOC-LT", bundles["voc"])
    append_margin_tables(lines, "7.2 COCO-LT", bundles["coco"])
    lines += [
        "结论：VOC/COCO margin AUROC 分别为 98.8419/94.0730，且困难负 margin 几乎都小于 0，支持正负记忆对真假 Tail 证据具有较强判别信息。",
        "",
        "## 8. 实验七：M2 内部可靠性精炼组件消融",
        "",
        "这些行是同一训练完成模型的推理分支移除，不是四个重新训练模型；因此可用于说明已学习组件的功能贡献，不能替代独立训练消融。",
        "",
    ]
    append_ablation_tables(lines, "8.1 VOC-LT", bundles["voc"])
    append_ablation_tables(lines, "8.2 COCO-LT", bundles["coco"])
    lines += [
        "结论：加入 shared reference 和视觉门控后，两套数据的 Tail mAP 与去共现 AP 均继续提高；但完整视觉门控在 FPR@R60 上不一定最优，说明其收益来自综合排序/召回权衡，而非单一工作点。",
        "",
        "## 9. 实验八：校准与阈值敏感性",
        "",
        "ECE/Brier 越低越好。最佳 F1 阈值是事后诊断，只用于解释固定0.5指标受校准影响，不参与正式结果或模型选择。",
        "",
    ]
    append_calibration_tables(lines, "9.1 VOC-LT", bundles["voc"])
    append_calibration_tables(lines, "9.2 COCO-LT", bundles["coco"])
    lines += [
        "结论：完整 M2 在两套数据上都显著降低 ECE/Brier 并提高 AUPRC；COCO AUROC 有下降，进一步说明必须同时报告排序、校准与公平操作点。",
        "",
        "## 10. 实验九：类别级机制一致性",
        "",
        "Spearman 分析检查“机制改善更大的 Tail 类是否也获得更大行为收益”。它是关联证据，不是严格因果证明。",
        "",
    ]
    append_correlation_tables(lines, "10.1 VOC-LT", bundles["voc"])
    append_correlation_tables(lines, "10.2 COCO-LT", bundles["coco"])
    lines += [
        "结论：多数类别级相关性不显著；COCO 的 TMR margin AUROC 与 FPR reduction 显著但方向为负。这些结果限制了“类间单调机制对应”的强表述，但不否定总体行为和干预结果。",
        "",
    ]

    lines += [
        "## 11. M2 改进后验证链的解释",
        "",
        "本轮对 M2 的改进是验证设计改进，而不是模型改动：把单一阈值结论改为四层证据链。第一层看 M1→M2 的总体与条件排序；第二层要求真实 Head 的 raw→private overlap 降幅超过随机/打乱 Head；第三层验证正样本与困难负样本的原型 margin 可分；第四层用同一训练完成模型的推理分支移除检查组件是否提供增益。",
        "",
        "VOC-LT 的空间去重叠相对随机 Head 显著，但相对 shuffled Head 不显著；COCO-LT 对两个对照均显著。两套数据上 margin AUROC 均很高，支持记忆可靠性复核具有判别信息。与此同时，FPR@R60 在部分完整模型对比中变差，VOC 路由真实配对也并非所有指标都优于随机路由。因此正确结论是“多项排序、低 FPR 召回、校准和机制指标共同支持缓解证据替代”，而不是“所有阈值和所有机制检查均改善”。",
        "",
        "## 12. 实验十：重构数值等价与 checkpoint 可追踪性",
        "",
        "### 12.1 VOC-LT",
        "",
        md_table(["测试", "VOC-LT最大差异", "阈值", "结论"], [["旧M1→新M1 logits", "0", "1e-6", "通过"], ["旧Full→新CEDR logits", "0", "1e-6", "通过"], ["private/shared/memory/final residual", "0", "1e-6", "通过"], ["Head/Middle logits变化", "0", "0", "通过"]]),
        "",
        "### 12.2 COCO-LT",
        "",
        md_table(["对象", "格式/状态", "SHA256", "数值回归结论"], [["M1 composite", "two_module_v1，迁移完成", "7e2c34fe94134cf8124ac67bf7181da010dbab5ceb87bc8822849af56d7dc97c", "未单独记录逐样本差异"], ["M2/CEDR composite", "two_module_v1，迁移完成", "1dd9e8b92991eb9ae06cfac631be2496f63f877d02e084a3b0a543ee721206a9", "未单独记录逐样本差异"]]),
        "",
        "COCO-LT checkpoint 已完成兼容封装并记录源哈希；当前归档没有另存一份与 VOC 同格式的真实样本逐张等价 JSON，因此只能确认迁移与可追踪性，不能把它表述成第二份数值回归通过。",
        "",
        "## 13. 最终结论",
        "",
        "- M1：VOC-LT 与 COCO-LT 均有效。场景替换波动在三种面积下稳定下降，目标核心削弱效应增强，并明显大于随机/反注意力对照。",
        "- M2：总体有效但存在操作点限制。两套数据的 Tail mAP、去共现 AP、Head在场 AUPRC、低 FPR 召回和校准总体改善；空间定向去重叠和记忆 margin 提供内部机制证据。但 FPR@R60、部分路由对照和类别级相关性不是全正向，不能声称完全消除 Head 替代。",
        "- Full：两套数据共同支持论文问题。证据表明模型既降低外围共享场景替代，也改善区域内共现混合与 Head-only 困难负样本下的 Tail 排序稳定性。",
        "",
        "## 14. 结果来源与科研诚信边界",
        "",
        f"- 两模块最新结果：`{two_root}`",
        f"- VOC/COCO 完整证据数据：`{root / 'runs/evidence_validation_v2'}`",
        f"- 本汇总输出：`{output}`",
        "- 本轮只进行确定性表格整理和类别配对统计，没有 GPU 推理、训练或测试集调参。",
        "- VOC 最新两模块主表和归档 evidence-v2 的完整机制表来自两套已锁定历史 checkpoint 口径；本文档在各节分开引用，不把二者数值混成同一模型。历史 checkpoint 存在 post-hoc official-test candidate selection，正式投稿仍需 validation-locked 多种子复验。",
    ]

    summary_path = output / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats_path = output / "paired_class_statistics.json"
    all_stats = {dataset: {"M1": m1_stats[dataset], "M2": stats[dataset]} for dataset in ("voc", "coco")}
    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sources = [
        root / "final_results/summary.json",
        two_root / "ablation/ablation_two_module.json",
    ]
    for bundle in bundles.values():
        for name in ("overall_metrics.csv", "subset_metrics.csv", "lci_intervention_metrics.csv", "ced_overlap_summary.csv", "ced_routing_control_metrics.csv", "ced_branch_metrics.csv", "tmr_margin_summary.csv", "tmr_ablation.csv", "operating_point_metrics.csv", "calibration_metrics.csv", "correlation_metrics.csv", "run_manifest.json"):
            sources.append(bundle["base"] / name)
    manifest = {
        "format_version": "two_module_full_tables_v1",
        "created_unix": time.time(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "new_training": False,
        "new_inference": False,
        "new_tuning": False,
        "seed_for_statistics": 42,
        "class_bootstrap_repetitions": 10_000,
        "source_sha256": {str(path): sha256_file(path) for path in sources},
        "report_builder_sha256": sha256_file(root / "tools/build_full_experiment_tables.py"),
        "output_sha256": {str(summary_path): sha256_file(summary_path), str(stats_path): sha256_file(stats_path)},
        "outputs": [str(summary_path), str(stats_path), str(output / "manifest.json")],
        "limitations": [
            "only one archived training seed is available",
            "historical checkpoints include post-hoc official-test candidate selection",
            "VOC latest two-module main result and evidence-v2 mechanism bundle are separately locked checkpoint protocols",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "stats": str(stats_path), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--two-module-root", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
