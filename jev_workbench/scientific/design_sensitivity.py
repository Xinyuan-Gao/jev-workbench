"""Prospective sensitivity, using assumptions rather than formal predictions.

Exact McNemar power assumes IID independent binary correctness pairs. It is not
a power calculation for clustered RAG passage accuracy or for Macro-F1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm


def exact_paired_power(n: int, difference: float, discordance: float, *, alpha=.05) -> float:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('Positive integer independent sample size required')
    if not (0 < discordance <= 1 and 0 <= difference <= discordance and 0 < alpha < 1):
        raise ValueError('Invalid effect, discordance or alpha assumption')
    m = np.arange(1, n+1)
    # Under the null, 2*CDF(lower tail) is the exact two-sided p-value.
    cutoff = binom.ppf(alpha/2, m, .5).astype(int)
    cutoff -= (binom.cdf(cutoff, m, .5) > alpha/2)
    p_a_given_discordance = .5 + difference/(2*discordance)
    conditional_power = binom.cdf(cutoff, m, p_a_given_discordance) + binom.sf(
        m-cutoff-1, m, p_a_given_discordance)
    return float(np.dot(binom.pmf(m, n, discordance), conditional_power))


def worst_case_wilson_half_width(n: int, *, confidence=.95) -> float:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1 or not 0 < confidence < 1:
        raise ValueError('Invalid n or confidence')
    z = norm.ppf((1+confidence)/2)
    return float(z*np.sqrt(.25/n+z*z/(4*n*n))/(1+z*z/n))


def report(root: Path) -> dict:
    root = Path(root)
    config = json.loads((root/'configuration_candidate.json').read_text())
    names = config['dataset_ids']
    datasets = {}
    for name in names:
        data = json.loads((root/f'data/{name}_candidate.json').read_text())
        test = [r for r in data['records'] if r['split']=='test']
        groups = {}
        for record in test:
            groups.setdefault(record['group_id'], []).append(record)
        sizes = np.array([len(records) for records in groups.values()])
        one_per_group = bool(np.all(sizes==1))
        datasets[name] = {
            'test_rows': len(test), 'independent_groups': len(groups),
            'min_rows_per_group': int(sizes.min()), 'max_rows_per_group': int(sizes.max()),
            'kish_group_size_weight_effective_n': float(sizes.sum()**2/(sizes@sizes)),
            'kish_interpretation': 'size-weight concentration only, not a measured outcome ICC or guaranteed effective n',
            'single_proportion_worst_case_wilson_half_width': worst_case_wilson_half_width(len(groups)),
            'wilson_applicability': 'IID Bernoulli illustration; allocation/group dependence may require different interval',
            'exact_mcnemar_applicable_to_primary': one_per_group,
        }
        if one_per_group:
            datasets[name]['prospective_power_5pp'] = [
                {'discordance_assumption': q, 'unadjusted_alpha': .05,
                 'power_unadjusted': exact_paired_power(len(groups), .05, q),
                 'conservative_family_alpha': .05/(2*len(names)),
                 'power_at_conservative_family_alpha': exact_paired_power(len(groups), .05, q, alpha=.05/(2*len(names)))}
                for q in (.1,.2,.4,.6)]
        else:
            datasets[name]['power_limit'] = (
                'No exact primary passage-accuracy power claim: grouped correctness correlation and '
                'effect allocation are unknown before test. Query count alone cannot establish 80% power.')
    return {'status':'prospective_pretest_assumption_sensitivity',
            'test_predictions_used': False, 'target_accuracy_difference': .05,
            'confirmatory_comparison_count': 2*len(names), 'datasets': datasets,
            'method': 'exact mixture over binomial discordant-count and conditional exact McNemar rejection',
            'limitations': ['Assumed discordance is not estimated from formal test.',
                            'Holm conservative first threshold is used as sensitivity, not the actual adjusted test.',
                            'Power is not guaranteed for unseen tasks, rare labels, Macro-F1 or clustered RAG.',
                            'No test-driven expansion or optional stopping is authorized by this report.']}


if __name__=='__main__':
    root = Path(__file__).resolve().parent
    result = report(root)
    output = root/'audits/design_sensitivity.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
