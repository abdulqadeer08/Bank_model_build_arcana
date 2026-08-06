"""
metrics_utils.py — Shared evaluation metrics for the Bank Cash Optimization project.

Provides reusable MAPE, SMAPE, and WMAPE functions with proper edge-case handling.
WMAPE is the supervisor-recommended metric for presenting to the bank, as it is
less distorted by low-volume / near-zero-demand days than standard MAPE.
"""

import numpy as np


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (standard).
    
    Excludes rows where y_true == 0 to avoid division by zero.
    Returns percentage value (e.g. 132.7 means 132.7%).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error.
    
    Formula: mean( |actual - predicted| / ((|actual| + |predicted|) / 2) ) * 100
    
    Edge case: rows where BOTH actual AND predicted are exactly 0 are excluded
    (would cause 0/0). The number of excluded rows is tracked internally.
    
    Returns: (smape_value, n_excluded) tuple.
      - smape_value: percentage (e.g. 45.2 means 45.2%)
      - n_excluded: count of rows excluded due to both being zero
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # Identify rows where both are zero (0/0 edge case)
    both_zero = (y_true == 0) & (y_pred == 0)
    n_excluded = int(both_zero.sum())
    
    # Keep only valid rows
    valid = ~both_zero
    if valid.sum() == 0:
        return np.nan, n_excluded
    
    result = np.mean(np.abs(y_true[valid] - y_pred[valid]) / denominator[valid]) * 100
    return float(result), n_excluded


def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error (aggregate formula).
    
    Formula: ( sum(|actual - predicted|) / sum(|actual|) ) * 100
    
    This is calculated as ONE ratio of sums across the full dataset (or per branch),
    NOT as an average of individual percentage errors. This is precisely why it's more
    stable than MAPE — large-volume days naturally dominate the denominator, preventing
    near-zero-demand days from inflating the metric.
    
    Returns percentage value (e.g. 28.5 means 28.5%).
    Returns NaN if sum of |actual| is zero.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    sum_abs_actual = np.sum(np.abs(y_true))
    if sum_abs_actual == 0:
        return np.nan
    
    return (np.sum(np.abs(y_true - y_pred)) / sum_abs_actual) * 100


def run_sanity_check():
    """Hand-calculated sanity check with known simple examples.
    
    Verifies SMAPE, WMAPE, and MAPE compute correctly before applying to the full dataset.
    """
    print("=" * 70)
    print("  SMAPE / WMAPE / MAPE -- SANITY CHECK (Hand-Calculated Examples)")
    print("=" * 70)
    
    # -- Example 1: actual=100, predicted=110 ------------------------------
    a1, p1 = np.array([100.0]), np.array([110.0])
    m1 = mape(a1, p1)
    s1, exc1 = smape(a1, p1)
    w1 = wmape(a1, p1)
    # Hand-calc:
    #   MAPE  = |100-110|/100 * 100 = 10.0%
    #   SMAPE = |100-110| / ((100+110)/2) * 100 = 10/105 * 100 = 9.524%
    #   WMAPE = 10/100 * 100 = 10.0%
    print(f"\n  Example 1: actual=100, predicted=110")
    print(f"    MAPE  = {m1:.3f}%   (expected: 10.000%)")
    print(f"    SMAPE = {s1:.3f}%   (expected:  9.524%)")
    print(f"    WMAPE = {w1:.3f}%   (expected: 10.000%)")
    assert abs(m1 - 10.0) < 0.01, f"MAPE sanity failed: {m1}"
    assert abs(s1 - 9.524) < 0.01, f"SMAPE sanity failed: {s1}"
    assert abs(w1 - 10.0) < 0.01, f"WMAPE sanity failed: {w1}"
    print("    [OK] PASS")
    
    # -- Example 2: actual=2, predicted=5 ----------------------------------
    a2, p2 = np.array([2.0]), np.array([5.0])
    m2 = mape(a2, p2)
    s2, exc2 = smape(a2, p2)
    w2 = wmape(a2, p2)
    # Hand-calc:
    #   MAPE  = |2-5|/2 * 100 = 150.0%
    #   SMAPE = |2-5| / ((2+5)/2) * 100 = 3/3.5 * 100 = 85.714%
    #   WMAPE = 3/2 * 100 = 150.0%
    print(f"\n  Example 2: actual=2, predicted=5")
    print(f"    MAPE  = {m2:.3f}%   (expected: 150.000%)")
    print(f"    SMAPE = {s2:.3f}%   (expected:  85.714%)")
    print(f"    WMAPE = {w2:.3f}%   (expected: 150.000%)")
    assert abs(m2 - 150.0) < 0.01, f"MAPE sanity failed: {m2}"
    assert abs(s2 - 85.714) < 0.01, f"SMAPE sanity failed: {s2}"
    assert abs(w2 - 150.0) < 0.01, f"WMAPE sanity failed: {w2}"
    print("    [OK] PASS")
    
    # -- Example 3: actual=0, predicted=3 (MAPE excludes, SMAPE handles) --
    a3, p3 = np.array([0.0]), np.array([3.0])
    m3 = mape(a3, p3)       # NaN — no valid rows (actual=0 excluded)
    s3, exc3 = smape(a3, p3)
    w3 = wmape(a3, p3)
    # Hand-calc:
    #   MAPE  = NaN (actual=0 excluded)
    #   SMAPE = |0-3| / ((0+3)/2) * 100 = 3/1.5 * 100 = 200.0%
    #   WMAPE = 3/0 = NaN (sum of |actual| = 0)
    print(f"\n  Example 3: actual=0, predicted=3 (edge case)")
    print(f"    MAPE  = {m3}       (expected: nan -- actual=0 excluded)")
    print(f"    SMAPE = {s3:.3f}%   (expected: 200.000%)")
    print(f"    WMAPE = {w3}       (expected: nan -- sum(|actual|)=0)")
    assert np.isnan(m3), f"MAPE should be NaN: {m3}"
    assert abs(s3 - 200.0) < 0.01, f"SMAPE sanity failed: {s3}"
    assert np.isnan(w3), f"WMAPE should be NaN: {w3}"
    print("    [OK] PASS")
    
    # -- Example 4: actual=0, predicted=0 (SMAPE both-zero exclusion) -----
    a4, p4 = np.array([0.0]), np.array([0.0])
    s4, exc4 = smape(a4, p4)
    # Hand-calc: SMAPE = NaN (both zero → excluded), n_excluded = 1
    print(f"\n  Example 4: actual=0, predicted=0 (both-zero edge case)")
    print(f"    SMAPE = {s4}, excluded={exc4}   (expected: nan, excluded=1)")
    assert np.isnan(s4), f"SMAPE should be NaN: {s4}"
    assert exc4 == 1, f"Should exclude 1 row: {exc4}"
    print("    [OK] PASS")
    
    # -- Combined array: tests aggregate WMAPE properly --------------------
    # actual = [100, 2, 50], predicted = [110, 5, 45]
    ac = np.array([100.0, 2.0, 50.0])
    pc = np.array([110.0, 5.0, 45.0])
    mc = mape(ac, pc)
    sc, excc = smape(ac, pc)
    wc = wmape(ac, pc)
    # Hand-calc:
    #   MAPE  = mean(10/100, 3/2, 5/50) * 100 = mean(0.1, 1.5, 0.1) * 100 = 56.667%
    #   SMAPE = mean(10/105, 3/3.5, 5/47.5) * 100
    #         = mean(0.09524, 0.85714, 0.10526) * 100 = 35.255%
    #   WMAPE = (10+3+5) / (100+2+50) * 100 = 18/152 * 100 = 11.842%
    #   ↑ Note how WMAPE (11.8%) is far lower than MAPE (56.7%) because the
    #     near-zero actual=2 row doesn't dominate the aggregate ratio.
    print(f"\n  Combined: actual=[100,2,50], predicted=[110,5,45]")
    print(f"    MAPE  = {mc:.3f}%   (expected: 56.667%)")
    print(f"    SMAPE = {sc:.3f}%   (expected: 35.255%)")
    print(f"    WMAPE = {wc:.3f}%   (expected: 11.842%)")
    print(f"    >> WMAPE ({wc:.1f}%) is far more stable than MAPE ({mc:.1f}%)")
    print(f"       because the near-zero actual=2 row doesn't dominate the aggregate.")
    assert abs(mc - 56.667) < 0.01, f"MAPE sanity failed: {mc}"
    assert abs(sc - 35.255) < 0.01, f"SMAPE sanity failed: {sc}"
    assert abs(wc - 11.842) < 0.01, f"WMAPE sanity failed: {wc}"
    print("    [OK] PASS")
    
    print("\n" + "=" * 70)
    print("  ALL SANITY CHECKS PASSED [OK]")
    print("=" * 70)


if __name__ == '__main__':
    run_sanity_check()
