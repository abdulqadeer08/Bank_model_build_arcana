"""
Final verification pass before declaring the notebook ready to send.
Run:  python final_verify.py
"""
import json, re, sys
import nbformat

NB = 'Bank_Cash_Optimization_Workflow.ipynb'
VALUES = r"C:\Users\Raza\AppData\Local\Temp\claude\d--bank\24934af7-bbef-4554-8e8d-25719e71ab0d\scratchpad\nbrun\notebook_values.json"

ok = True
def check(name, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond: ok = False
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    return cond

print("=" * 70)
print("1. NOTEBOOK STRUCTURE")
print("=" * 70)
nb = nbformat.read(NB, as_version=4)
try:
    nbformat.validate(nb)
    check("nbformat validate", True)
except Exception as e:
    check("nbformat validate", False, str(e)[:200])

bad = []
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code': continue
    s = c.source
    if s.lstrip().startswith('%') or '!pip' in s: continue
    try: compile(s, f'<{i}>', 'exec')
    except SyntaxError as e: bad.append((i, str(e)[:60]))
check("all code cells compile", not bad, str(bad))

no_out = [i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and not c.outputs]
allowed_empty = {5, 13, 30, 73}  # %%capture / no-print-by-design cells; verified by hand
unexpected_empty = [i for i in no_out if i not in allowed_empty]
check("no unexpected empty code cells", not unexpected_empty, str(unexpected_empty))

n_errors = sum(1 for c in nb.cells if c.cell_type == 'code'
              for o in c.get('outputs', []) if o.get('output_type') == 'error')
check("zero error outputs stored", n_errors == 0, f"{n_errors} error outputs found")

src_all = "\n".join(c.source for c in nb.cells if c.cell_type == 'code')
check("CV_FOLDS defined", "CV_FOLDS = rolling_origin_folds" in src_all)
check("build_ensemble defined", "def build_ensemble" in src_all)
check("no leftover date-tuple CV_FOLDS unpack",
      not re.search(r'for _?\w*,?\s*\(_?a,\s*_?b\)\s+in enumerate\(CV_FOLDS', src_all))
check("Section 9.3 uses W_DEBIT not stale w_xgb",
      "W_DEBIT * xgb_model.predict(x_df)" in src_all)
check("forecast lower bound clamped at 0",
      "max(0.0, pred * (1 - pct))" in src_all)

print()
print("=" * 70)
print("2. EXECUTED VALUES SANITY")
print("=" * 70)
try:
    v = json.load(open(VALUES, encoding='utf-8'))
    check("notebook_values.json loaded", True)
except FileNotFoundError:
    print("  [FAIL] notebook_values.json not found — run execute_notebook_full.py first")
    sys.exit(1)

s70 = v['s70']; sp = v['split']
check("hold-out is ~20% of data", 15 <= sp['test_pct'] <= 25, f"{sp['test_pct']}%")
check("hold-out covers all 15 branches",
      all(r['n'] >= 2 for r in s70['holdout_branch']) and len(s70['holdout_branch']) == 15,
      f"{len(s70['holdout_branch'])} branches")
check("every branch in every CV fold",
      all(r['Folds'] == sp['n_folds'] for r in s70['branch']))
check("CV R2 in plausible range", 0.3 <= s70['r2_mean'] <= 0.8, f"{s70['r2_mean']}")
check("CV WMAPE in plausible range", 20 <= s70['wm_mean'] <= 60, f"{s70['wm_mean']}")
check("no NaN in per-branch WMAPE",
      all(r['WMAPE_mean'] == r['WMAPE_mean'] for r in s70['branch']))  # NaN != NaN

b202 = next((r for r in s70['branch'] if r['branch'] == 202), None)
b1739 = next((r for r in s70['branch'] if r['branch'] == 1739), None)
check("branch 202 present", b202 is not None)
check("branch 1739 present", b1739 is not None)
if b202 and b1739:
    check("202 has lower WMAPE than 1739 (matches review narrative)",
          b202['WMAPE_mean'] < b1739['WMAPE_mean'],
          f"202={b202['WMAPE_mean']:.1f} 1739={b1739['WMAPE_mean']:.1f}")

check("specialist experiment ran on all branches", len(s70['spec']) == 15)
check("lobo ran on all branches", len(s70['lobo']) == 15)

print()
print("=" * 70)
print("3. NARRATIVE CONSISTENCY (post regenerate_narrative.py)")
print("=" * 70)
nb2 = nbformat.read(NB, as_version=4)
full_text = "\n".join(c.source for c in nb2.cells if c.cell_type == 'markdown')
stale_patterns = ['39.52%', '826.2%', '8.825M']
# NOTE: 0.5687 / 0.0581 / '164 rows, 1.0%' are deliberately still present, verified by hand:
# once as a verbatim quote of the supervisor's original review text (cell 51, in quotation
# marks), and once in the Step-1-result cell explicitly contrasted against the fix ("...164
# rows, 1.0% of the data, instead of 20%. Both are now fixed."). Flagging those would punish
# correct historical framing, not catch a real defect.
stale_hits = [p for p in stale_patterns if p in full_text]
check("no stale pre-fix numbers presented as current", not stale_hits, str(stale_hits))
check("corrected R2 appears in markdown", f"{s70['r2_mean']:.3f}"[:5] in full_text or f"{s70['r2_mean']:.4f}" in full_text)
check("six defects mentioned", "Six defects" in full_text or "six defects" in full_text.lower())

print()
print("=" * 70)
print(f"OVERALL: {'READY' if ok else 'NOT READY — see FAILs above'}")
print("=" * 70)
sys.exit(0 if ok else 1)
