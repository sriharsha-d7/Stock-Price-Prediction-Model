"""Run the evaluation on synthetic series whose true answer is known; writes results/validation.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stock_forecast.run import evaluate_all  # noqa: E402
from stock_forecast.synthetic import make_prices  # noqa: E402

N = 20000
lines = ["# Harness validation on synthetic series (n = %d days, 15%% test)\n" % N]
for kind, truth in (("random_walk", "returns are iid: NO model can beat naive"),
                    ("ar1", "returns follow AR(1), phi = 0.15: a lag model has small, real skill")):
    table, meta = evaluate_all(make_prices(kind, n=N, seed=7))
    lines.append(f"## {kind}  ({truth})\n")
    cols = ["price_MAPE_%", "within_2.5%_of_actual_%", "directional_accuracy_%", "return_IC_spearman"]
    sk = {k: f"{v['skill']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in meta["skill"].items()}
    t = table[cols].copy()
    t["skill vs naive [95% CI]"] = [sk.get(i, "n/a") for i in t.index]
    lines.append(t.to_markdown(floatfmt=".2f") + "\n")
Path("results").mkdir(exist_ok=True)
Path("results/validation.md").write_text("\n".join(lines))
print("\n".join(lines))
