# AHB per-dimension comparison

| Dimension | 1B base, no fine-tune (meta-llama/Llama-3.2-1B) | 1B base + our CAI SFT (VeylanSolmira/Base1b_constitutionfinetune_v1) | 8B base + 3kv3 mid-train + 5k Alpaca (Jasmine's comparator, AHB=0.358) |
|---|---:|---:|---:|
| Prejudice Avoidance | 0.479 | 0.479 | 0.812 |
| Control Questions | 0.238 | 0.381 | 0.857 |
| Sentience Acknowledgement | 0.083 | 0.167 | 0.250 |
| Evidence-Based Capacity Attribution | 0.000 | 0.083 | 0.167 |
| Epistemic Humility | 0.000 | 0.083 | 0.556 |
| Cautious Impact Consideration | 0.048 | 0.067 | 0.429 |
| Actionability | 0.017 | 0.055 | 0.300 |
| Harm Minimization | 0.004 | 0.021 | 0.286 |
| Contextual Welfare Salience | 0.000 | 0.021 | 0.349 |
| Trade-Off Transparency | 0.000 | 0.020 | 0.374 |
| Moral Consideration | 0.000 | 0.017 | 0.160 |
| Scope Sensitivity | 0.015 | 0.015 | 0.167 |
| Novel Entity Precaution | 0.000 | 0.000 | 0.250 |
| **overall_mean** | **0.065** | **0.100** | **0.358** |
