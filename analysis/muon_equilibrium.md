# Muon equilibrium AUS: where the 13% gap comes from (2026-10-30)

Open item in PITFALLS F: Kosson's √(2ηλ(1−β)/(1+β)) with β = 0.95 is 13% below the measured Muon AUS.
Scripts: `muon_equilibrium_logs.py` (decomposes the logged runs), `muon_equilibrium_sim.py` (CPU toy).

## Formula
Exact per-step norm identity, with e = ELR, ε = ηλ, k = −cos(W,U)/e, δ = per-step change of ln‖W‖²:

$$ e^2\,\bigl(1 + 2(1-\varepsilon)k\bigr) = 2\varepsilon - \varepsilon^2 + \delta $$

If the gradient is orthogonal to W (scale invariance) and carries no information from before the
momentum window, ⟨W_t, U_t⟩ = −η Σ_{d≥1} (1−ε)^{d−1} ⟨U_{t−d}, U_t⟩, so k = Σ_{d≥1} C(d), with C(d)
the update autocorrelation ⟨U_t,U_{t−d}⟩/‖U‖². To first order in ε:

$$ \mathrm{AUS}_{\mathrm{eq}} \approx \sqrt{\frac{2\eta\lambda + \delta}{1 + 2k}} $$

For a linear map U ∝ Σ a_j g_{t−j} with white-noise gradients, 1 + 2k = (Σa)²/Σa².
Heavy ball: (1+β)/(1−β) = 39 (Kosson). Nesterov as in `train.py` (a_0 = 1+β, a_j = β^{j+1}): 32.9, k = 15.95.

## Decomposition of the measured gap (last 10% of const + WD 0.1 runs, block matrices)
| run | measured AUS | Kosson HB | Nesterov | Nesterov + δ | measured k (model 15.95) | δ/2ε |
|---|---|---|---|---|---|---|
| Muon 6e-4 s0 | 0.002015 | 0.001754 (−13.0%) | 0.00191 (−5.2%) | 0.001994 (−1.0%) | 15.70 | 0.090 |
| Muon 6e-4 s1 | 0.002007 | 0.001754 (−12.6%) | 0.00191 (−4.8%) | 0.001992 (−0.7%) | 15.78 | 0.088 |
| Muon 2.4e-3 | 0.004072 | 0.003508 (−13.8%) | 0.003819 (−6.2%) | 0.003821 (−6.2%) | 13.96 | 0.0006 |

With the measured k and δ the identity closes to 0.2–0.3% (prediction 0.002009 / 0.00408).
- **Nesterov weights explain ~8 of the 13 points** in every run: √(39/32.9) = 1.089.
- **6e-4 (1.5 timescales):** the rest is non-equilibrium. ‖W‖² still grows by 9% of the decay rate, which lifts the ELR by √1.09 ≈ 4%. k is within 2% of the model.
- **2.4e-3 (6 timescales, δ ≈ 0):** the rest is a weaker lean, k = 14.0 instead of 15.95: √(32.9/28.9) = 1.065.
- Second-order terms (ε², the (1−ε) factors, WD inside the momentum window) are < 0.5%.
- AdamW shows the same lr dependence: k = 9.4 (6e-4) and 7.9 (2.4e-3) against the model 9.

## Why k is below the linear model (not settled)
CPU toy (gradient ⟂ W, white noise, train.py's Nesterov + NS5 and scaling, ηλ = 1e-3):
| map | 128×384 | 128×128 |
|---|---|---|
| linear (U ∝ M) | 15.66 | 15.65 |
| exact polar UVᵀ | 14.59 | 11.74 |
| NS5 | 13.49 | 11.00 |

- Orthogonalization decorrelates successive updates (C(d) falls below the linear value), so it lowers k. The size depends on the matrix shape and on the gradient spectrum; for white noise and square matrices it is −31%.
- The real runs lose only 2% (6e-4) and 12% (2.4e-3), and q/k reach k ≈ 19 at 6e-4 (above the linear model), v/o 8–11. So real gradients carry positive temporal correlation that partly cancels the NS5 effect, and that correlation falls as η rises. A restoring force in the toy (h > 0) reproduces the drop of k with η (13.7 → 13.5 at h = 0; 13.0 → 12.4 at h = 2), and AdamW shows the same drop, so the η-dependent part is not Muon-specific.
- Consequence: for Muon, k cannot be computed from β alone. Use √((2ηλ+δ)/(1+2k)) with k measured (or k from logged update autocorrelation C(d)). β-only Nesterov prediction is 5–6% low at equilibrium.

## Open
- Log ⟨U_t, U_{t−d}⟩ for d ≤ 40 to test k = Σ C(d) directly and split NS5 decorrelation from gradient correlation.
- Non-scale-invariance (B6) enters through ⟨W_{t−D}, U_t⟩ ≠ 0; not separated here.
