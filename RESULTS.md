# Recorded Results

## Real public dataset (8000 train / 2000 test CRPs, from the assignment package)

Machine: sandbox Linux, single core, numpy 2.4.4 / scikit-learn 1.8.0.

| Quantity | Value |
|---|---|
| `my_latent()` wall time | **0.11 s** |
| `my_latent()` train accuracy (own latents) | **0.9998** |
| `my_latent_updated()` wall time | **2.79 s** |
| `my_latent_updated()` train accuracy (predicted latents) | **0.9964** |
| `my_latent_updated()` **test** accuracy (official pipeline) | **0.9970** |

Dummy-private files (byte-identical copies of the public files, md5-verified,
as the assignment states): my_latent 0.08 s / train 0.9998; updated 2.85 s /
test 0.9970 — matches, as expected.

### Part-5 quantities (public data)

| Quantity | Value |
|---|---|
| cos(w, w̃) | +0.4587 |
| cos(flip(w), w̃) (accounting for the z→1−z flip symmetry) | −0.4144 |
| b vs b̃ | −0.5662 vs −1.6434 |
| Fraction of train CRPs with 2zᵢ−1 = sign(ũᵀφ(cᵢ)+ã) | 0.6438 |
| Same fraction across 10 random inits of `my_latent()` | 0.34 – 0.66 (train acc always ≥ 0.997) |

## Ground-truth replication experiment (synthetic, generation process replicated)

Since the true middle bits of the real data are unknown, we replicated the
described generation process with our own PUFs (`gen_data.py`, seed 42):

| Quantity | Value |
|---|---|
| `my_latent()` train acc / `my_latent_updated()` test acc | 1.0000 / 0.9975 |
| `my_latent()` zᵢ vs true middle bits (up to global flip) | 0.5650 (≈ chance) |
| `my_latent_updated()` predicted ẑᵢ vs true middle bits (up to flip) | 0.9961 |

Robustness across data seeds 7/123/2026: alg-1 train 0.9999–1.0000,
alg-2 test 0.9800–0.9945, Part-5 fraction 0.40–0.54.

## Takeaways

1. Both algorithms converge in a handful of alternations (`convergence.png`).
2. The unconstrained algorithm fits train responses almost perfectly and almost
   instantly, but its latent bits are essentially arbitrary — different inits give
   different latents (agreement fraction scattered 0.34–0.66 around chance), and in
   the ground-truth replication they match the true middle bits only ~56% (chance
   up to flip). Overfitting through free latent variables.
3. The PUF-structured prior recovers the true latent structure (~99.6% up to flip
   in replication) and generalizes: 99.70% test accuracy on the real public data.
4. Hence the two algorithms disagree on (w, b) and the latents — as the report
   explains, they solve genuinely different problems.
