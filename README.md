# CS771 Assignment 2 — Bit-in-the-Middle Attack

Solution to CS771 (IIT Kanpur, 2025-26-III) Assignment 2: learning a 17-bit arbiter
PUF from CRPs whose middle challenge bit was tampered with and then erased, posed as
a latent-variable MLE problem solved by alternating optimization (hard-EM /
winner-take-all).

## Repository contents

| File | What it is |
|---|---|
| `submit.py` | The code deliverable. Implements `my_latent()` (Part 2) and `my_latent_updated()` (Part 4) using only `numpy` + `sklearn` linear models, as the assignment allows. |
| `report.tex` / `report.pdf` | The report deliverable (Parts 1, 3, 5): full derivations of both alternating optimization algorithms and the agreement analysis. NeurIPS 2025 preprint format. |
| `submit.zip` | Password-protected ZIP containing only `submit.py`, per the submission instructions. Password is in `zip_password.txt` (10 alphanumeric chars). |
| `gen_data.py` | Synthetic data generator that follows the assignment's exact generation process (hidden 16-bit PUF → middle bit → insert → 17-bit PUF → response → erase). Used because the official package is on IITK servers. |
| `evaluate.py` | Local evaluation harness mirroring the official judge (train accuracy for `my_latent`, test accuracy via predicted middle bits for `my_latent_updated`) plus the Part-5 quantities. |
| `make_plots.py` | Generates `convergence.png` used in the report. |
| `train.dat`, `test.dat` | The **real public dataset** from the assignment package (copied from `public_trn.txt` / `public_tst.txt`). `private_trn.txt` / `private_tst.txt` are the dummy-private files (byte-identical to public, md5-verified). |
| `RESULTS.md` | Recorded results. |

## How to run

```bash
python3 gen_data.py 42     # skip this if you drop in the real train.dat / test.dat
python3 evaluate.py        # runs both methods, prints timings + accuracies, saves results.npz
python3 make_plots.py      # regenerates convergence.png for the report
pdflatex report.tex && pdflatex report.tex
```

## Data note

The repo now uses the **real public dataset** from the assignment package
(`train.dat` = `public_trn.txt`, `test.dat` = `public_tst.txt`); all numbers in
the report, README and RESULTS.md were produced on it. `gen_data.py` remains only
for the ground-truth replication experiment cited in Part 5 of the report. If you
ever adapt this for actual use: paste the bodies of `my_latent()` /
`my_latent_updated()` into the official `submit_template.py` (keeping its
non-editable regions intact) and validate on the Google Colab script.

## Method in one paragraph

Both methods are hard-EM. `my_latent()`: with a uniform latent prior, the z-step
picks, per CRP, the middle bit whose score best explains the observed response, and
the (w,b)-step is a plain logistic regression on the completed 17-bit embeddings —
alternate to convergence with random restarts. `my_latent_updated()`: the latent
prior is itself a 16-bit arbiter PUF (u,a), so the z-step scores each candidate bit
by the joint likelihood (response term + prior term), and there are two logistic
regressions per alternation — responses on completed 17-bit features for (w,b), and
current latent estimates on 16-bit features for (u,a). A key trick: flipping the
inserted middle bit just negates the first 9 coordinates of the 17-dim arbiter-PUF
embedding, so both candidate feature vectors per CRP are precomputed once.

## Results (real public dataset; see RESULTS.md)

| Method | Time | Train acc | Test acc |
|---|---|---|---|
| `my_latent()` | 0.11 s | 0.9998 | — |
| `my_latent_updated()` | 2.8 s | 0.9964 | 0.9970 |

Part 5: the two algorithms **disagree** — cos(w, w̃) ≈ +0.46, b ≠ b̃, and the
fraction with 2zᵢ−1 = sign(ũ·φ(cᵢ)+ã) is ≈ 0.64 in the main run and scatters over 0.34–0.66 across inits (≈ chance). The unconstrained latents
overfit the train responses (in a ground-truth replication their bits match the true middle bits only ~56%, chance
up to the global flip symmetry), whereas the PUF-structured prior recovers the true
middle bits ~99.6% of the time and generalizes. Full explanation in the report.

