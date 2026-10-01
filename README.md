# smFRET / colocalization time-trace pipeline

`smfret_hmm_pipeline.py` (jupytext percent script) / `smfret_hmm_pipeline.ipynb` (executed notebook):
bleach/blink exclusion -> leakage & gamma -> normalisation -> global 5-state HMM -> per-trace HMM ->
acceptor-bound calls -> 1- vs 2-component dwell-time fits -> FRET histogram + Gaussian mixture.
Helper code is in `fretlib.py` (tests: `pytest tests`).

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace smfret_hmm_pipeline.ipynb
# other data / settings:
papermill smfret_hmm_pipeline.ipynb out.ipynb -p DATA_DIR path/to/traces -p FRAME_TIME_S 0.1
```
Input: iSMS `Traces_*_pair<N>.txt` files (`Dem-Dexc`, `Aem-Dexc`). Examples are in `data/example/`.
Outputs (CSV tables, figures, `parameters_used.json`) go to `results/`. Edit the parameters cell to change thresholds.
