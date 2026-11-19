# ADME-related chemistry project: aqueous solubility prediction

## Case study

**Question.** Can basic molecular descriptors estimate a compound's measured aqueous solubility? Solubility is one physicochemical property relevant to early ADME screening; this model does not predict the other ADME endpoints.

**Data.** The [DeepChem MoleculeNet Delaney (ESOL) dataset](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html) contains 1,128 compounds with measured log solubility in mol/L and molecular structures. The file is downloaded from DeepChem's public data bucket; the original dataset is described by Delaney (2004), [DOI 10.1021/ci034243x](https://doi.org/10.1021/ci034243x). The exact file hash and retrieval time are recorded in `data/data_provenance.json`.

**Cleaning and EDA.** The analysis parses the measured target and six numeric descriptors, checks missing values and exact duplicates, and excludes the provided `ESOL predicted log solubility` column so the prediction cannot leak into the features. All 1,128 rows were complete and no exact duplicates were removed. The EDA chart shows the target distribution and descriptor correlations.

**Model.** A fixed 80/20 random train/test split (seed 42) compares a mean baseline, standardized ridge regression, and a random forest tuned with five-fold cross-validation on the training split. The inputs are minimum degree, molecular weight, hydrogen-bond donors, ring count, rotatable bonds, and polar surface area.

**Holdout results.**

| Model | MAE (logS) | RMSE (logS) | R² |
|---|---:|---:|---:|
| Mean baseline | 1.732 | 2.174 | -0.000 |
| Ridge regression | 0.892 | 1.199 | 0.696 |
| Tuned random forest | **0.596** | **0.882** | **0.836** |

On this random holdout, the random forest reduced MAE by about 66% relative to the mean baseline. Five-fold tuning CV MAE was 0.625. Model predictions are in `results/test_predictions.csv`; descriptor importance and plots are also included.

**Limits.** This is a small benchmark dataset, not a clinical or compound-development dataset. A random row split can put closely related chemical scaffolds in both train and test sets, making performance optimistic for genuinely novel chemistry. A scaffold split and external validation are the next steps before making claims about novel molecules.

## Run

```powershell
python -m pip install -r requirements.txt
python analysis.py
```

The script writes cleaning/EDA summaries, metrics, a saved model, predictions, feature importance, and PNG charts into `results/`.
