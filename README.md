# COPD Supervised Learning

Supervised-learning models that classify COPD-related outcomes from multi-omics
data (transcriptomics, proteomics, metabolomics) and clinical variables drawn
from the [COPDGene](http://www.copdgene.org/) study.

The project trains and evaluates a soft-voting ensemble (logistic regression
with elastic net, an MLP, and an SVC) on each individual omics layer and on
the concatenation of all three, predicting:

- **GOLD stage** — binarized to GOLD 0 vs GOLD 2–4 (subjects with GOLD 1 are
  excluded as an intermediate, ambiguous group).
- **Emphysema burden** — binarized at 5 % (`pctEmph < 5` vs `pctEmph >= 5`).

This was a rotation project (Jan–Mar 2022). The code is preserved as-is for
reproducibility; this README documents how to run it.

## Repository layout

```
COPDSupervisedLearning/
├── src/                    # All source code lives here
│   ├── main.py             # End-to-end pipeline: load → align → filter → classify
│   ├── getData.py          # Reads clinical CSV, joins omics on `sid`
│   ├── dataExploration.py  # PCA, log transform, standardization, k-best feature selection
│   ├── neuralNetwork.py    # Voting classifier, RandomForest, XGBoost wrappers
│   ├── plots.py            # ROC, class-probability bars, prediction histograms
│   ├── geneSetEnrichment.py# Stub for GSEA via gseapy
│   └── test.py             # Earlier scratch driver (broken, kept for reference)
│
├── dataInfo.txt            # Notes on which raw COPDGene files this expects
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore              # Excludes data folders, venv, archive/, IDE files
├── archive/                # Old slide decks, presentation figures (gitignored)
│
├── Clinical Variables/     # COPDGene visit-level CSV   (gitignored, restricted)
├── Metabolomics/           # COPDGene P2 metabolite CSV (gitignored, restricted)
├── Proteomics/             # COPDGene SomaScan 5K text  (gitignored, restricted)
├── Transcriptomics/        # Raw RNA-seq counts         (gitignored, restricted)
└── preprocessedRNAseq/     # VST-normalized RNA-seq     (gitignored, restricted)
```

## Data

The pipeline expects the following files at the paths shown (paths are
relative to the repository root, since `src/main.py` is run from the root —
see *Usage* below). The data is **not** included in this repository —
COPDGene is access-controlled and must be requested from the
[COPDGene study](http://www.copdgene.org/).

| Modality                       | Expected path                                                              |
| ------------------------------ | -------------------------------------------------------------------------- |
| Clinical                       | `Clinical Variables/COPDGene_P1P2P3_25SEP2020_VisitLevel.csv`              |
| Metabolomics                   | `Metabolomics/COPDGene_P2_LT20missing_knnImpute_metabolites_20211021.csv`  |
| Proteomics                     | `Proteomics/COPDGeneSoma_SMP_5K_P2_16Jun20.txt` (tab-separated)            |
| Transcriptomics (preprocessed) | `preprocessedRNAseq/X_gene_vst_3270subjects_010822.csv`                    |
| Transcriptomics (raw counts)   | `Transcriptomics/2021-09-28 original/counts_raw.tsv`                       |

Subjects are joined across modalities via the `sid` column. Only Phase-2
visits (`visitnum == 2`) are kept. Clinical features extracted: `gender`,
`smoking_status`, `cohort`, `BMI`, `finalGold`, `pctEmph`.

See `dataInfo.txt` for the original notes on file selection.

## Pipeline (`src/main.py`)

1. **Load** — read clinical, metabolomic, proteomic, and transcriptomic tables.
2. **Align** — inner-join all four on `sid` so every retained subject has
   measurements in every modality.
3. **Clean** — drop rows containing NaN; transcriptomics is transposed so
   subjects become rows.
4. **Transform** — `log(1+x)` on proteomics and metabolomics, then per-feature
   standardization. Features whose coefficient of variation falls below `0.025`
   are filtered out.
5. **Label** — derive binary GOLD (0 vs 2–4, dropping GOLD 1) and binary
   emphysema (< 5 % vs ≥ 5 %).
6. **Feature select** — `SelectKBest` with χ² on each modality
   (`k = 5000` transcriptomics, `k = 1000` proteomics, `k = 500` metabolomics).
7. **Classify** — soft-voting ensemble of:
   - `LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.75)`
   - `MLPClassifier(hidden_layer_sizes=(5, 2), solver='lbfgs')`
   - `SVC(probability=True)`

   Evaluated with 5-fold cross-validation; outputs ROC curves, confusion
   matrices, and per-classifier accuracy bars.

## Setup

This code targets Python 3.9. Because it uses the legacy
`sklearn.metrics.plot_confusion_matrix` and `metrics.plot_roc_curve`
APIs (removed in scikit-learn 1.2), the requirements pin scikit-learn `<1.2`.

```bash
git clone <your-fork-url>
cd COPDSupervisedLearning
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the COPDGene data files in the expected paths (see *Data* section above).

## Usage

Run the full pipeline **from the repository root** so that the relative data
paths inside `main.py` resolve correctly. Python automatically adds the
script's directory to `sys.path`, so the sibling-module imports inside `src/`
(e.g. `import dataExploration, getData`) just work:

```bash
python src/main.py
```

`main.py` opens matplotlib windows for each plot (PCA, ROC, confusion
matrices, etc.). Close each window to advance to the next.

`src/test.py` is an earlier scratch driver kept for reference; it is **not
runnable as-is** (it references symbols never defined in the file and uses
function signatures that have since changed). See its module docstring for
details.

## Module reference (`src/`)

- `getData.getClinicalData(df)` — selects the Phase-2 clinical columns used
  downstream and filters to `visitnum == 2`.
- `getData.getUnionofPatients(a, b)` — inner-merges two tables on `sid`.
- `dataExploration.standardizeData(X, cutoff, title)` — standardizes columns
  and keeps those whose coefficient of variation exceeds `cutoff`.
- `dataExploration.uniFeatureSelec(X, y, k)` — `SelectKBest(chi2, k)`.
- `dataExploration.logTransform(X)` — `log(1+x)`.
- `dataExploration.getRidofGold1(clinGold, y)` — drops the GOLD-1 stratum.
- `neuralNetwork.votingClassifier(X, y, y_orig, title)` — fits the soft-voting
  ensemble and renders ROC, confusion matrix, and class-probability plots.
- `plots.kROC` / `plots.classProbs` / `plots.predGold` / `plots.predEmph` —
  evaluation visualizations.

Each module has a top-of-file docstring summarizing its public surface, and
every function carries a docstring describing its inputs, outputs, and role
in the pipeline.

## Author

Katherina Cortes (rotation project, Jan – Mar 2022).
