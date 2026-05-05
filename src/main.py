"""
End-to-end supervised-learning pipeline for COPD subtype classification.

Pipeline (top to bottom in this file):
    1. Load   — clinical, transcriptomics (VST), proteomics, metabolomics.
    2. Align  — keep only Phase-2 subjects (`visitnum == 2`) and intersect
                across all four modalities so every subject has measurements
                in every modality.
    3. Clean  — drop NaN rows; transpose transcriptomics so subjects are rows.
    4. Transform — `log(1+x)` on proteomics + metabolomics, then per-feature
                   standardization with a coefficient-of-variation cutoff
                   (0.025) used to drop low-information features.
    5. Label  — derive two binary outcomes from the clinical table:
                   * `clinGoldStrat`: GOLD 0 vs GOLD 2-4 (GOLD 1 is dropped).
                   * `clinEmphB`:     pctEmph < 5% vs pctEmph >= 5%.
    6. Feature select — `SelectKBest(chi2)` per modality:
                   transcriptomics k=5000 / 1000, proteomics k=1000,
                   metabolomics k=500.
    7. Classify — soft-voting ensemble (LogReg-elasticnet / MLP / SVC),
                  fit on train, scored with 5-fold CV. ROC, confusion matrix,
                  and per-classifier accuracy plots are emitted by
                  `neuralNetwork.votingClassifier` as the script runs.

Run:
    python main.py

Author: Katherina Cortes
Date:   January 7, 2022
"""

import argparse

import numpy as np
import pandas as pd
import dataExploration, getData
import neuralNetwork
from sklearn.model_selection import train_test_split

# Argparse is set up for future CLI flags but no arguments are consumed yet;
# all paths and hyperparameters are hard-coded below.
parser = argparse.ArgumentParser(description='Supervised Learning model for COPD subtypes')

# ---------------------------------------------------------------------------
# 1. Load raw tables
# ---------------------------------------------------------------------------
# Transcriptomics (VST-normalized RNA-seq, genes x subjects)
dfT = pd.read_csv('preprocessedRNAseq/X_gene_vst_3270subjects_010822.csv', sep=',')
# Metabolomics (kNN-imputed, outliers removed; subjects x metabolites)
dfM = pd.read_csv('Metabolomics/COPDGene_P2_LT20missing_knnImpute_metabolites_20211021.csv', sep=',')
# Proteomics (SomaScan 5K, tab-separated)
dfP = pd.read_csv('Proteomics/COPDGeneSoma_SMP_5K_P2_16Jun20.txt', sep='\t')
# Clinical (visit-level, all phases)
clinData = pd.read_csv('Clinical Variables/COPDGene_P1P2P3_25SEP2020_VisitLevel.csv', sep=',')
# Raw counts table — kept around for gene-id lookup downstream
geneIDs = pd.read_csv('Transcriptomics/2021-09-28 original/counts_raw.tsv', sep='\t')

# Metabolomics CSV's first column is anonymous; rename to 'sid' so it can be
# used as the join key with the other tables.
dfM = dfM.rename(columns={'Unnamed: 0': 'sid'})

# Filter clinical down to Phase-2 visits and project the columns we need
# (sid, visitnum, gender, smoking_status, cohort, BMI, finalgold_visit,
#  finalGold, pctEmph).
clinData = getData.getClinicalData(clinData)

# ---------------------------------------------------------------------------
# 2. Align: build the patient intersection across modalities
# ---------------------------------------------------------------------------
# `u` = clinical inner-join metabolomics (used only to derive the metabolomics
# subject list `momicstemp`).
u = getData.getUnionofPatients(clinData, dfM)
momicstemp = getData.getUnionofPatients(u.iloc[:, 0], dfM)

# Transcriptomics arrives gene-rows x subject-cols; transpose so subjects are
# rows, then add an `sid` column from the index for joining.
dfT = dfT.transpose()
patientsT = dfT.index.values
dfT.insert(0, 'sid', patientsT, True)
ut = getData.getUnionofPatients(clinData, dfT)
tomicstemp = getData.getUnionofPatients(ut.iloc[:, 0], dfT)

# Proteomics aligned to clinical.
uP = getData.getUnionofPatients(clinData, dfP)
pomicstemp = getData.getUnionofPatients(uP.iloc[:, 0], dfP)

# Final per-modality matrices = subjects present in *all three* omics.
# Each `final{P,M,T}` keeps that modality's feature columns but is restricted
# to the common subject set (chained inner-joins on sid).
finalP = getData.getUnionofPatients(pomicstemp, momicstemp.iloc[:, 0])
finalP = getData.getUnionofPatients(finalP, tomicstemp.iloc[:, 0])

# Pull the clinical labels for the common-subject cohort (used downstream).
clinP = getData.getUnionofPatients(finalP, clinData)
clinSP = clinP['smoking_status']
genderP = clinP['gender']
clinGold = clinP['finalGold']
clinEmph = clinP['pctEmph']

# Drop the leading `sid` column and impute remaining NaNs to 0.
finalP = finalP.iloc[:, 1:]
finalP = finalP.replace(np.nan, 0)

finalM = getData.getUnionofPatients(pomicstemp.iloc[:, 0], momicstemp)
finalM = getData.getUnionofPatients(finalM, tomicstemp.iloc[:, 0])
clinM = getData.getUnionofPatients(finalM, clinData)
genderM = clinM['gender']
finalM = finalM.iloc[:, 1:]
finalM = finalM.replace(np.nan, 0)

finalT = getData.getUnionofPatients(pomicstemp.iloc[:, 0], momicstemp.iloc[:, 0])
finalT = getData.getUnionofPatients(finalT, tomicstemp)
clinT = getData.getUnionofPatients(finalT, clinData)
genderT = clinT['gender']
finalT = finalT.iloc[:, 1:]
finalT = finalT.replace(np.nan, 0)

# Concatenated raw-feature matrix across all three omics (kept for parity
# with downstream concatenations; not currently fed to any classifier).
allOmics = np.concatenate((finalT, finalM), axis=1)
allOmics = np.concatenate((allOmics, finalP), axis=1)

# ---------------------------------------------------------------------------
# 3. Transform: log + standardize, drop low-CV features
# ---------------------------------------------------------------------------
finalT = np.array(finalT)
finalP = np.array(finalP)
finalM = np.array(finalM)

# Log(1+x) on proteomics + metabolomics so heavy-tailed abundance values are
# better behaved for chi^2 feature selection. Transcriptomics is already VST.
finalP = dataExploration.logTransform(finalP)
finalM = dataExploration.logTransform(finalM)

# Per-feature z-score; keep only features whose std/mean exceeds 0.025
# (drops low-variance/low-information features).
featuresT = dataExploration.standardizeData(finalT, 0.025, 'Transcriptomics')
featuresP = dataExploration.standardizeData(finalP, 0.025, 'Proteomics')
featuresM = dataExploration.standardizeData(finalM, 0.025, 'Metabolomics')

# Concatenated standardized-feature matrix (used as input to the global GOLD
# stratum split below).
allOmicsF = np.concatenate((featuresT, featuresP), axis=1)
allOmicsF = np.concatenate((allOmicsF, featuresM), axis=1)

# ---------------------------------------------------------------------------
# 4. Build labels
# ---------------------------------------------------------------------------
# GOLD: drop GOLD-1 subjects (intermediate / ambiguous) then binarize to
# 0 = GOLD 0, 1 = GOLD 2-4.
clinGoldStrat, yGold = dataExploration.getRidofGold1(np.array(clinGold), np.array(allOmicsF))
clinGoldStrat = np.where(clinGoldStrat < 2, 0, 1)

# Emphysema: drop subjects with NaN pctEmph (also drop the same rows from
# every feature matrix), then binarize at 5%.
eNan = np.argwhere(np.isnan(np.array(clinEmph)))
clinEmph = np.delete(np.array(clinEmph), eNan, 0)
clinEmphB = np.where(clinEmph < 5, 0, 1)

print('-----------------------------')
vals, conts = np.unique(clinEmphB, return_counts=True)
print(vals)
print(conts)

# Apply the same NaN-row mask to each modality's feature matrix.
featuresTE = np.delete(featuresT, eNan, 0)
featuresPE = np.delete(featuresP, eNan, 0)
featuresME = np.delete(featuresM, eNan, 0)

# Train/test split on transcriptomics features against the *continuous*
# pctEmph (used only to establish y_test for downstream plots; the
# classifiers below are fit against the binarized label).
X_train, X_test, y_train, y_test = train_test_split(
    featuresTE, np.array(clinEmph), random_state=1
)

# ---------------------------------------------------------------------------
# 5. Univariate feature selection per modality (Emphysema target)
# ---------------------------------------------------------------------------
# SelectKBest(chi2): pick the top-k features most associated with the
# binary emphysema label. k differs per modality based on its dimensionality.
uniFeatTE = dataExploration.uniFeatureSelec(featuresTE, clinEmphB, 1000)
#neuralNetwork.votingClassifier(uniFeatTE, np.array(clinEmphB), y_test, 'Transciptomics Emphysema')

uniFeatPE = dataExploration.uniFeatureSelec(featuresPE, clinEmphB, 1000)
#neuralNetwork.votingClassifier(uniFeatPE, np.array(clinEmphB), y_test,'Proteomics Emphysema')

uniFeatME = dataExploration.uniFeatureSelec(featuresME, clinEmphB, 500)
#neuralNetwork.votingClassifier(uniFeatME, np.array(clinEmphB), y_test,'Metabolomics Emphysema')

# Concatenated emphysema-target feature matrix (kept for the commented-out
# all-omics emphysema run below).
allOmicsFE = np.concatenate((uniFeatTE, uniFeatME), axis=1)
allOmicsFE = np.concatenate((allOmicsFE, uniFeatPE), axis=1)

# ---------------------------------------------------------------------------
# 6. Univariate feature selection per modality (GOLD target)
# ---------------------------------------------------------------------------
# Re-run getRidofGold1 against each modality's standardized matrix so the
# row count matches clinGoldStrat after dropping GOLD-1 subjects.
clinGoldStrat, yGT = dataExploration.getRidofGold1(np.array(clinGold), np.array(featuresT))
clinGoldStrat, yGP = dataExploration.getRidofGold1(np.array(clinGold), np.array(featuresP))
clinGoldStrat, yGM = dataExploration.getRidofGold1(np.array(clinGold), np.array(featuresM))
clinGoldO = clinGoldStrat                # raw 0/2/3/4 GOLD label (kept for the test split)
c, v = np.unique(clinGoldStrat, return_counts=True)
print(c)
print(v)

# Binarize GOLD: 0 -> 0, 2-4 -> 1.
clinGoldStrat = np.where(clinGoldStrat < 2, 0, 1)

# Per-modality top-k features against binarized GOLD.
uniFT = dataExploration.uniFeatureSelec(yGT, clinGoldStrat, 5000)
uniFP = dataExploration.uniFeatureSelec(yGP, clinGoldStrat, 1000)
uniFM = dataExploration.uniFeatureSelec(yGM, clinGoldStrat, 500)

# Concatenated all-omics feature matrix for GOLD prediction.
allOmicsFG = np.concatenate((uniFT, uniFM), axis=1)
allOmicsFG = np.concatenate((allOmicsFG, uniFP), axis=1)

#neuralNetwork.votingClassifier(allOmicsFE, np.array(clinEmphB), y_test,'All Omics Emphysema')

# ---------------------------------------------------------------------------
# 7. Train / evaluate the voting classifier on each feature set
# ---------------------------------------------------------------------------
# Build the test split on the all-omics GOLD matrix so y_test (raw GOLD
# labels) can be passed in for the prediction-distribution plots.
X_train, X_test, y_train, y_test = train_test_split(
    allOmicsFG, np.array(clinGoldO), random_state=1
)
v, c = np.unique(y_train, return_counts=True)
print(v, c)
v, c = np.unique(y_test, return_counts=True)
print(v, c)

# Fit + evaluate on each modality and on the concatenated all-omics matrix.
# Each call emits ROC, confusion matrix, class-probability bar chart, and
# prediction-distribution histogram via matplotlib.
neuralNetwork.votingClassifier(allOmicsFG, np.array(clinGoldStrat), y_test, 'All Omics GOLD')
neuralNetwork.votingClassifier(uniFT,      np.array(clinGoldStrat), y_test, 'Transcriptomics GOLD')
neuralNetwork.votingClassifier(uniFP,      np.array(clinGoldStrat), y_test, 'Proteomics GOLD')
neuralNetwork.votingClassifier(uniFM,      np.array(clinGoldStrat), y_test, 'Metabolomics GOLD')
