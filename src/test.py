"""
Earlier scratch driver — superseded by `main.py`.

WARNING: This script is **not runnable as-is**. It was an early exploratory
driver written before the multi-omics intersection was fully wired up, and
it references symbols that were never defined in this file (e.g. `finalP`
and `momicstemp` are used before assignment), and calls functions with
signatures that have since changed (e.g. `dataExploration.standardizeData`
now requires a CV cutoff and a title; `neuralNetwork.votingClassifier`
now takes a fourth `title` argument).

It is preserved here only as a record of the early per-modality
gender-classification and PCA experiments. For an executable end-to-end
pipeline see `main.py` in this directory.

Author: Katherina Cortes
Date:   January 7, 2022
"""

import argparse

import numpy as np
import pandas as pd
import dataExploration, getData
import neuralNetwork

parser = argparse.ArgumentParser(description='Supervised Learning model for COPD subtypes')

# Load the same four COPDGene tables as main.py.
dfT = pd.read_csv('preprocessedRNAseq/X_gene_vst_3270subjects_010822.csv', sep=',')
dfM = pd.read_csv('Metabolomics/COPDGene_P2_LT20missing_knnImpute_metabolites_20211021.csv', sep=',')
dfP = pd.read_csv('Proteomics/COPDGeneSoma_SMP_5K_P2_16Jun20.txt', sep='\t')
clinData = pd.read_csv('Clinical Variables/COPDGene_P1P2P3_25SEP2020_VisitLevel.csv', sep=',')

dfM = dfM.rename(columns={'Unnamed: 0': 'sid'})
clinData = getData.getClinicalData(clinData)

# Per-modality intersection with clinical (used to derive label vectors).
u = getData.getUnionofPatients(clinData, dfM)
smokingS = u['smoking_status']
gender = u['gender']
emph = u['pctEmph']

# Transcriptomics: transpose so subjects are rows; tag with sid.
dfT = dfT.transpose()
patientsT = dfT.index.values
dfT.insert(0, 'sid', patientsT, True)
ut = getData.getUnionofPatients(clinData, dfT)
smokingST = ut['smoking_status']
genderT = ut['gender']
tomicstemp = getData.getUnionofPatients(ut.iloc[:, 0], dfT)
tomics = tomicstemp.iloc[:, 1:]
tomics = tomics.replace(np.nan, 0)


# Proteomics intersection with clinical.
uP = getData.getUnionofPatients(clinData, dfP)
smokingSP = uP['smoking_status']
genderP = uP['gender']
pomicstemp = getData.getUnionofPatients(uP.iloc[:, 0], dfP)
pomics = pomicstemp.iloc[:, 1:]
pomics = pomics.replace(np.nan, 0)


# NOTE: `finalP` is referenced below but never defined in this file —
# the original script was abandoned mid-edit. Left intact for historical
# reference; see `main.py` for the working version.
clinP = getData.getUnionofPatients(finalP, clinData)
clinSP = clinP['smoking_status']
genderP = clinP['gender']
clinGold = clinP['finalGold']
clinEmph = clinP['pctEmph']

finalP = finalP.iloc[:, 1:]
finalP = finalP.replace(np.nan, 0)
#dataExploration.pca(finalP, genderP)

print(finalP.shape)
# Per-modality gender classification baseline (proteomics).
xPNew = dataExploration.uniFeatureSelec(finalP, genderP, 1500)

# NOTE: `momicstemp` referenced but never defined here.
finalM = getData.getUnionofPatients(pomicstemp.iloc[:, 0], momicstemp)
finalM = getData.getUnionofPatients(finalM, tomicstemp.iloc[:, 0])
clinM = getData.getUnionofPatients(finalM, clinData)
genderM = clinM['gender']

finalM = finalM.iloc[:, 1:]
finalM = finalM.replace(np.nan, 0)

#dataExploration.pca(finalM, genderM)

# Metabolomics gender classification baseline + XGBoost test.
xMNew = dataExploration.uniFeatureSelec(finalM, genderM, 500)
#dataExploration.pca(xMNew, genderM)
neuralNetwork.xgb(xMNew, genderM)

finalT = getData.getUnionofPatients(pomicstemp.iloc[:, 0], momicstemp.iloc[:, 0])
finalT = getData.getUnionofPatients(finalT, tomicstemp)
clinT = getData.getUnionofPatients(finalT, clinData)
genderT = clinT['gender']

finalT = finalT.iloc[:, 1:]
finalT = finalT.replace(np.nan, 0)

#dataExploration.pca(finalT, genderT)

# Transcriptomics gender baseline.
xTNew = dataExploration.uniFeatureSelec(finalT, genderT, 3000)
#dataExploration.pca(xTNew, genderT)

#neuralNetwork.xgb(xTNew, genderT)
#neuralNetwork.votingClassifier(xTNew,genderT)

# Concatenate all omics (raw, then standardized + filtered).
allOmics = np.concatenate((finalT, finalM), axis=1)
allOmics = np.concatenate((allOmics, finalP), axis=1)

finalT = np.array(finalT)
finalP = np.array(finalP)
finalM = np.array(finalM)

# NOTE: `standardizeData` now requires (X, cutoff, title) — these calls
# would fail under the current signature. Kept as historical reference.
finalT = dataExploration.standardizeData(finalT)
finalP = dataExploration.standardizeData(finalP)
finalM = dataExploration.standardizeData(finalM)

# CV-based feature filter (per-modality cutoffs, hand-picked).
varsT, highT = dataExploration.featureFilter(finalT, 0.5, 'Transcriptomics')
varsP, highP = dataExploration.featureFilter(finalP, 1, 'Proteomics')
varsM, highM = dataExploration.featureFilter(finalM, 0.25, 'Metabolomics')

featuresT = finalT[:, highT]
featuresP = finalP[:, highP]
featuresM = finalM[:, highM]
print('shape')
print(featuresT.shape)
print(featuresP.shape)
print(featuresM.shape)

allOmicsF = np.concatenate((featuresT, featuresP), axis=1)
allOmicsF = np.concatenate((allOmicsF, featuresM), axis=1)

vals, conts = np.unique(clinGold, return_counts=True)
print(vals)
print(conts)

# Drop GOLD-1 + binarize to 0/1.
clinGoldStrat, yGold = dataExploration.getRidofGold1(np.array(clinGold), np.array(allOmicsF))
clinGoldStrat = np.where(clinGoldStrat < 2, 0, 1)

# Drop NaN-pctEmph rows from labels and from each modality's feature matrix.
eNan = np.argwhere(np.isnan(np.array(clinEmph)))
clinEmph = np.delete(np.array(clinEmph), eNan, 0)
print(np.unique(clinEmph))

# NOTE: this comparison is inverted relative to main.py (`> 5 -> 0`). In
# main.py the convention is `< 5 -> 0` (i.e. low emph = class 0).
clinEmph = np.where(clinEmph > 5, 0, 1)

print('-----------------------------')
vals, conts = np.unique(clinEmph, return_counts=True)
print(vals)
print(conts)

featuresTE = np.delete(featuresT, eNan, 0)
featuresPE = np.delete(featuresP, eNan, 0)
featuresME = np.delete(featuresM, eNan, 0)

# Per-modality voting-classifier runs against binary emphysema (with and
# without univariate feature selection). NOTE: votingClassifier signature
# now requires a fourth `title` argument; these three-arg calls are stale.
uniFeatTE = dataExploration.uniFeatureSelec(featuresTE, clinEmph, 5000)
neuralNetwork.votingClassifier(uniFeatTE, np.array(clinEmph), 'Uni Transcriptomics Emphysema')
neuralNetwork.votingClassifier(featuresTE, np.array(clinEmph), 'Transcriptomics Emphysema')

uniFeatPE = dataExploration.uniFeatureSelec(featuresPE, clinEmph, 1000)
neuralNetwork.votingClassifier(uniFeatPE, np.array(clinEmph), 'Uni Proteomics Emphysema')
neuralNetwork.votingClassifier(featuresPE, np.array(clinEmph), 'Proteomics Emphysema')

uniFeatME = dataExploration.uniFeatureSelec(featuresME, clinEmph, 500)
neuralNetwork.votingClassifier(uniFeatME, np.array(clinEmph), 'Uni Metabolomics Emphysema')
neuralNetwork.votingClassifier(featuresME, np.array(clinEmph), 'Metabolomics Emphysema')

allOmicsFE = np.concatenate((uniFeatTE, uniFeatME), axis=1)
allOmicsFE = np.concatenate((allOmicsFE, uniFeatPE), axis=1)

# Same again for binary GOLD.
clinGoldStrat, yGT = dataExploration.getRidofGold1(np.array(clinGold), np.array(featuresT))
clinGoldStrat, yGP = dataExploration.getRidofGold1(np.array(clinGold), np.array(featuresP))
clinGoldStrat, yGM = dataExploration.getRidofGold1(np.array(clinGold), np.array(featuresM))
clinGoldStrat = np.where(clinGoldStrat < 2, 0, 1)

uniFT = dataExploration.uniFeatureSelec(yGT, clinGoldStrat, 5000)
uniFP = dataExploration.uniFeatureSelec(yGP, clinGoldStrat, 1000)
uniFM = dataExploration.uniFeatureSelec(yGM, clinGoldStrat, 500)

allOmicsFG = np.concatenate((uniFT, uniFM), axis=1)
allOmicsFG = np.concatenate((allOmicsFG, uniFP), axis=1)

neuralNetwork.votingClassifier(allOmicsFE, np.array(clinEmph), 'Uni all Emph')
neuralNetwork.votingClassifier(allOmicsFG, np.array(clinGoldStrat), 'Uni all Gold')

standardizedX = dataExploration.standardizeData(allOmics)
standardizedX = np.nan_to_num(standardizedX)
