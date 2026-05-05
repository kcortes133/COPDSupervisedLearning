# Author: Katherina Cortes
# Date: January 7, 2022
# Purpose:
import argparse

import numpy as np
import pandas as pd
import dataExploration, getData
import neuralNetwork

parser = argparse.ArgumentParser(description='Supervised Learning model for COPD subtypes')

dfT = pd.read_csv('preprocessedRNAseq/X_gene_vst_3270subjects_010822.csv', sep=',')
dfM = pd.read_csv('Metabolomics/COPDGene_P2_LT20missing_knnImpute_metabolites_20211021.csv', sep=',')
dfP = pd.read_csv('Proteomics/COPDGeneSoma_SMP_5K_P2_16Jun20.txt', sep='\t')
clinData = pd.read_csv('Clinical Variables/COPDGene_P1P2P3_25SEP2020_VisitLevel.csv', sep=',')

dfM = dfM.rename(columns={'Unnamed: 0':'sid'})
clinData = getData.getClinicalData(clinData)

u = getData.getUnionofPatients(clinData, dfM)
smokingS = u['smoking_status']
gender = u['gender']
emph = u['pctEmph']

dfT = dfT.transpose()
patientsT = dfT.index.values
dfT.insert(0, 'sid', patientsT, True)
ut = getData.getUnionofPatients(clinData, dfT)
smokingST = ut['smoking_status']
genderT = ut['gender']
tomicstemp = getData.getUnionofPatients(ut.iloc[:,0], dfT)
tomics = tomicstemp.iloc[:,1:]
tomics = tomics.replace(np.nan, 0)


uP = getData.getUnionofPatients(clinData, dfP)
smokingSP = uP['smoking_status']
genderP = uP['gender']
pomicstemp = getData.getUnionofPatients(uP.iloc[:,0], dfP)
pomics = pomicstemp.iloc[:,1:]
pomics = pomics.replace(np.nan, 0)


clinP = getData.getUnionofPatients(finalP, clinData)
clinSP = clinP['smoking_status']
genderP = clinP['gender']
clinGold = clinP['finalGold']
clinEmph = clinP['pctEmph']

finalP = finalP.iloc[:,1:]
finalP = finalP.replace(np.nan, 0)
#dataExploration.pca(finalP, genderP)

print(finalP.shape)
xPNew = dataExploration.uniFeatureSelec(finalP, genderP, 1500)

finalM = getData.getUnionofPatients(pomicstemp.iloc[:,0], momicstemp)
finalM = getData.getUnionofPatients(finalM, tomicstemp.iloc[:,0])
clinM = getData.getUnionofPatients(finalM, clinData)
genderM = clinM['gender']

finalM = finalM.iloc[:,1:]
finalM = finalM.replace(np.nan, 0)

#dataExploration.pca(finalM, genderM)

#print(finalM.shape)
xMNew = dataExploration.uniFeatureSelec(finalM, genderM, 500)
#dataExploration.pca(xMNew, genderM)
neuralNetwork.xgb(xMNew, genderM)

finalT = getData.getUnionofPatients(pomicstemp.iloc[:,0], momicstemp.iloc[:,0])
finalT = getData.getUnionofPatients(finalT, tomicstemp)
clinT = getData.getUnionofPatients(finalT, clinData)
#clinT = clinT['smoking_status']
genderT = clinT['gender']

finalT = finalT.iloc[:,1:]
finalT = finalT.replace(np.nan, 0)

#dataExploration.pca(finalT, genderT)

xTNew = dataExploration.uniFeatureSelec(finalT, genderT, 3000)
#dataExploration.pca(xTNew, genderT)

#neuralNetwork.xgb(xTNew, genderT)
#neuralNetwork.votingClassifier(xTNew,genderT)

# concatenate all omics
allOmics = np.concatenate((finalT, finalM), axis=1)
allOmics = np.concatenate((allOmics, finalP), axis=1)

finalT = np.array(finalT)
finalP = np.array(finalP)
finalM = np.array(finalM)

finalT = dataExploration.standardizeData(finalT)
finalP = dataExploration.standardizeData(finalP)
finalM = dataExploration.standardizeData(finalM)

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

#standFOmics = dataExploration.standardizeData(allOmicsF)

vals, conts = np.unique(clinGold, return_counts=True)

print(vals)
print(conts)

clinGoldStrat, yGold = dataExploration.getRidofGold1(np.array(clinGold), np.array(allOmicsF))
clinGoldStrat = np.where(clinGoldStrat < 2, 0, 1)
#neuralNetwork.votingClassifier(yGold, clinGoldStrat, 'GOLD without 1')
#neuralNetwork.votingClassifier(allOmicsF, clinSP, 'Smoking')


#neuralNetwork.votingClassifier(standFOmics, clinSP, 'Smoking Standardized Filtered')
#neuralNetwork.votingClassifier(standFOmics, clinGold, 'GOLD Standardized Filtered')

eNan = np.argwhere(np.isnan(np.array(clinEmph)))
clinEmph = np.delete(np.array(clinEmph), eNan, 0)
print(np.unique(clinEmph))

clinEmph = np.where(clinEmph > 5, 0, 1)

print('-----------------------------')
vals, conts  = np.unique(clinEmph, return_counts=True)
print(vals)
print(conts)

featuresTE = np.delete(featuresT, eNan, 0)
featuresPE = np.delete(featuresP, eNan, 0)
featuresME = np.delete(featuresM, eNan, 0)

#allOmicsFE = np.delete(np.array(allOmicsF), eNan, 0)

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
