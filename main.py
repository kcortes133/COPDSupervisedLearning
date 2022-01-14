# Author: Katherina Cortes
# Date: January 7, 2022
# Purpose:
import argparse

import numpy as np
import pandas as pd
import dataExploration, getData

parser = argparse.ArgumentParser(description='Supervised Learning model for COPD subtypes')

df = pd.read_csv('preprocessedRNAseq/X_gene_3270subjects_010822.csv', sep=',')
dfM = pd.read_csv('Metabolomics/COPDGene_P2_ALL_metabolites_20211020.csv', sep=',')
#df = pd.read_csv('Proteomics/COPDGeneSoma_SMP_5K_P2_16Jun20.txt', sep='\t')
clinData = pd.read_csv('Clinical Variables/COPDGene_P1P2P3_25SEP2020_VisitLevel.csv', sep=',')
# trim first row
# trim first col
# //TODO what should nans be replaced with

dfM = dfM.rename(columns={'Unnamed: 0':'sid'})
clinData = getData.getClinicalData(clinData)

u = getData.getPatients(clinData, dfM)
print(u)