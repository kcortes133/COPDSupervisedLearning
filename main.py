# Author: Katherina Cortes
# Date: January 7, 2022
# Purpose:
import argparse

import numpy as np
import pandas as pd
import dataExploration

parser = argparse.ArgumentParser(description='Supervised Learning model for COPD subtypes')

df = pd.read_csv('preprocessedRNAseq/X_gene_3270subjects_010822.csv', sep=',')
df = pd.read_csv('Metabolomics/COPDGene_P2_ALL_metabolites_20211020.csv', sep=',')
#df = pd.read_csv('Proteomics/COPDGeneSoma_SMP_5K_P2_16Jun20.txt', sep='\t')
df.shape
# trim first row
df = df.iloc[1:]
# trim first col
df = df.iloc[:, 1:]
# //TODO what should nans be replaced with
df = df.fillna(0)


dataExploration.pca(df)
#dataExploration.featureSelec(df)