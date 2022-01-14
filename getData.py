import pandas as pd


def getClinicalData(data):
    # sid, visitnum, gender, smoking_status
    # cohort
    print(data.shape)
    cols = ['sid', 'visitnum', 'gender', 'smoking_status', 'cohort', 'BMI', 'finalgold_visit', 'finalGold']
    df = pd.DataFrame(data, columns=cols)
    return df.loc[df['visitnum'] == 2]

def getUnionofPatients(data1, data2):
    unionData = pd.merge(data1, data2, how='inner', on='sid')
    return unionData.iloc[:,0]