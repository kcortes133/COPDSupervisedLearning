"""
Data-loading helpers for the COPD pipeline.

Two thin wrappers used throughout `main.py`:

    getClinicalData     — project the COPDGene visit-level CSV down to the
                          clinical columns we care about and keep only the
                          Phase-2 visit (visitnum == 2).
    getUnionofPatients  — inner-join two tables on the subject-id column
                          (`sid`) so that downstream code only ever sees
                          subjects present in *both* inputs.
"""

import pandas as pd


def getClinicalData(data):
    """Project the COPDGene visit-level table to the columns we use.

    The COPDGene visit-level CSV has hundreds of columns; this pipeline only
    uses a small handful. Projecting early keeps memory + downstream joins
    small and makes the column set explicit.

    Parameters
    ----------
    data : pandas.DataFrame
        The full visit-level COPDGene clinical CSV.

    Returns
    -------
    pandas.DataFrame
        Phase-2 rows (`visitnum == 2`) with columns:
        sid, visitnum, gender, smoking_status, cohort, BMI,
        finalgold_visit, finalGold, pctEmph.
    """
    print(data.shape)
    cols = ['sid', 'visitnum', 'gender', 'smoking_status', 'cohort',
            'BMI', 'finalgold_visit', 'finalGold', 'pctEmph']
    df = pd.DataFrame(data, columns=cols)
    return df.loc[df['visitnum'] == 2]


def getUnionofPatients(data1, data2):
    """Inner-join two subject-keyed tables on the `sid` column.

    Despite the function name, this is an *intersection* of subjects (an
    inner join), not a union — only subjects appearing in both tables are
    kept. The name is preserved for compatibility with the rest of the
    pipeline.

    Parameters
    ----------
    data1, data2 : pandas.DataFrame
        Each must contain an `sid` column.

    Returns
    -------
    pandas.DataFrame
        Rows for subjects present in both inputs, with columns from both
        sides concatenated.
    """
    unionData = pd.merge(data1, data2, how='inner', on='sid')
    return unionData
