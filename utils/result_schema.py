STANDARD_COLUMNS = [
    'mtx',
    'kernel',
    'threads',
    'Analyze',
    'Factorization_1',
    'Factorization_2',
    'Factorization',
    'Read',
    'RHS_Creation',
    'Average_Solve'
]

import pandas as pd


def normalize_result(df: pd.DataFrame, kernel_name: str) -> pd.DataFrame:
    """Map solver specific columns to the common schema."""
    df = df.copy()
    df['kernel'] = kernel_name
    if 'threads' not in df.columns:
        df['threads'] = 1
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[STANDARD_COLUMNS]
