import os
import pandas as pd
from utils.tagging import assign_nature_weights

CATEGORIES = ["неживая природа", "живая природа", "активный отдых", "музыкальные", "гастрономия"]
SEASON_MAP = {'Летний': 1, 'Зимний': 0, 'Осенний': 0.5, 'Весенний': 0.3}

def load_cruises_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8', sep=';')

    df['Сезонность'] = df['Сезонность'].apply(lambda x: SEASON_MAP.get(str(x).strip(), 0.0))
    df['Цена в евро'] = (
        df['Цена в евро'].astype(str)
        .str.replace(r'[^\d.,]', '', regex=True)
        .str.replace(',', '.', regex=False)
        .astype(float)
        .fillna(0.0)
    )

    df['ТЭГИ'] = df['ТЭГИ'].fillna("").astype(str)
    df['Категории (веса)'] = df['ТЭГИ'].apply(assign_nature_weights)

    weights_df = pd.json_normalize(df['Категории (веса)'])
    df = pd.concat([df, weights_df[CATEGORIES]], axis=1)

    return df
