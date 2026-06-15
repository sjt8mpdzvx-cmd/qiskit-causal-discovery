"""Sachs 데이터 로드 및 전처리."""

import pandas as pd
import os


def load_sachs_data(data_dir=None):
    """Sachs 데이터셋 로드 (전체 11개 변수)."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(data_dir, "sachs_raw.csv")
    return pd.read_csv(path)


def load_sachs_subset(variables=None, data_dir=None):
    """4개 변수만 추출한 서브셋 로드."""
    if variables is None:
        variables = ["Raf", "Mek", "Erk", "Akt"]
    df = load_sachs_data(data_dir)
    return df[variables]
