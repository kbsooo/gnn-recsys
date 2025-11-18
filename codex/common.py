"""
공통 유틸: 데이터 로드/인덱싱, 스플릿, 네거티브 샘플링 도우미.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def load_interactions(path: Path) -> pd.DataFrame:
    """CSV에서 user, item, rating 컬럼을 읽어온다."""
    df = pd.read_csv(path)
    expected = {"user", "item"}
    if not expected.issubset(df.columns):
        raise ValueError(f"columns must include {expected}, got {df.columns}")
    return df


def encode_ids(
    df: pd.DataFrame, user_col: str = "user", item_col: str = "item"
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    user/item을 0부터 시작하는 연속 인덱스로 변환.
    반환: 변환된 df, user2idx, item2idx 매핑.
    """
    users = pd.Index(sorted(df[user_col].unique()))
    items = pd.Index(sorted(df[item_col].unique()))

    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {v: i for i, v in enumerate(items)}

    df = df.copy()
    df["user_idx"] = df[user_col].map(user2idx)
    df["item_idx"] = df[item_col].map(item2idx)
    return df, user2idx, item2idx


def split_userwise(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    user_col: str = "user_idx",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    각 유저 내에서 무작위로 섞어 8/1/1 비율로 분리.
    시간 정보가 없으므로 순서를 섞은 뒤 슬라이스.
    """
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []

    for _, group in df.groupby(user_col):
        idx = rng.permutation(len(group))
        group_shuffled = group.iloc[idx]
        n = len(group_shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = group_shuffled.iloc[:n_train]
        val = group_shuffled.iloc[n_train : n_train + n_val]
        test = group_shuffled.iloc[n_train + n_val :]
        parts.append((train, val, test))

    trains, vals, tests = zip(*parts)
    return (
        pd.concat(trains, ignore_index=True),
        pd.concat(vals, ignore_index=True),
        pd.concat(tests, ignore_index=True),
    )


def build_user_pos_dict(
    df: pd.DataFrame, user_col: str = "user_idx", item_col: str = "item_idx"
) -> Dict[int, set]:
    """유저별 관측 아이템 집합 생성."""
    return (
        df.groupby(user_col)[item_col]
        .agg(lambda x: set(x.tolist()))
        .to_dict()
    )


def item_popularity_weights(
    df: pd.DataFrame, item_col: str = "item_idx"
) -> np.ndarray:
    """
    아이템 빈도를 확률로 변환. 1e-8 가중치 더해 0 확률 방지.
    반환 배열 길이는 아이템 개수, 인덱스 = item_idx.
    """
    counts = df[item_col].value_counts().sort_index()
    n_items = counts.index.max() + 1
    freq = np.zeros(n_items, dtype=np.float64)
    freq[counts.index.to_numpy()] = counts.to_numpy()
    freq = freq + 1e-8
    prob = freq / freq.sum()
    return prob


def sample_negatives_popular(
    users: Iterable[int],
    user_pos: Dict[int, set],
    num_items: int,
    pop_prob: np.ndarray,
    n_neg: int = 1,
    rng: np.random.Generator | None = None,
) -> List[List[int]]:
    """
    인기 분포(pop_prob)를 우선 사용하여 미관측 아이템을 샘플링.
    pop_prob는 item_idx 길이의 확률 분포.
    """
    rng = rng or np.random.default_rng()
    sampled: List[List[int]] = []
    for u in users:
        seen = user_pos.get(u, set())
        negs: List[int] = []
        # 반복 샘플링으로 seen을 피한다.
        while len(negs) < n_neg:
            cand = int(rng.choice(num_items, p=pop_prob))
            if cand in seen:
                continue
            negs.append(cand)
        sampled.append(negs)
    return sampled


def make_binary_label(df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """rating >= threshold -> 1, else 0"""
    if "rating" not in df.columns:
        raise ValueError("rating column is required for binary label")
    df = df.copy()
    df["label"] = (df["rating"] >= threshold).astype(int)
    return df
