import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable
from dataclasses import dataclass
from SpectreEstimator.config import g_numpy_type


@dataclass(frozen=True)
class TransportCoefficients:
    xs: np.ndarray
    L11s: np.ndarray
    L12s: np.ndarray
    L22s: np.ndarray

    def __post_init__(self):
        assert self.xs.dtype == g_numpy_type
        assert self.L11s.dtype == g_numpy_type
        assert self.L12s.dtype == g_numpy_type
        assert self.L22s.dtype == g_numpy_type
        assert self.L11s.shape == self.L12s.shape == self.L22s.shape
        assert self.xs.shape[0] == self.L11s.shape[0]
        assert self.L11s.ndim == 1
        assert self.L12s.ndim == 1
        assert self.L22s.ndim == 1


def make_transport_coeffs_from_csv(
        in_csv_file: Path,
        in_x_names: str | list[str],
        in_L11_name: str,
        in_L12_name: str,
        in_L22_name: str,
        *,
        in_x_preprocess: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        in_L11_preprocess: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        in_L12_preprocess: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        in_L22_preprocess: Callable[[np.ndarray], np.ndarray] = lambda x: x
) -> TransportCoefficients:
    """
    CSVファイルから、TransportCoefficientsを作成する。
    各列の名前は、in_x_names, in_L11_name, in_L12_name, in_L22_nameで指定する。
    欠損値のある行は削除される。
    in_x_preprocess, in_L11_preprocess, in_L12_preprocess, in_L22_preprocessは、それぞれ、
    x, L11, L12, L22のデータを加工する関数を指定する。
    例えば、lambda x: np.log(x) とすると、xの値を対数変換する。
    指定しなければ、何もしない。
    """
    l_df = pd.read_csv(in_csv_file)
    if isinstance(in_x_names, str):
        in_x_names = [in_x_names]
    l_df = l_df[in_x_names + [in_L11_name, in_L12_name, in_L22_name]]
    l_df = l_df.dropna()

    l_xs = l_df[in_x_names].to_numpy(dtype=g_numpy_type)
    l_L11s = l_df[in_L11_name].to_numpy(dtype=g_numpy_type)
    l_L12s = l_df[in_L12_name].to_numpy(dtype=g_numpy_type)
    l_L22s = l_df[in_L22_name].to_numpy(dtype=g_numpy_type)

    l_xs = in_x_preprocess(l_xs)
    l_L11s = in_L11_preprocess(l_L11s)
    l_L12s = in_L12_preprocess(l_L12s)
    l_L22s = in_L22_preprocess(l_L22s)

    return TransportCoefficients(l_xs, l_L11s, l_L12s, l_L22s)
