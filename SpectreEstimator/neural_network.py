import numpy as np
import torch
from SpectreEstimator.config import g_torch_type, g_device
from typing import Callable, Type


def learn_neural_network(
    in_xs: torch.Tensor,
    loss_func: Callable[[torch.Tensor], torch.Tensor],
    in_module: Type[torch.nn.Module],
    in_optimizer: Type[torch.optim.Optimizer],
    in_loop_num: int = 1000,
):
    """
    in_xs: 入力データ
    loss_func: in_module(in_xs) を入力として受け取り、スカラーの損失を返す関数
    in_module: ニューラルネットワークモジュールのクラス
    in_optimizer: 最適化手法 in_moduleのパラメータと結びついている必要がある
    in_loop_num: 学習ループの回数
    """
    for _ in range(in_loop_num):
        in_optimizer.zero_grad()
        l_loss = loss_func(in_module(in_xs))
        l_loss.backward()
        in_optimizer.step()
    return l_loss
