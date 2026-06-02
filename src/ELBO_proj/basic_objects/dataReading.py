import os
from dataclasses import dataclass

import torch
import scipy.io as sio
import numpy as np


@dataclass
class TrainData:
    X_seq: torch.Tensor
    U_seq: torch.Tensor


def _load_mat_file(file_path):
    """
    Load MATLAB .mat data.

    scipy.io.loadmat supports MATLAB v7.2 and earlier files.
    MATLAB v7.3 files are HDF5-based, so they need to be read by h5py.
    """
    try:
        return sio.loadmat(file_path)
    except NotImplementedError as err:
        if "matlab v7.3" not in str(err).lower():
            raise

        try:
            import h5py
        except ImportError as h5py_err:
            raise ImportError(
                "当前 .mat 文件是 MATLAB v7.3 格式，scipy.io.loadmat 无法读取。"
                "请先安装 h5py，例如执行：python -m pip install h5py"
            ) from h5py_err

        mat_data = {}
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                obj = f[key]
                if not hasattr(obj, "shape"):
                    continue

                arr = np.array(obj)

                # MATLAB v7.3 uses HDF5 storage and matrices are commonly read
                # with reversed dimension order by h5py. For the 2D simulation
                # matrices used here, transpose them back to MATLAB's original
                # [time, dim] convention.
                if arr.ndim == 2:
                    arr = arr.T

                mat_data[key] = arr

        return mat_data


def load_matlab_simulation_data(
    file_path: str = None,
    S: int = None,
    dtype=torch.float32,
):
    """
    从 ../data/orinigalData.mat 中读取 S 组仿真数据，并构造 trainData。

    每组数据要求：
        simu_result_i.shape == [T + 1, x_dim]
        simu_input_i.shape  == [T, u_dim]

    返回：
        trainData.X_seq.shape == [S, T * x_dim]
        trainData.U_seq.shape == [S, x_dim + T * u_dim]
    """

    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "../data", "trainData.mat")

    mat_data = _load_mat_file(file_path)

    if S is None:
        S = 0
        while f"simu_result_{S + 1}" in mat_data:
            S += 1

        if S == 0:
            raise ValueError("没有在 .mat 文件中找到 simu_result_i 数据。")

    X_seq_list = []
    U_seq_list = []

    for i in range(1, S + 1):
        x_key = f"simu_result_{i}"
        u_key = f"simu_input_{i}"

        if x_key not in mat_data:
            raise KeyError(f"缺少变量：{x_key}")

        if u_key not in mat_data:
            raise KeyError(f"缺少变量：{u_key}")

        X_i = mat_data[x_key]      # shape: [T + 1, x_dim]
        U_i = mat_data[u_key]      # shape: [T, u_dim]

        X_i = torch.tensor(X_i, dtype=dtype)
        U_i = torch.tensor(U_i, dtype=dtype)

        if X_i.ndim != 2:
            raise ValueError(f"{x_key} 应该是二维矩阵，但实际维度为 {X_i.shape}")

        if U_i.ndim != 2:
            raise ValueError(f"{u_key} 应该是二维矩阵，但实际维度为 {U_i.shape}")

        T_plus_1, x_dim = X_i.shape
        T, u_dim = U_i.shape

        if T_plus_1 != T + 1:
            raise ValueError(
                f"{x_key} 和 {u_key} 的时间长度不匹配："
                f"{x_key}.shape={X_i.shape}, {u_key}.shape={U_i.shape}"
            )

        # X_seq 中只保存 x_1 到 x_T
        # X_i[1:, :].shape == [T, x_dim]
        # reshape 后得到 [T * x_dim]
        X_vec = X_i[1:, :].reshape(-1)

        # U_seq 中保存 x_0, u_0, ..., u_{T-1}
        # X_i[0, :].shape == [x_dim]
        # U_i.reshape(-1).shape == [T * u_dim]
        U_vec = torch.cat(
            [
                X_i[0, :],
                U_i.reshape(-1),
            ],
            dim=0,
        )

        X_seq_list.append(X_vec)
        U_seq_list.append(U_vec)

    X_seq = torch.stack(X_seq_list, dim=0)
    U_seq = torch.stack(U_seq_list, dim=0)

    trainData = TrainData(
        X_seq=X_seq,
        U_seq=U_seq,
    )

    return trainData