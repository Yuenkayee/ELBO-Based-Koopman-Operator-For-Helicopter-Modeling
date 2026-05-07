import os
import re
import numpy as np
import scipy.io as sio
import torch


def get_number_from_name(name):
    """
    从变量名中提取最后的编号。
    例如：
        simu_result_12 -> 12
        simu_input_3   -> 3
    """
    match = re.search(r"_(\d+)$", name)

    if match is None:
        raise ValueError(f"变量名 {name} 不符合 xxx_编号 的格式")

    return int(match.group(1))


def load_matlab_simulation_data(mat_filename="Data.mat"):
    """
    从 Data.mat 中读取：
        simu_result_1, simu_result_2, ..., simu_result_batch
        simu_input_1,  simu_input_2,  ..., simu_input_batch

    每个 simu_result_i 的 shape 应为 [T, dim]
    每个 simu_input_i  的 shape 应为 [T, dim_u]

    返回：
        data:       torch.Tensor, shape = [batch, T, dim]
        data_input: torch.Tensor, shape = [batch, T, dim_u]
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    mat_path = os.path.join(current_dir, mat_filename)

    mat_data = sio.loadmat(mat_path)

    result_names = [
        name for name in mat_data.keys()
        if name.startswith("simu_result_")
    ]

    input_names = [
        name for name in mat_data.keys()
        if name.startswith("simu_input_")
    ]

    result_names = sorted(result_names, key=get_number_from_name)
    input_names = sorted(input_names, key=get_number_from_name)

    if len(result_names) == 0:
        raise ValueError("Data.mat 中没有找到 simu_result_i 形式的变量")

    if len(input_names) == 0:
        raise ValueError("Data.mat 中没有找到 simu_input_i 形式的变量")

    if len(result_names) != len(input_names):
        raise ValueError(
            f"状态结果数量和输入数量不一致："
            f"simu_result 数量 = {len(result_names)}, "
            f"simu_input 数量 = {len(input_names)}"
        )

    result_indices = [get_number_from_name(name) for name in result_names]
    input_indices = [get_number_from_name(name) for name in input_names]

    if result_indices != input_indices:
        raise ValueError(
            f"simu_result_i 和 simu_input_i 的编号不一致：\n"
            f"result 编号 = {result_indices}\n"
            f"input 编号  = {input_indices}"
        )

    result_list = []
    input_list = []

    for result_name, input_name in zip(result_names, input_names):
        result = np.array(mat_data[result_name])
        simu_input = np.array(mat_data[input_name])

        if result.ndim != 2:
            raise ValueError(
                f"{result_name} 的维度不是二维矩阵，当前 shape = {result.shape}"
            )

        if simu_input.ndim != 2:
            raise ValueError(
                f"{input_name} 的维度不是二维矩阵，当前 shape = {simu_input.shape}"
            )

        if result.shape[0] != simu_input.shape[0]:
            raise ValueError(
                f"{result_name} 和 {input_name} 的时间长度 T 不一致："
                f"{result_name}.shape = {result.shape}, "
                f"{input_name}.shape = {simu_input.shape}"
            )

        result_list.append(result)
        input_list.append(simu_input)

    data_states = np.stack(result_list, axis=0)
    data_input = np.stack(input_list, axis=0)

    data_states = torch.tensor(data_states, dtype=torch.float32)
    data_input = torch.tensor(data_input, dtype=torch.float32)

    return data_states, data_input


if __name__ == "__main__":
    data, data_input = load_matlab_simulation_data("Data.mat")

    print("data.shape =", data.shape)
    print("data_input.shape =", data_input.shape)

    print("batch =", data.shape[0])
    print("T =", data.shape[1])
    print("dim =", data.shape[2])
    print("dim_u =", data_input.shape[2])