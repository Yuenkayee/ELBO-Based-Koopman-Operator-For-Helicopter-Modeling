import torch
import torch.nn as nn

# ============================================================
# 2. Temporal Encoder: bidirectional LSTM
# ============================================================

class TemporalEncoder(nn.Module):
    """
    Temporal encoder:
        input:  I_t = [x_t; u_t]
        output: temporal embedding \hat{I}_t

    Shape:
        x_seq: [batch, T, x_dim]
        u_seq: [batch, T-1, u_dim]

    Since u has length T-1, we pad the last control input with zero.
    """

    def __init__(self, x_dim, u_dim, hidden_dim, embed_dim):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim

        # 构造一个 LSTM 类
        """
            LSTM 的具体用法如下所示，其输入张量是 
            包含了多段时序仿真结果的 x : 其段数是 batch, 时间长度是 T, 每一个时间步的输入向量维度是 input_size
            即 x.shape = [batch, T, input_size]
            输出为 output.shape = [batch , T , hidden_size] (bidirectional = false 时)
            或  output.shape = [batch , T , 2 * hidden_size] (bidirectional = true 时)
            lstm = nn.LSTM(
                input_size=10,
                hidden_size=64,
                num_layers=1,
                batch_first=True
            )
            x = torch.randn(32, 20, 10)
            output, (h_n, c_n) = lstm(x)
            print(output.shape)
            print(h_n.shape)
            print(c_n.shape)
        """
        self.lstm = nn.LSTM(
            input_size=x_dim + u_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True # 表明这是双向 LSTM
        )
        ## nn.Linear 是全连接线性层 y = x * W^T + b，主要作用是改变特征维度，可以作为编码器和解码器，参数是（输入维度，输出维度）
        # 这里的主要作用是进行线性投影（初始化了一个线性投影的类）
        # 线性投影层的作用是将 LSTM 输出的 2 * hidden_dim
        self.proj = nn.Linear(2 * hidden_dim, embed_dim)

    def forward(self, x_seq, u_seq):
        batch_size, T, _ = x_seq.shape

        # Pad u_T as zero so that u_seq_pad has length T
        # 待拼接的张量 u_pad，相当于为了补全 T_t = [x_t, u_t] 对，人为添加了一个零控制输入
        u_pad = torch.zeros(
            batch_size, 1, self.u_dim,
            device=x_seq.device, # 创建出来的零张量和 x_seq 使用相同的设备（相同的 CPU 或者 GPU）
            dtype=x_seq.dtype    # 创建出来的零张量和 x_seq 使用相同的数据类型
        )

        # torch.cat 是用于拼接张量的函数
        u_seq_pad = torch.cat([u_seq, u_pad], dim=1)

        I_seq = torch.cat([x_seq, u_seq_pad], dim=-1)

        lstm_out, _ = self.lstm(I_seq)

        I_hat = torch.tanh(self.proj(lstm_out))

        return I_hat

