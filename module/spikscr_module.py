import torch
import torch.nn as nn

from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode
from spikingjelly.activation_based import neuron, layer
# import torch.nn.functional as F
import math
# TODO: 借鉴了Transformer的原版代码，加入droupout等等
# TODO: 现在的Spiking-Driven Transformer是Norm First的，但Transformer中默认，Norm_First=False, 是否需要修改？
from .conv import Transpose,PointwiseConv1d,DepthwiseConv1d
from rotary_embedding_torch import RotaryEmbedding

class MS_MLP(nn.Module):
    def __init__(
        self,
        config,
        in_features,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.config = config
        if self.config.use_ln:
            self.ln = nn.LayerNorm(in_features)
        # batch, time, dim
        self.trans1 = Transpose(1,0,2)


        self.fc1_conv = layer.Linear(in_features, hidden_features, bias=False, step_mode='m')
        self.fc1_bn = layer.BatchNorm1d(hidden_features, step_mode='m')


        if spike_mode == "lif":
            self.fc1_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True ,backend=config.backend)

        elif spike_mode == "plif":
            self.fc1_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)
        if self.config.use_dp:
            self.dropout1 = layer.Dropout(config.dropout_p, step_mode='m')

        self.fc2_conv = layer.Linear(hidden_features, out_features, bias=False, step_mode='m')
        self.fc2_bn = layer.BatchNorm1d(out_features, step_mode='m')
        # self.fc2_ln = nn.LayerNorm(out_features)

        if spike_mode == "lif":
            self.fc2_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        elif spike_mode == "plif":
            self.fc2_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)
        if self.config.use_dp:
            self.dropout2 = layer.Dropout(config.dropout_p, step_mode='m')



    def forward(self, x):
        if self.config.use_ln:
            #  batch, time, dim
            x = self.trans1(x)
            x = self.ln(x)
            # time, batch, dim
            x = self.trans1(x)

        x = self.fc1_conv(x)
        # Transformer原版中，没有LN这个操作
        x = self.fc1_bn(x).contiguous()
        # fc1_lif 可以放在fc1前面
        x = self.fc1_lif(x)
        if self.config.use_dp:
            x = self.dropout1(x)

        x = self.fc2_conv(x)
        # Transformer原版中，没有LN和LIF
        x = self.fc2_bn(x).contiguous()
        # fc2_lif可以放在fc2前面
        x = self.fc2_lif(x)
        if self.config.use_dp:
            x = self.dropout2(x)
        # x = x + identity
        return x


class RLIF(nn.Module):
    def __init__(
        self,
        dim,
        config,
        kernel_size,
    ):
        super().__init__()
        self.lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)
        self.dwconv = DepthwiseConv1d(dim, dim, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False) # 现在的kernel size很大
        self.pwconv = PointwiseConv1d(dim, dim, stride=1, padding=0, bias=True) # bias= config.bias

        # time, batch, dim => batch, dim, time
        self.trans1 = Transpose(1, 2, 0)
        # batch, dim, time => time, batch ,dim
        self.trans2 = Transpose(2, 0, 1)

    def forward(self, inputs):
        x = self.trans1(inputs)
        x = self.pwconv(x)
        x = self.dwconv(x)
        x = self.trans2(x)
        x = self.lif(x)
        return x+inputs


class MS_SSA(nn.Module):
    def __init__(
        self,
        dim,
        config,
        num_heads=8,
        init_tau = 2.0,
        spike_mode="lif",
        layers=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        # self.dvs = dvs
        self.num_heads = num_heads
        self.config = config
        if self.config.use_ln:
            self.ln = nn.LayerNorm(dim)
            if spike_mode == 'lif':
                self.head_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                                        surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                                        step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)
            elif spike_mode == 'plif':
                self.head_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                                                  surrogate_function=config.surrogate_function,
                                                  detach_reset=config.detach_reset,
                                                  step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)
            if self.config.use_dp:
                self.head_dp = layer.Dropout(config.dropout_p, step_mode='m')
        # batch, time, dim
        self.trans1 = Transpose(1,0,2)

        self.scale = 1/math.sqrt(self.dim//self.num_heads)



        self.q_conv = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias, step_mode='m')
        # self.q = layer.Lin
        self.q_bn = layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m')
        # self.q_ln = nn.LayerNorm(self.config.n_hidden_neurons)

        if spike_mode == "lif":
            self.q_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)

            self.q_lif2 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)

        elif spike_mode == "plif":
            self.q_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)

            self.q_lif2 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)


        self.k_conv = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias,step_mode='m')

        self.k_bn = layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m')

        if spike_mode == "lif":
            self.k_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

            self.k_lif2 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        elif spike_mode == "plif":
            self.k_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)

            self.k_lif2 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)

        self.v_conv = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias,step_mode='m')

        self.v_bn = layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m')

        if spike_mode == "lif":
            self.v_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)


        elif spike_mode == "plif":
            self.v_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)

        if self.config.use_dp:
            self.attn_dropout = layer.Dropout(config.dropout_p, step_mode='m')

        if spike_mode == "lif":
            self.attn_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        elif spike_mode == "plif":
            self.attn_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)

        self.proj_conv = layer.Linear(dim, dim, bias=self.config.bias,step_mode='m')
        self.proj_bn = layer.BatchNorm1d(dim, step_mode='m')
        if self.config.use_dp:
            self.dropout = layer.Dropout(config.dropout_p, step_mode='m')

        if spike_mode == "lif":
            self.proj_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        elif spike_mode == "plif":
            self.proj_lif = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)

        self.layers = layers
        self.rotary_emb = RotaryEmbedding(dim=self.dim//self.num_heads)

    def forward(self, x, attention_mask=None):
        # identity = x
        if self.config.use_ln:
            #  batch, time, dim
            x = self.trans1(x)
            x = self.ln(x)
            # time, batch, dim
            x = self.trans1(x)
            x = self.head_lif(x)


        T, B, N = x.shape

        x_qkv = x # B*N*T


        q_conv_out = self.q_conv(x_qkv)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out)

        q = (
            q_conv_out
                .reshape(T, B, self.num_heads, N // self.num_heads)
                .permute(1, 2, 0, 3) # => (B, H, T, D)
                .contiguous()
        )
        # q = self.rotary_emb.rotate_queries_or_keys(q) # .permute(2,0,1,3)
        q = self.rotary_emb.rotate_queries_or_keys(q).reshape(T,B,-1)  # .permute(2,0,1,3)
        # q = self.q_lif2(q) # .permute(1,2,0,3)
        q = self.q_lif2(q).reshape(B,self.num_heads, T, N // self.num_heads) # .permute(1,2,0,3) # (B,H,T,D)

        k_conv_out = self.k_conv(x_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out)

        k = (
            k_conv_out
                .reshape(T, B, self.num_heads, N // self.num_heads)
                .permute(1, 2, 0, 3) # => (B,H,T,D)
                .contiguous()
        )
        # k = self.rotary_emb.rotate_queries_or_keys(k) # .permute(2,0,1,3) # (T,B,H,D)
        k = self.rotary_emb.rotate_queries_or_keys(k).reshape(T,B,-1)  # .permute(2,0,1,3) # (T,B,H,D)
        # k = self.k_lif2(k)  # .permute(1,2,0,3) # (B,H,T,D)
        k = self.k_lif2(k).reshape(B, self.num_heads, T, N // self.num_heads) # .permute(1,2,0,3) # (B,H,T,D)


        v_conv_out = self.v_conv(x_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_lif(v_conv_out)

        v = (
            v_conv_out
                .reshape(T, B, self.num_heads, N // self.num_heads)
                .permute(1, 2, 0, 3) # => (B,H,T,D)
                .contiguous()
        )
        # v = self.rotary_emb.rotate_queries_or_keys(v)
        # v = self.v_lif2(v)

        # v = v.permute(1,0,2,3)

        if self.config.attn_mode == 'v1':
            qk = q.mul(k) # (B,H,T,D)

            # 这是SDT V2的做法,在qkv计算完成后,使用LIF
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # (Batch, 1,1,Time)
            # 将 attention_mask 广播到与 scores 形状相同
            attention_mask = attention_mask.expand(qk.size())
            # 应用掩码：将掩码为 False 的位置设为负无穷
            qk = qk.masked_fill(attention_mask == False, float(0.))

            qk = qk.sum(dim=-2, keepdim=True) # (B,H,1,D)
            qk = qk * self.scale

            qk = self.attn_lif(qk)
            qk = self.attn_dropout(qk)

            # x = q.mul(kv * self.scale) # (B,H,T,D)
            x = v.mul(qk)  # (B,H,T,D)

            x = x.permute(2,0,1,3) # (T,B,H,D)

            # Flatten the last two dimensions
            x = x.reshape(T, B, -1).contiguous()  # Ensure the tensor is stored in a contiguous chunk of memory
            # x = x.permute(0,2,1)
            x = self.proj_bn(self.proj_conv(x)).contiguous()
            # 注意在原版的SDT 中没有这个LIF
            x = self.proj_lif(x)
            x = self.dropout(x)
            # x = x + identity
            return x

        elif self.config.attn_mode == 'v2':

            if attention_mask is not None: # attention_mask:(B,T)

                attn = (q@ k.transpose(-2, -1))

                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # (Batch, 1,1,Time)
                # 将 attention_mask 广播到与 scores 形状相同
                attention_mask = attention_mask.expand(attn.size())

                # 应用掩码：将掩码为 False 的位置设为负无穷
                attn = attn.masked_fill(attention_mask == False, float(0.))

                x = attn@v # (B,H,T,T) * (B,H,T,D)
                x = x*self.scale  # (B,H,T,T)
                x = x.permute(2, 0, 1, 3)  # (T,B,H,D)

                # 这是SDT V2的做法,在qkv计算完成后,使用LIF
                x = self.attn_lif(x)
                if self.config.use_dp:
                    x = self.attn_dropout(x)
                x = x.reshape(T, B, -1).contiguous()  # Ensure the tensor is stored in a contiguous chunk of memory

                x = self.proj_bn(self.proj_conv(x)).contiguous()
                # 注意在原版的SDT 中没有这个LIF
                x = self.proj_lif(x)
                if self.config.use_dp:
                    x = self.dropout(x)
                # x = x + identity
                return x

                # if visual is needed
                # return x, attn


class GSU(nn.Module):
    """
    The gating mechanism is called Gated Spiking Units (GSU)
    """
    def __init__(self, config, split_dim: int, dim: int, spike_mode) -> None:
        super(GSU, self).__init__()
        self.split_dim = split_dim
        self.config = config
        # self.scale = 1 / dim
        if spike_mode == "lif":

            self.lif1 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq = True, backend=config.backend)
            self.lif2 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq = True, backend=config.backend)


        elif spike_mode == "plif":
            self.lif1 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                                          surrogate_function=config.surrogate_function,
                                          detach_reset=config.detach_reset,
                                          step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
            self.lif2 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                                          surrogate_function=config.surrogate_function,
                                          detach_reset=config.detach_reset,
                                          step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)


        self.proj_conv = layer.Linear(dim, dim, bias=config.bias, step_mode='m')

        if self.config.use_bn:
            self.proj_bn = layer.BatchNorm1d(dim, step_mode='m')


    def forward(self, inputs):
        # num_heads = self.config.num_heads * self.config.split_ratio
        # outputs, gate = inputs.chunk(2, dim=self.split_dim)
        # T, B, N = gate.shape
        # gate = self.lif1(gate)
        # gate = self.proj_conv(gate)
        # if self.config.use_bn:
        #     gate = self.lif2(self.proj_bn(gate)).reshape(T, B, num_heads, N// num_heads)
        # else:
        #     gate = self.lif2(gate).reshape(T, B, num_heads, N// num_heads)
        # outputs = outputs.reshape(T, B, num_heads, N// num_heads)
        # outputs = (outputs * gate).reshape(T, B, -1)
        # return outputs
        #
        outputs, gate = inputs.chunk(2, dim=self.split_dim)
        gate = self.lif1(gate)
        gate = self.proj_conv(gate)
        if self.config.use_bn:
            gate = self.lif2(self.proj_bn(gate))
        else:
            gate = self.lif2(gate)
        outputs = outputs * gate
        return outputs


# class GSU(nn.Module):
#     """
#     The gating mechanism is called Gated Spiking Units (GSU)
#     """
#     def __init__(self, config, split_dim: int, dim: int, spike_mode) -> None:
#         super(GSU, self).__init__()
#         self.split_dim = split_dim
#         self.config = config
#         # self.scale = 1 / dim
#         if spike_mode == "lif":
#
#             self.lif1 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
#                             surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
#                             step_mode='m', decay_input=False, store_v_seq = True, backend=config.backend)
#
#             # self.lif2 = LIFNode(tau=config.init_tau, v_threshold=config.gate_v_threshold,
#             #                 surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
#             #                 step_mode='m', decay_input=False, store_v_seq = True, backend=config.backend)
#
#
#         elif spike_mode == "plif":
#             self.lif1 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
#                                           surrogate_function=config.surrogate_function,
#                                           detach_reset=config.detach_reset,
#                                           step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
#
#             self.lif2 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
#                                           surrogate_function=config.surrogate_function,
#                                           detach_reset=config.detach_reset,
#                                           step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
#
#         self.proj_conv = layer.Linear(dim, dim, bias=config.bias, step_mode='m')
#
#         if self.config.use_bn:
#             self.proj_bn = layer.BatchNorm1d(dim, step_mode='m')
#         # self.rlif = RLIF(dim, config, kernel_size=3)
#
#         # self.proj_layer2 = layer.Linear(dim, dim, bias=config.bias, step_mode='m')
#
#     def forward(self, inputs):
#
#         # Sencond Way +++
#
#         # num_heads = self.config.num_heads * self.config.split_ratio
#         # outputs, gate = inputs.chunk(2, dim=self.split_dim)
#         # T, B, N = gate.shape
#         # # gate = self.proj_conv(gate)
#         # gate = self.proj_conv(gate)
#         # if self.config.use_bn:
#         #     gate = self.proj_bn(self.lif1(gate)).reshape(T, B, num_heads, N// num_heads)
#         # else:
#         #     gate = self.lif1(gate).reshape(T, B, num_heads, N// num_heads)
#         # gate = gate.sum(dim=-1, keepdim=True) # (T,B,H,*)
#         # gate = self.lif2(gate)
#         # outputs = outputs.reshape(T, B, num_heads, N// num_heads)
#         # outputs = (outputs * gate).reshape(T, B, -1)
#         # return outputs
#
#         num_heads = self.config.num_heads * self.config.split_ratio
#         outputs, gate = inputs.chunk(2, dim=self.split_dim)
#         T, B, N = gate.shape
#         gate = self.proj_conv(gate)
#         if self.config.use_bn:
#             gate = self.lif1(self.proj_bn(gate)).reshape(T, B, num_heads, N// num_heads)
#         else:
#             gate = self.lif1(gate).reshape(T, B, num_heads, N// num_heads)
#         outputs = outputs.reshape(T, B, num_heads, N// num_heads)
#         outputs = (outputs * gate).reshape(T, B, -1)
#         return outputs



class ConformerConvModule(nn.Module):

    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            config,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
            spike_mode="lif",
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        # assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"
        self.config = config
        if self.config.use_ln:
            self.ln = nn.LayerNorm(in_channels)
        # batch, time, dim
        self.trans1 = Transpose(1,0,2)
        # batch, dim, time
        self.trans2 = Transpose(1, 2, 0)
        # time, batch ,dim
        self.trans3 = Transpose(2, 0, 1)
        self.bn3 = layer.BatchNorm1d(in_channels*expansion_factor, step_mode='m') #  self.bn1 = layer.BatchNorm1d(in_channels*expansion_factor, step_mode='m')
        self.bn2 = layer.BatchNorm1d(in_channels, step_mode='m')
        self.bn1 = layer.BatchNorm1d(in_channels, step_mode='m')
        if spike_mode == "lif":
            self.lif1 = neuron.LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)
            self.lif2 = neuron.LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                           surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                           step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
            self.lif3 = neuron.LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                           surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                           step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
        elif spike_mode == "plif":
            self.lif1 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
            self.lif2 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
            self.lif3 = ParametricLIFNode(init_tau=config.init_tau, v_threshold=config.v_threshold,
                              surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                              step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
        self.dw = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=config.use_dw_bias) # 现在的kernel size很大
        self.pw1 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)  # bias= config.bias
        self.pw2 = PointwiseConv1d(in_channels, in_channels* expansion_factor, stride=1, padding=0, bias=True) # # bias= config.bias


        if self.config.use_dp:
            self.dropout1 = layer.Dropout(dropout_p, step_mode='m')
            self.dropout2 = layer.Dropout(dropout_p, step_mode='m')
            self.dropout3 = layer.Dropout(dropout_p, step_mode='m')
        self.gsu = GSU(self.config, split_dim=-1, dim=in_channels, spike_mode=spike_mode)

    def forward(self, inputs):
        if self.config.use_ln:
            #  batch, time, dim
            x = self.trans1(inputs)
            x = self.ln(x)
            # time, batch, dim
            x = self.trans1(x)
        else:
            x = inputs
        # time, batch, dim
        x = self.trans2(x)
        # batch, dim, time
        x = self.pw1(x)
        # time, batch, dim*2
        x = self.trans3(x)
        x = self.bn1(x)

        x = self.lif1(x)
        if self.config.use_dp:
            x = self.dropout1(x)

        # batch, dim, time
        x = self.trans2(x)
        x = self.dw(x)
        # time, batch, dim
        x = self.trans3(x)
        x = self.bn2(x)
        x = self.lif2(x)
        if self.config.use_dp:
            x = self.dropout2(x)

        # batch, dim, time
        x = self.trans2(x)
        x = self.pw2(x)
        # time, batch, dim
        x = self.trans3(x)
        x = self.bn3(x)
        x = self.gsu(x) # 考虑glu和bn1换顺序
        x = self.lif3(x)
        if self.config.use_dp:
            x = self.dropout3(x)

        return x


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        config,
        dim,
        num_heads,
        init_tau=2.0,
        spike_mode="lif",
        layers=0,
        # norm_first = False
    ):
        super().__init__()
        self.config = config

        # SDSA
        self.attn = MS_SSA(
            dim,
            config,
            init_tau = init_tau,
            num_heads=num_heads,
            spike_mode=spike_mode,
            layers=layers,
        )

        mlp_hidden_dim = config.hidden_dims
        # MLP
        self.mlp1 = MS_MLP(
            config,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_mode=spike_mode,
        )
        self.mlp2 = MS_MLP(
            config,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            spike_mode=spike_mode,
        )

        self.conformer_module = ConformerConvModule(
            config,
            in_channels=dim,
            kernel_size=config.kernel_size,
            expansion_factor=2,
            dropout_p=config.dropout_l,
            spike_mode=config.spike_mode
        )


    def forward(self, x,attention_mask=None):

        # Attention with residual
        attn_output = self.attn(x, attention_mask=attention_mask)
        # if visual is need
        # attn_output, attn_weights = self.attn(x, attention_mask=attention_mask)
        x = x + attn_output

        # x = self.rlif1(x)

        # First MLP with residual
        mlp1_output = self.mlp1(x)
        x = x + mlp1_output

        # x = self.rlif2(x)

        # Convolution module with residual
        conv_output = self.conformer_module(x)
        x = x + conv_output

        # x = self.rlif3(x)

        # Second MLP with residual
        mlp2_output = self.mlp2(x)
        x = x + mlp2_output

        # x = self.rlif4(x)

        return x

        # if visual is need
        # return x, attn_weights