import torch
import torch.nn.functional as F
from torch import nn
import warnings

from mmdet.models.builder import NECKS

def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "SyncBN": nn.SyncBatchNorm,
        }[norm]
    return norm(out_channels)

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            with warnings.catch_warnings(record=True):
                if x.numel() == 0 and self.training:
                    # https://github.com/pytorch/pytorch/issues/12013
                    assert not isinstance(
                        self.norm, torch.nn.SyncBatchNorm
                    ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def swish(x):
    return x * x.sigmoid()


class FeatureMapResampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm=""):
        super(FeatureMapResampler, self).__init__()
        if in_channels != out_channels:
            self.reduction = Conv2d(
                in_channels, out_channels, kernel_size=1,
                bias=(norm == ""),
                norm=get_norm(norm, out_channels),
                activation=None
            )
        else:
            self.reduction = None

        assert stride <= 2
        self.stride = stride

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)

        if self.stride == 2:
            x = F.max_pool2d(
                x, kernel_size=self.stride + 1,
                stride=self.stride, padding=1
            )
        elif self.stride == 1:
            pass
        else:
            raise NotImplementedError()
        return x



class SingleBiFPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, in_channels_list, out_channels, norm=""
    ):
        """
        Args:
            in_channels_list (list[int])
            out_channels (int)
            norm (str): the normalization to use.
        """
        super(SingleBiFPN, self).__init__()

        self.out_channels = out_channels
        # build 5-levels bifpn
        if len(in_channels_list) == 5:
            self.nodes = [
                {'feat_level': 3, 'inputs_offsets': [3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
                {'feat_level': 1, 'inputs_offsets': [1, 6]},
                {'feat_level': 0, 'inputs_offsets': [0, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 7, 8]},
                {'feat_level': 2, 'inputs_offsets': [2, 6, 9]},
                {'feat_level': 3, 'inputs_offsets': [3, 5, 10]},
                {'feat_level': 4, 'inputs_offsets': [4, 11]},
            ]
        elif len(in_channels_list) == 6:
            self.nodes = [
                {'feat_level': 4, 'inputs_offsets': [4, 5]},
                {'feat_level': 3, 'inputs_offsets': [3, 6]},
                {'feat_level': 2, 'inputs_offsets': [2, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 8]},
                {'feat_level': 0, 'inputs_offsets': [0, 9]},
                {'feat_level': 1, 'inputs_offsets': [1, 9, 10]},
                {'feat_level': 2, 'inputs_offsets': [2, 8, 11]},
                {'feat_level': 3, 'inputs_offsets': [3, 7, 12]},
                {'feat_level': 4, 'inputs_offsets': [4, 6, 13]},
                {'feat_level': 5, 'inputs_offsets': [5, 14]},
            ]
        elif len(in_channels_list) == 3:
            self.nodes = [
                {'feat_level': 1, 'inputs_offsets': [1, 2]},
                {'feat_level': 0, 'inputs_offsets': [0, 3]},
                {'feat_level': 1, 'inputs_offsets': [1, 3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
            ]
        else:
            raise NotImplementedError

        node_info = [_ for _ in in_channels_list]

        num_output_connections = [0 for _ in in_channels_list]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1

                in_channels = node_info[input_offset]
                if in_channels != out_channels:
                    lateral_conv = Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        norm=get_norm(norm, out_channels)
                    )
                    self.add_module(
                        "lateral_{}_f{}".format(input_offset, feat_level), lateral_conv
                    )
            node_info.append(out_channels)
            num_output_connections.append(0)

            # generate attention weights
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            self.__setattr__(name, nn.Parameter(
                    torch.ones(len(inputs_offsets), dtype=torch.float32),
                    requires_grad=True
                ))

            # generate convolutions after combination
            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            self.add_module(name, Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm=get_norm(norm, out_channels),
                bias=(norm == "")
            ))

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): feature map tensor for each feature level in high to low resolution order.
        Returns:
            list[Tensor]:
                feature map tensor in high to low resolution order. Returned feature tensor has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """
        feats = [_ for _ in feats]
        num_levels = len(feats)
        num_output_connections = [0 for _ in feats]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            input_nodes = []
            _, _, target_h, target_w = feats[feat_level].size()
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1
                input_node = feats[input_offset]

                # reduction
                if input_node.size(1) != self.out_channels:
                    name = "lateral_{}_f{}".format(input_offset, feat_level)
                    input_node = self.__getattr__(name)(input_node)

                # maybe downsample
                _, _, h, w = input_node.size()
                if h > target_h and w > target_w:
                    height_stride_size = int((h - 1) // target_h + 1)
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=(height_stride_size + 1, width_stride_size + 1),
                        stride=(height_stride_size, width_stride_size), padding=1
                    )
                elif h <= target_h and w <= target_w:
                    if h < target_h or w < target_w:
                        input_node = F.interpolate(
                            input_node,
                            size=(target_h, target_w),
                            mode="nearest"
                        )
                else:
                    raise NotImplementedError()
                input_nodes.append(input_node)

            # attention
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            weights = F.relu(self.__getattr__(name))
            norm_weights = weights / (weights.sum() + 0.0001)

            new_node = torch.stack(input_nodes, dim=-1)
            new_node = (norm_weights * new_node).sum(dim=-1)
            new_node = swish(new_node)

            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            feats.append(self.__getattr__(name)(new_node))

            num_output_connections.append(0)

        output_feats = []
        for idx in range(num_levels):
            for i, fnode in enumerate(reversed(self.nodes)):
                if fnode['feat_level'] == idx:
                    output_feats.append(feats[-1 - i])
                    break
            else:
                raise ValueError()
        return output_feats

@NECKS.register_module
class BiFPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, in_channels, out_channels, num_outs, num_repeats=6, norm=""
    ):
        """
        Args:
            in_channels (list[int])
            out_channels (int): number of channels in the output feature maps.
            num_out (int): the number of the output levels.
            num_repeats (int): the number of repeats of BiFPN.
            norm (str): the normalization to use.
        """
        super(BiFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn_convs = nn.ModuleList()
        extra_levels = self.num_outs - self.num_ins
        prev_channels = self.in_channels[-1]
        for i in range(extra_levels):
            self.fpn_convs.append(FeatureMapResampler(
                prev_channels, out_channels, 2, norm
            ))
            self.in_channels.append(out_channels)
            prev_channels = out_channels

        # build bifpn
        self.repeated_bifpn = nn.ModuleList()
        for i in range(num_repeats):
            if i == 0:
                in_channels_list = self.in_channels 
            else:
                in_channels_list = [out_channels] * len(self.in_channels)
            self.repeated_bifpn.append(SingleBiFPN(
                in_channels_list, out_channels, norm
            ))

    def forward(self, inputs):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """
        inputs = [_ for _ in inputs]
        for extra_conv in self.fpn_convs:
            inputs.append(extra_conv(inputs[-1]))

        assert len(inputs) == len(self.in_channels)

        for bifpn in self.repeated_bifpn:
             inputs = bifpn(inputs)

        return inputs


