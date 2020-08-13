#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: define normalization api  
from ...fluid.layers import l2_normalize  #DEFINE_ALIAS
from paddle.fluid.layer_helper import LayerHelper
from ...fluid.data_feeder import check_variable_and_dtype

__all__ = [
    #       'batch_norm',
    #       'data_norm',
    #       'group_norm',
    #       'instance_norm',
    'l2_normalize',
    #       'layer_norm',
    'local_response_norm',
    #       'spectral_norm'
]


def local_response_norm(x,
                        size=5,
                        alpha=1e-4,
                        beta=0.75,
                        k=1.0,
                        data_format='NCHW',
                        name=None):
    """
    :alias_main: paddle.nn.functional.local_response_norm
	:alias: paddle.nn.functional.local_response_norm,paddle.nn.functional.norm.local_response_norm
	:old_api: paddle.fluid.layers.lrn

    This operator implements the Local Response Normalization Layer.
    This layer performs a type of "lateral inhibition" by normalizing over local input regions.
    For more information, please refer to `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_
    The formula is as follows:
    .. math::
        Output(i, x, y) = Input(i, x, y) / \\left(k + \\alpha \\sum\\limits^{\\min(C-1, i + n/2)}_{j = \\max(0, i - n/2)}(Input(j, x, y))^2\\right)^{\\beta}
    In the above equation:
    - :math:`n` : The number of channels to sum over.
    - :math:`k` : The offset (avoid being divided by 0).
    - :math:`\\alpha` : The scaling parameter.
    - :math:`\\beta` : The exponent parameter.
    Args:
        x (Variable): Input feature, 4D-Tensor with the shape of [N,C,H,W] or [N, H, W, C],
            where N is the batch size, C is the input channel, H is Height, W is weight. The data
            type is float32. The rank of this tensor must be 4, otherwise it will raise ValueError.
        size (int, optional): The number of channels to sum over. Default: 5
        alpha (float, optional): The scaling parameter, positive. Default:1e-4
        beta (float, optional): The exponent, positive. Default:0.75
        k (float, optional): An offset, positive. Default: 1.0
        data_format (str, optional): Specify the data format of the x, and the data format of the output
            will be consistent with that of the x. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.
        name (str, optional): The default value is None. Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Variable: A tensor variable storing the transformation result with the same shape and data type as input.
    Examples:
    .. code-block:: python
        import paddle
        import numpy as np
        from paddle.fluid.dygraph.base import to_variable

        paddle.enable_imperative()
        data = np.random.random(size=(2, 3, 4, 5)).astype('float32')
        data = to_variable(data)
        y = paddle.nn.functional.local_response_norm(x=data)
        print(y.numpy().shape)  # [2, 3, 4, 5]
        print(y.numpy().dtype)  # float32
        print(y.numpy())
    """
    helper = LayerHelper('local_response_norm', **locals())
    check_variable_and_dtype(x, 'x', ['float32'], 'local_response_norm')
    dtype = helper.input_dtype()
    input_shape = x.shape
    dims = len(input_shape)

    if dims != 4:
        raise ValueError(
            "X's dimension size of Op(local_response_norm) must be 4, but received %d."
            % (dims))
    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Attr(data_format) of Op(local_response_norm) got wrong value: received "
            + data_format + " but only NCHW or NHWC supported.")

    mid_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    lrn_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="lrn",
        inputs={"X": x},
        outputs={
            "Out": lrn_out,
            "MidOut": mid_out,
        },
        attrs={
            "n": size,
            "k": k,
            "alpha": alpha,
            "beta": beta,
            "data_format": data_format
        })

    return lrn_out
