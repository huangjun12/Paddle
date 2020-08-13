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

from ...fluid.dygraph.nn import InstanceNorm

from ...fluid.dygraph import BatchNorm  #DEFINE_ALIAS
from ...fluid.dygraph import GroupNorm  #DEFINE_ALIAS
from ...fluid.dygraph import LayerNorm  #DEFINE_ALIAS
from ...fluid.dygraph import SpectralNorm  #DEFINE_ALIAS

from ...fluid.dygraph import layers
from .. import functional as F

__all__ = [
    'BatchNorm', 'GroupNorm', 'LayerNorm', 'SpectralNorm', 'InstanceNorm',
    'LocalResponseNorm'
]


class LocalResponseNorm(layers.Layer):
    """
    :alias_main: paddle.nn.LocalResponseNorm
	:alias: paddle.nn.LocalResponseNorm,paddle.nn.layer.LocalResponseNorm,paddle.nn.layer.norm.LocalResponseNorm
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
        m = paddle.nn.LocalResponseNorm()
        y = m(data)
        print(y.numpy().shape)  # [2, 3, 4, 5]
        print(y.numpy().dtype)  # float32
        print(y.numpy())
    """

    def __init__(self,
                 size=5,
                 alpha=1e-4,
                 beta=0.75,
                 k=1.0,
                 data_format='NCHW',
                 name=None):
        super(LocalResponseNorm, self).__init__()

        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.data_format = data_format
        self.name = name

    def forward(self, input):
        out = F.local_response_norm(
            input,
            size=self.size,
            alpha=self.alpha,
            beta=self.beta,
            k=self.k,
            data_format=self.data_format,
            name=self.name)
        return out
