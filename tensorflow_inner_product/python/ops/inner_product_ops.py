# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Use inner_product ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

inner_product_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_inner_product_ops.so'))

inner_product_grad = inner_product_ops.inner_product_grad

@ops.RegisterGradient("InnerProduct")
def _inner_product_grad_cc(op, grad):
    """
    The gradient for `inner_product` using the operation implemented in C++.
    
    :param op: `inner_product` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    :param grad: gradient with respect to the output of the `inner_product` op.
    :return: gradients with respect to the input of `inner_product`.
    """
    return inner_product_ops.inner_product_grad(grad, op.inputs[0], op.inputs[1])

inner_product = inner_product_ops.inner_product


# ..or maybe here register gradient
