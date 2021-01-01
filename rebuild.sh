#!/bin/bash
# new file not present in the original https://github.com/tensorflow/custom-op repo.
rm -R /root/.cache
pip3 uninstall tensorflow-custom-ops
./configure.sh
bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip3 install --no-cache-dir artifacts/*.whl
ls /usr/local/lib/python3.7/site-packages/tensorflow_inner_product/python/ops/
python3 tensorflow_inner_product/python/inner_product_grad_test.py
