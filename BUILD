# Change to the original code from https://github.com/tensorflow/custom-op: added the line with inner_product_py.
sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.sh"],
    data = [
        "LICENSE",
        "MANIFEST.in",
        "setup.py",
        "//tensorflow_zero_out:zero_out_py",
        "//tensorflow_time_two:time_two_py",
        "//tensorflow_inner_product:inner_product_py"
    ],
)
