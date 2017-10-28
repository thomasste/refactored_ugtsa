cc_library(
    name = "libugtsa",
    srcs = [
        "common/tensorflow_wrapper.cc",
    ],
    hdrs = [
        "common/tensorflow_wrapper.h",
        "common/typedefs.h",
    ],
    deps = [
        "//tensorflow/core:tensorflow",
        "//third_party/eigen3",
        "@boost//:filesystem",
    ],
)

cc_binary(
    name = "ucb_trainer",
    srcs = ["ucb_trainer.cc"],
    deps = [
        ":libugtsa",
    ],
)
