cc_library(
    name = "libugtsa",
    srcs = [
        "algorithms/algorithm.cc",
        "algorithms/mcts_algorithm.cc",
        "algorithms/ucb_algorithm.cc",
        "algorithms/ugtsa_algorithm.cc",
        "common/tensorflow_wrapper.cc",
        "games/game.cc",
        "games/omringa.cc",
    ],
    hdrs = [
        "algorithms/algorithm.h",
        "algorithms/mcts_algorithm.h",
        "algorithms/ucb_algorithm.h",
        "algorithms/ugtsa_algorithm.h",
        "common/tensorflow_wrapper.h",
        "common/typedefs.h",
        "games/game.h",
        "games/omringa.h",
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

cc_binary(
    name = "evaluator",
    srcs = ["evaluator.cc"],
    deps = [
        ":libugtsa",
    ],
)

cc_binary(
    name = "ucb_vs_ucb",
    srcs = ["tests/ucb_vs_ucb.cc"],
    deps = [
        ":libugtsa",
    ],
)
