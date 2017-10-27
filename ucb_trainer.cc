#include "refactored_ugtsa/common/tensorflow_wrapper.h"

#include <iostream>
#include <string>

using namespace ugtsa::common;

int main(int argc, char **argv) {
    auto graph_name = std::string(argv[1]);
    auto ucb_strength = std::atoi(argv[2]);
    auto ugtsa_strength = std::atoi(argv[3]);

    auto tensorflow_wrapper = TensorflowWrapper(graph_name, -1);

    std::cout << tensorflow_wrapper.EvalIntScalar(TensorflowWrapper::STEP_TENSOR) << std::endl;

    tensorflow_wrapper.SaveModel();

    return 0;
}