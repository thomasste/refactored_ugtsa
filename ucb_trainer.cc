//#include "refactored_ugtsa/algorithms/ucb_algorithm.h"
//#include "refactored_ugtsa/algorithms/ugtsa_algorithm.h"
//#include "refactored_ugtsa/games/omringa.h"
#include "refactored_ugtsa/common/tensorflow_wrapper.h"
#include "refactored_ugtsa/common/typedefs.h"
//#include "refactored_ugtsa/computation_graphs/basic_computation_graph.h"

#include <iostream>
#include <string>

using namespace ugtsa::common;

int main(int argc, char **argv) {
    auto graph_name = std::string(argv[1]);
    auto ugtsa_strength = std::atoi(argv[2]);
    auto ucb_strength_multiplier = std::atoi(argv[3]);

    auto tensorflow_wrapper = TensorflowWrapper(graph_name, -1);

    std::cout << "step: " << tensorflow_wrapper.EvalIntScalar(TensorflowWrapper::STEP_TENSOR) << std::endl;

    // auto game_state = OmringaGameState();
    // game_state.MoveToRandomGameState();
    // if (game_state.IsFinal()) {
    //     game_state.UndoMove();
    // }

    // auto ucb_algorithm = UCBAlgorithm(&game_state, 5, std::sqrt(2.));
    // auto computation_graph = BasicComputationGraph(&tensorflow_wrapper);
    // auto ugtsa_algorithm = UGTSAAlgorithm(&game_state, 5, &tensorflow_wrapper, &computation_graph);

    // VectorVectorXf labels;
    // std::vector<int> ugtsa_move_rates;
    // VectorVectorXf logits;
    // for (int i = 0; i < ugtsa_strength; i++) {
    //     for (int j = 0; j < ucb_strength_multiplier; j++) {
    //         ucb_algorithm.Improve();
    //     }
    //     labels.push_back(ucb_algorithm.Value(ucb_algorithm.MoveRates()));

    //     ugtsa_algorithm.Improve();
    //     ugtsa_move_rates.push_back(ugtsa_algorithm.MoveRates());
    //     logits.push_back(ugtsa_algorithm.Value(ugtsa_move_rates.back()));
    // }

    auto a = Eigen::VectorXf(2);
    auto b = Eigen::VectorXf(2);
    auto c = Eigen::VectorXf(2);
    auto d = Eigen::VectorXf(2);
    a(0) = 1.;
    a(1) = 2.;
    b(0) = 2.;
    b(1) = 1.;
    c(0) = 2.;
    c(1) = -1.;
    d(0) = 1.;
    d(1) = 1.;
    auto logits = VectorVectorXf({a, b});
    auto labels = VectorVectorXf({c, d});

    auto loss = tensorflow_wrapper.CostFunction(logits, labels);
    std::cout << "loss: " << loss << std::endl;

    auto untrainable_model = tensorflow_wrapper.GetUntrainableModel();
    tensorflow_wrapper.ZeroGradientAccumulators();
    auto logits_gradients = tensorflow_wrapper.BackpropagateCostFunction(logits, labels);
    for (auto logit_gradient : logits_gradients) {
        std::cout << logit_gradient << std::endl;
    }
    //computation_graph.Backpropagate(ugtsa_move_rates, logits_gradients);
    tensorflow_wrapper.ApplyGradients();
    tensorflow_wrapper.SetUntrainableModel(untrainable_model);

    tensorflow_wrapper.SaveModel();

    return 0;
}