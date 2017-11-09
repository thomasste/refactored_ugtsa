#include "refactored_ugtsa/algorithms/ucb_algorithm.h"
#include "refactored_ugtsa/algorithms/ugtsa_algorithm.h"
#include "refactored_ugtsa/games/omringa.h"
#include "refactored_ugtsa/common/tensorflow_wrapper.h"
#include "refactored_ugtsa/common/typedefs.h"

#include <iostream>
#include <string>

using namespace ugtsa::algorithms;
using namespace ugtsa::common;
using namespace ugtsa::games;

int main(int argc, char **argv) {
    auto graph_name = std::string(argv[1]);
    auto ugtsa_strength = std::atoi(argv[2]);
    auto ucb_strength_multiplier = std::atoi(argv[3]);
    auto use_heavy_playouts = (bool) std::atoi(argv[4]);

    auto tensorflow_wrapper = TensorflowWrapper(graph_name, -1);

    std::cout << "step: " << tensorflow_wrapper.EvalIntScalar(TensorflowWrapper::STEP_TENSOR) << std::endl;

    int seed = time(0);

    auto game_state = OmringaGameState(seed);
    game_state.MoveToRandomState();
    if (game_state.IsFinal()) {
        game_state.UndoMove();
    }

    std::cout << game_state.Info() << std::endl;
    std::cout << game_state.Board() << std::endl;

    auto ucb_algorithm = UCBAlgorithm(&game_state, seed, 5, std::sqrt(2.));
    auto ugtsa_algorithm = UGTSAAlgorithm(&game_state, seed, 5, &tensorflow_wrapper, true, use_heavy_playouts);

    VectorVectorXf labels;
    std::vector<int> ugtsa_move_rates;
    VectorVectorXf logits;
    for (int i = 0; i < ugtsa_strength; i++) {
        //std::cout << "iteration: " << i << std::endl;
        for (int j = 0; j < ucb_strength_multiplier; j++) {
            ucb_algorithm.Improve();
        }
        ugtsa_algorithm.Improve();

        for (int move_rate : ucb_algorithm.MoveRates()) {
            labels.push_back(ucb_algorithm.Value(move_rate));
        }
        for (int move_rate : ugtsa_algorithm.MoveRates()) {
            ugtsa_move_rates.push_back(move_rate);
            logits.push_back(ugtsa_algorithm.Value(move_rate));
        }
    }

    for (int i = 0; i < 20 && i < labels.size(); i++) {
        std::cout << labels[labels.size() - i - 1](0) << " " << labels[labels.size() - i - 1](1) << ", ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 20 && i < logits.size(); i++) {
        std::cout << logits[logits.size() - i - 1](0) << " " << logits[logits.size() - i - 1](1) << ", ";
    }
    std::cout << std::endl;

    auto loss = tensorflow_wrapper.CostFunction(logits, labels);
    std::cout << "loss: " << loss << std::endl;


    auto untrainable_model = tensorflow_wrapper.GetUntrainableModel();
    tensorflow_wrapper.ZeroGradientAccumulators();
    auto logits_gradients = tensorflow_wrapper.BackpropagateCostFunction(logits, labels);
    ugtsa_algorithm.Backpropagate(ugtsa_move_rates, logits_gradients);
    tensorflow_wrapper.ApplyGradients();
    tensorflow_wrapper.SetUntrainableModel(untrainable_model);
    tensorflow_wrapper.SaveModel();

    std::cout << ucb_algorithm.DebugString() << std::endl;
    std::cout << ugtsa_algorithm.DebugString() << std::endl;

    return 0;
}