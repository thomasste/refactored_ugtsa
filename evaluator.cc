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

bool is_ugtsa_better(TensorflowWrapper *tensorflow_wrapper, int ugtsa_player, int ucb_strength, int ugtsa_strength, bool use_heavy_playouts) {
    auto generator = std::default_random_engine(time(0));
    auto game_state = OmringaGameState(time(0));

    while (!game_state.IsFinal()) {
        std::cout << game_state.Info() << std::endl;
        std::cout << game_state.Board() << std::endl;
        if (game_state.player == -1) {
            game_state.ApplyMove(std::uniform_int_distribution<int>(0, game_state.MoveCount() - 1)(generator));
        } else if (game_state.player == ugtsa_player) {
            auto ugtsa_algorithm = UGTSAAlgorithm(&game_state, time(0), 5, tensorflow_wrapper, true, use_heavy_playouts);
            for (int i = 0; i < ugtsa_strength; i++) ugtsa_algorithm.Improve();
            game_state.ApplyMove(ugtsa_algorithm.BestMove());
        } else {
            auto ucb_algorithm = UCBAlgorithm(&game_state, time(0), 5, std::sqrt(2.));
            for (int i = 0; i < ucb_strength; i++) ucb_algorithm.Improve();
            game_state.ApplyMove(ucb_algorithm.BestMove());
        }
    }


    auto payoff = game_state.Payoff();
    std::cout << payoff << std::endl;
    if (payoff.maxCoeff() == payoff(ugtsa_player)) return true;
    else return false;
}

int main(int argc, char **argv) {
    auto graph_name = std::string(argv[1]);
    auto ugtsa_strength = std::atoi(argv[2]);
    auto use_heavy_playouts = (bool) std::atoi(argv[3]);

    auto tensorflow_wrapper = TensorflowWrapper(graph_name, -1);
    std::cout << "step: " << tensorflow_wrapper.EvalIntScalar(TensorflowWrapper::STEP_TENSOR) << std::endl;

    int begin, end = 128, middle;
    while (is_ugtsa_better(&tensorflow_wrapper, 0, end, ugtsa_strength, use_heavy_playouts)) {
        std::cout << "at least " << end << std::endl;
        end *= 2;
    }

    begin = end / 2;

    while (begin < end) {
        std::cout << "range " << begin << " " << end << std::endl;
        middle = (begin + end) / 2;
        if (is_ugtsa_better(&tensorflow_wrapper, 0, middle, ugtsa_strength, use_heavy_playouts)) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    std::cout << "result " << end << std::endl;

    return 0;
}