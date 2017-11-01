#include "refactored_ugtsa/algorithms/ucb_algorithm.h"
#include "refactored_ugtsa/games/omringa.h"

#include <iostream>
#include <random>

using namespace ugtsa::algorithms;
using namespace ugtsa::games;

int main(int argc, char **argv) {
    int seed = 1;
    auto generator = std::default_random_engine(seed);
    auto game_state = OmringaGameState(seed);

    while (!game_state.IsFinal()) {
        std::cout << game_state.Info() << std::endl;
        std::cout << game_state.Board() << std::endl;
        if (game_state.player == -1) {
            game_state.ApplyMove(std::uniform_int_distribution<int>(0, game_state.MoveCount() - 1)(generator));
        } else {
            auto ucb_algorithm = UCBAlgorithm(&game_state, seed, 5, std::sqrt(2.));
            for (int i = 0; i < 500000; i++) {
                ucb_algorithm.Improve();
            }
            game_state.ApplyMove(ucb_algorithm.BestMove());
        }
    }

    std::cout << game_state.Info() << std::endl;
    std::cout << game_state.Board() << std::endl;
    std::cout << game_state.Payoff() << std::endl;

    return 0;
}
