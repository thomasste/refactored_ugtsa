#include "refactored_ugtsa/algorithms/ucb_algorithm.h"
#include "refactored_ugtsa/games/game.h"

using namespace ugtsa::algorithms;

namespace ugtsa {
namespace games {

GameState::GameState(unsigned seed, int player_count, int player)
        : generator(seed), player_count(player_count), player(player) {}

Eigen::VectorXf GameState::LightPlayoutPayoff() {
    int counter = 0;

    while (!IsFinal()) {
        auto move = std::uniform_int_distribution<int>(0, MoveCount() - 1)(generator);
        ApplyMove(move);
        counter++;
    }

    auto payoff = Payoff();

    for (int i = 0; i < counter; i++) {
        UndoMove();
    }

    return payoff;
}

void GameState::MoveToRandomState() {
    int counter = 0;

    while (!IsFinal()) {
        if (player == -1) {
            ApplyMove(std::uniform_int_distribution<int>(0, MoveCount() - 1)(generator));
        } else {
            auto ucb_algorithm = UCBAlgorithm(this, std::uniform_int_distribution<int>()(generator), 5, std::sqrt(2.));
            for (int i = 0; i < 10000; i++) {
                ucb_algorithm.Improve();
            }
            ApplyMove(ucb_algorithm.BestMove());
        }
        counter++;
    }

    int undo_times = std::uniform_int_distribution<int>(0, counter)(generator);

    for (int i = 0; i < undo_times; i++) {
        UndoMove();
    }
}

}
}