#include "refactored_ugtsa/algorithms/algorithm.h"

namespace ugtsa {
namespace algorithms {

Algorithm::Algorithm(GameState *game_state)
        : game_state(game_state) {}

int Algorithm::BestMove() {
    assert(!game_state->IsFinal());
    assert(game_state->player != -1);

    int best_move = -1;
    float best_rate = -std::numeric_limits<float>::infinity();

    auto move_rates = MoveRates();

    for (int i = 0; i < move_rates.size(); i++) {
        auto move_rate = Value(move_rates[i]);
        if (best_rate < move_rate(game_state->player)) {
            best_rate = move_rate(game_state->player);
            best_move = i;
        }
    }

    return best_move;
}

}
}