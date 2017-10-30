#pragma once

#include "refactored_ugtsa/games/game.h"

#include <vector>

namespace ugtsa {
namespace algorithms {

using namespace ugtsa::games;

class Algorithm {
protected:
    GameState *game_state;

public:
    Algorithm(GameState *game_state);

    int BestMove();

    virtual void Improve() = 0;
    virtual std::vector<int> MoveRates() = 0;
    virtual Eigen::VectorXf Value(int move_rate) = 0;
};

}
}