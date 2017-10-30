#pragma once

#include "refactored_ugtsa/games/game.h"

namespace ugtsa {
namespace games {

class OmringaGameState : public GameState {
private:
    static const int BOARD_SIZE;
    static const int GROUP_PENALTY;
    static const int MIN_BET;
    static const int MAX_BET;

    struct Position {
        int x;
        int y;
    };

    enum State {
        BET,
        NATURE,
        PLACE
    };

    struct Move {
        State state;
        int player;
        Position position;
        int value;
        int index;
    };

    State state;
    int bets[2];
    int chosen_player;
    Eigen::MatrixXf board;
    std::vector<Position> empty_positions;
    std::vector<Move> move_history;

    Move GetMove(int i);

public:
    OmringaGameState(unsigned seed);

    int MoveCount();
    void ApplyMove(int i);
    void UndoMove();

    bool IsFinal();

    Eigen::MatrixXf Board();
    Eigen::VectorXf Info();
    Eigen::VectorXf Payoff();

    void PrintDebugInfo();
};

}
}