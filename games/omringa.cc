#include "refactored_ugtsa/games/omringa.h"

namespace ugtsa {
namespace games {

const int OmringaGameState::BOARD_SIZE = 7;
const int OmringaGameState::GROUP_PENALTY = -5;
const int OmringaGameState::MIN_BET = 0;
const int OmringaGameState::MAX_BET = 9;

OmringaGameState::Move OmringaGameState::GetMove(int i) {
    Move move;
    move.state = state;
    move.player = player;

    if (state == State::BET) {
        move.value = MIN_BET + i;
        move.index = -1;
    } else if (state == State::NATURE) {
        move.value = i;
        move.index = -1;
    } else {
        move.position = empty_positions[i];
        move.index = i;
    }

    return move;
}

OmringaGameState::OmringaGameState(unsigned seed)
        : GameState(seed, 2, 0), state(State::BET), bets{-1, -1}, board(Eigen::MatrixXf::Zero(7, 7)) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            empty_positions.push_back({i, j});
        }
    }
}

int OmringaGameState::MoveCount() {
    if (state == State::BET) {
        return MAX_BET - MIN_BET;
    } else if (state == State::NATURE) {
        return 2;
    } else {
        return empty_positions.size();
    }
}

void OmringaGameState::ApplyMove(int i) {
    auto move = GetMove(i);
    move_history.push_back(move);

    if (state == State::BET) {
        bets[move.player] = move.value;
        if (bets[move.player ^ 1] == -1) {
            state = State::BET;
            player ^= 1;
        } else if (bets[0] == bets[1]) {
            state = State::NATURE;
            player = -1;
        } else {
            state = State::PLACE;
            player = bets[0] < bets[1];
        }
    } else if (state == State::NATURE) {
        state = State::PLACE;
        player = move.value;
        chosen_player = move.value;
    } else {
        std::swap(empty_positions[move.index], empty_positions.back());
        empty_positions.pop_back();
        board(move.position.y, move.position.x) = move.player + 1;
        player = move.player ^ 1;
    }
}

void OmringaGameState::UndoMove() {
    auto move = move_history.back();
    move_history.pop_back();

    state = move.state;
    player = move.player;
    if (move.state == State::BET) {
        bets[move.player] = -1;
    } else if (move.state == State::NATURE) {
        chosen_player = -1;
    } else {
        empty_positions.push_back(move.position);
        std::swap(empty_positions[move.index], empty_positions.back());
        board(move.position.y, move.position.x) = 0;
    }
}

bool OmringaGameState::IsFinal() {
    return empty_positions.empty();
}

Eigen::MatrixXf OmringaGameState::Board() {
    return board;
}

Eigen::VectorXf OmringaGameState::Info() {
    Eigen::VectorXf result(2);
    result << (float) bets[0], (float) bets[1];
    return result;
}

Eigen::VectorXf OmringaGameState::Payoff() {
    // TODO: change
    Eigen::VectorXf result(2);
    result << (float) bets[0], (float) bets[1];
    return result;
}

void OmringaGameState::PrintDebugInfo() {
}


}
}
