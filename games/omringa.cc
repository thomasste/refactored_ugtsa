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
    Eigen::VectorXf result(2);
    result << (float) GroupsCount(0) * GROUP_PENALTY, (float) GroupsCount(1) * GROUP_PENALTY;

    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board(y, x) == 1.) {
                result[0] += 1.;
            } else if (board(y, x) == 2.) {
                result[1] += 1.;
            }
        }
    }

    if (bets[0] < bets[1]) {
        result[0] += (float) bets[0] + 0.5;
    } else if (bets[0] > bets[1]) {
        result[1] += (float) bets[1] + 0.5;
    } else {
        result[chosen_player ^ 1] += bets[chosen_player ^ 1] + 0.5;
    }

    return result;
}

void OmringaGameState::PrintDebugInfo() {
}

int OmringaGameState::GroupsCount(int player) {
    auto groups = 0;
    auto fplayer = (float) player + 1.;
    auto stack = std::vector<Position>();
    bool visited[BOARD_SIZE][BOARD_SIZE];

    for (int i = 0; i < BOARD_SIZE; i++) for (int j = 0; j < BOARD_SIZE; j++) visited[i][j] = false;

    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (!visited[y][x] && board(y, x) == fplayer) {
                visited[y][x] = true;
                groups++;

                stack.push_back({x, y});
                while (!stack.empty()) {
                    auto p = stack.back(); stack.pop_back();

                    int dy[] = {-1, 0, 0, 1};
                    int dx[] = {0, -1, 1, 0};

                    for (int i = 0; i < 4; i++) {
                        auto np = Position({p.x + dx[i], p.y + dy[i]});

                        if (0 <= np.x && np.x < BOARD_SIZE &&
                                0 <= np.y && np.y <= BOARD_SIZE &&
                                !visited[np.y][np.x] &&
                                board(np.y, np.x) == fplayer) {
                            visited[np.y][np.x] = true;
                            stack.push_back(np);
                        }
                    }
                }
            }
        }
    }

    return groups;
}

}
}
