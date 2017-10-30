#include "refactored_ugtsa/algorithms/mcts_algorithm.h"

namespace ugtsa {
namespace algorithms {

MCTSAlgorithm::MCTSAlgorithm(GameState *game_state, unsigned seed, int grow_factor)
        : Algorithm(game_state), generator(seed), grow_factor(grow_factor) {}

void MCTSAlgorithm::Improve() {
    if (tree.empty()) {
        tree.push_back({0, {1, game_state->MoveCount() + 1}, Statistic()});
        for (int i = 0; i < game_state->MoveCount(); i++) {
            game_state->ApplyMove(i);
            tree.push_back({0, {-1, -1}, Statistic()});
            game_state->UndoMove();
        }
    } else {
        auto state_stack = std::vector<int>({0});

        while(tree[state_stack.back()].children[0] != -1) {
            auto &node = tree[state_stack.back()];
            node.number_of_visits++;

            int move;

            if (game_state->player == -1) {
                move = std::uniform_int_distribution<int>(0, node.children[1] - node.children[0] - 1)(generator);
            } else {
                auto &parent_statistic = tree[state_stack.back()].statistic;
                auto best_rate = -std::numeric_limits<float>::infinity();

                for (auto i = node.children[0]; i < node.children[1]; i++) {
                    auto &child_statistic = tree[i].statistic;
                    auto move_rate = Value(MoveRate(parent_statistic, child_statistic))(game_state->player);
                    if (best_rate < move_rate) {
                        best_rate = move_rate;
                        move = i - node.children[0];
                    }
                }
            }
            state_stack.push_back(node.children[0] + move);
            game_state->ApplyMove(move);
        }

        auto &leaf = tree[state_stack.back()];
        leaf.number_of_visits++;

        if (!game_state->IsFinal() && leaf.number_of_visits == grow_factor) {
            leaf.children[0] = tree.size();
            leaf.children[1] = tree.size() + game_state->MoveCount();
            for (auto i = 0; i < game_state->MoveCount(); i++) {
                game_state->ApplyMove(i);
                tree.push_back({0, {-1, -1}, Statistic()});
                game_state->UndoMove();
            }
        }

        auto update = Update();

        while(!state_stack.empty()) {
            auto &node = tree[state_stack.back()];
            node.statistic = ModifiedStatistic(node.statistic, update);
            update = ModifiedUpdate(update, node.statistic);
            state_stack.pop_back();
            if (state_stack.size() > 0) {
                game_state->UndoMove();
            }
        }
    }
}

std::vector<int> MCTSAlgorithm::MoveRates() {
    std::vector<int> result;

    Node& root = tree[0];
    for (int i = root.children[0]; i < root.children[1]; i++) {
        Node &child = tree[i];
        result.push_back(MoveRate(root.statistic, child.statistic));
    }

    return result;
}

}
}