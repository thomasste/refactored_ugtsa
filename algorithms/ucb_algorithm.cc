#include "refactored_ugtsa/algorithms/ucb_algorithm.h"

#include <iostream>

namespace ugtsa {
namespace algorithms {

float UCBAlgorithm::UCB(float pn, float cn, float w) {
    return (w / (cn + 1.)) + exploration_factor * std::sqrt(std::log(pn + 1.) / (cn + 1.));
}

UCBAlgorithm::UCBAlgorithm(GameState *game_state, unsigned seed, int grow_factor, float exploration_factor)
        : MCTSAlgorithm(game_state, seed, grow_factor), exploration_factor(exploration_factor) {}

std::string UCBAlgorithm::DebugString() {
    return MCTSAlgorithm::DebugString() + "\n" +
        "UCB statistics size: " + std::to_string(statistics.size()) + "\n" +
        "UCB updates size: " + std::to_string(updates.size()) + "\n" +
        "UCB move_rates size: " + std::to_string(move_rates.size());
}

Eigen::VectorXf UCBAlgorithm::Value(int move_rate) {
    return move_rates[move_rate];
}

int UCBAlgorithm::Statistic() {
    statistics.push_back({0, std::vector<int>(game_state->player_count, 0)});
    return statistics.size() - 1;
}

int UCBAlgorithm::Update() {
    updates.push_back(game_state->LightPlayoutPayoff());
    return updates.size() - 1;
}

int UCBAlgorithm::ModifiedStatistic(int statistic, int update) {
    auto best_score = -std::numeric_limits<float>::infinity();
    auto best_player = -1;

    for (auto i = 0; i < game_state->player_count; i++) {
        if (best_score < updates[update](i)) {
            best_score = updates[update](i);
            best_player = i;
        }
    }

    statistics.push_back(statistics[statistic]);
    statistics.back().number_of_visits++;
    statistics.back().wins[best_player] += 1;
    return statistics.size() - 1;
}

int UCBAlgorithm::ModifiedUpdate(int update, int statistic) {
    return update;
}

int UCBAlgorithm::MoveRate(int parent_statistic, int child_statistic) {
    move_rates.push_back(Eigen::VectorXf::Zero(game_state->player_count));
    auto pn = statistics[parent_statistic].number_of_visits;
    auto cn = statistics[child_statistic].number_of_visits;
    auto &w = statistics[child_statistic].wins;

    for (int i = 0; i < game_state->player_count; i++) {
        move_rates.back()(i) = UCB((float) pn, (float) cn, (float) w[i]);
    }

    return move_rates.size() - 1;
}

VectorVectorXf UCBAlgorithm::UntrackedMoveRates(const std::vector<int> &parent_statistics, const std::vector<int> &child_statistics) {
    auto result = VectorVectorXf();
    for (int i = 0; i < parent_statistics.size(); i++) {
        result.push_back(Value(MoveRate(parent_statistics[i], child_statistics[i])));
    }
    return result;
}

}
}