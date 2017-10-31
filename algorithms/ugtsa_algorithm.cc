#include "refactored_ugtsa/algorithms/ugtsa_algorithm.h"

#include <iostream>

namespace ugtsa {
namespace algorithms {

std::array<long long int, 2> UGTSAAlgorithm::Seed() {
    return { distribution(generator), distribution(generator) };
}

UGTSAAlgorithm::UGTSAAlgorithm(GameState *game_state, unsigned seed, int grow_factor, TensorflowWrapper *tensorflow_wrapper, bool training)
        : MCTSAlgorithm(game_state, seed, grow_factor), tensorflow_wrapper(tensorflow_wrapper), training(training) {}

Eigen::VectorXf UGTSAAlgorithm::Value(int move_rate) {
    return move_rates[move_rate].value;
}

int UGTSAAlgorithm::Statistic() {
    boards.push_back(game_state->Board());
    game_state_infos.push_back(game_state->Info());
    statistics.push_back(Statistic_{Type::INITIAL, Seed(), {}, (int) boards.size() - 1, (int) game_state_infos.size() - 1});
    statistics.back().value = tensorflow_wrapper->Statistic(
        statistics.back().seed,
        training,
        VectorMatrixXf{boards.back()},
        VectorVectorXf{game_state_infos.back()})[0];
    return statistics.size() - 1;
}

int UGTSAAlgorithm::Update() {
    payoffs.push_back(game_state->LightPlayoutPayoff());
    updates.push_back({Type::INITIAL, Seed(), {}, (int) payoffs.size() - 1, -100});
    updates.back().value = tensorflow_wrapper->Update(
        updates.back().seed,
        training,
        VectorVectorXf{payoffs.back()})[0];
    return updates.size() - 1;
}

int UGTSAAlgorithm::ModifiedStatistic(int statistic, int update) {
    statistics.push_back({Type::MODIFIED, Seed(), {}, statistic, update});
    statistics.back().value = tensorflow_wrapper->ModifiedStatistic(
        statistics.back().seed,
        training,
        VectorVectorXf{statistics[statistic].value},
        std::vector<VectorVectorXf>{{updates[update].value}})[0];
    return statistics.size() - 1;
}

int UGTSAAlgorithm::ModifiedUpdate(int update, int statistic) {
    updates.push_back({Type::MODIFIED, Seed(), {}, update, statistic});
    updates.back().value = tensorflow_wrapper->ModifiedUpdate(
        updates.back().seed,
        training,
        {updates[update].value},
        {statistics[statistic].value})[0];
    return updates.size() - 1;
}

int UGTSAAlgorithm::MoveRate(int parent_statistic, int child_statistic) {
    move_rates.push_back({Seed(), {}, parent_statistic, child_statistic});
    move_rates.back().value = tensorflow_wrapper->MoveRate(
        move_rates.back().seed,
        training,
        {statistics[parent_statistic].value},
        {statistics[child_statistic].value})[0];
    return move_rates.size() - 1;
}

}
}