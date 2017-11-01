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
    order.push_back(Type::STATISTIC);
    boards.push_back(game_state->Board());
    game_state_infos.push_back(game_state->Info());
    statistics.push_back(Statistic_{Seed(), {}, (int) boards.size() - 1, (int) game_state_infos.size() - 1});
    statistics.back().value = tensorflow_wrapper->Statistic(
        statistics.back().seed,
        training,
        VectorMatrixXf{boards.back()},
        VectorVectorXf{game_state_infos.back()})[0];
    return statistics.size() - 1;
}

int UGTSAAlgorithm::Update() {
    order.push_back(Type::UPDATE);
    payoffs.push_back(game_state->LightPlayoutPayoff());
    updates.push_back({Seed(), {}, (int) payoffs.size() - 1, -100});
    updates.back().value = tensorflow_wrapper->Update(
        updates.back().seed,
        training,
        VectorVectorXf{payoffs.back()})[0];
    return updates.size() - 1;
}

int UGTSAAlgorithm::ModifiedStatistic(int statistic, int update) {
    order.push_back(Type::MODIFIED_STATISTIC);
    statistics.push_back({Seed(), {}, statistic, update});
    statistics.back().value = tensorflow_wrapper->ModifiedStatistic(
        statistics.back().seed,
        training,
        VectorVectorXf{statistics[statistic].value},
        std::vector<VectorVectorXf>{{updates[update].value}})[0];
    return statistics.size() - 1;
}

int UGTSAAlgorithm::ModifiedUpdate(int update, int statistic) {
    order.push_back(Type::MODIFIED_UPDATE);
    updates.push_back({Seed(), {}, update, statistic});
    updates.back().value = tensorflow_wrapper->ModifiedUpdate(
        updates.back().seed,
        training,
        {updates[update].value},
        {statistics[statistic].value})[0];
    return updates.size() - 1;
}

int UGTSAAlgorithm::MoveRate(int parent_statistic, int child_statistic) {
    order.push_back(Type::MOVE_RATE);
    move_rates.push_back({Seed(), {}, parent_statistic, child_statistic});
    move_rates.back().value = tensorflow_wrapper->MoveRate(
        move_rates.back().seed,
        training,
        {statistics[parent_statistic].value},
        {statistics[child_statistic].value})[0];
    return move_rates.size() - 1;
}

Eigen::VectorXf UGTSAAlgorithm::UntrackedMoveRate(int parent_statistic, int child_statistic) {
    return tensorflow_wrapper->MoveRate(
        Seed(),
        false,
        {statistics[parent_statistic].value},
        {statistics[child_statistic].value})[0];
}

void UGTSAAlgorithm::Backpropagate(const std::vector<int> &move_rates_, const VectorVectorXf &move_rate_gradients_) {
    auto statistic_gradients = VectorVectorXf();
    auto update_gradients = VectorVectorXf();
    auto move_rate_gradients = VectorVectorXf();

    for (auto &statistic : statistics) statistic_gradients.push_back(Eigen::VectorXf::Zero(statistic.value.size()));
    for (auto &update : updates) update_gradients.push_back(Eigen::VectorXf::Zero(update.value.size()));
    for (auto &move_rate : move_rates) move_rate_gradients.push_back(Eigen::VectorXf::Zero(move_rate.value.size()));

    for (int i = 0; i < move_rates_.size(); i++) {
        move_rate_gradients[move_rates_[i]] += move_rate_gradients_[i];
    }

    auto sit = statistics.rbegin();
    auto sgit = statistic_gradients.rbegin();
    auto uit = updates.rbegin();
    auto ugit = update_gradients.rbegin();
    auto mit = move_rates.rbegin();
    auto mgit = move_rate_gradients.rbegin();

    for (auto oit = order.rbegin(); oit != order.rend(); oit++) {
        switch (*oit) {
            case Type::STATISTIC:
                tensorflow_wrapper->BackpropagateStatistic(
                    sit->seed,
                    training,
                    VectorMatrixXf{boards[sit->board]},
                    VectorVectorXf{game_state_infos[sit->game_state_info]},
                    VectorVectorXf{*sgit});
                sit++;
                sgit++;
                break;
            case Type::UPDATE:
                tensorflow_wrapper->BackpropagateUpdate(
                    uit->seed,
                    training,
                    VectorVectorXf{payoffs[uit->payoff]},
                    VectorVectorXf{*ugit});
                uit++;
                ugit++;
                break;
            case Type::MODIFIED_STATISTIC:
                {
                    auto result = tensorflow_wrapper->BackpropagateModifiedStatistic(
                        sit->seed,
                        training,
                        VectorVectorXf{statistics[sit->statistic].value},
                        std::vector<VectorVectorXf>{{updates[sit->update].value}},
                        VectorVectorXf{*sgit});
                    statistic_gradients[sit->statistic] += result.first[0];
                    update_gradients[uit->update] += result.second[0][0];
                    sit++;
                    sgit++;
                }
                break;
            case Type::MODIFIED_UPDATE:
                {
                    auto result = tensorflow_wrapper->BackpropagateModifiedUpdate(
                        uit->seed,
                        training,
                        VectorVectorXf{updates[uit->update].value},
                        VectorVectorXf{statistics[uit->statistic].value},
                        VectorVectorXf{*ugit});
                    update_gradients[uit->update] += result.first[0];
                    statistic_gradients[uit->statistic] += result.second[0];
                    uit++;
                    ugit++;
                }
                break;
            case Type::MOVE_RATE:
                {
                    auto result = tensorflow_wrapper->BackpropagateMoveRate(
                        mit->seed,
                        training,
                        VectorVectorXf{statistics[mit->parent_statistic].value},
                        VectorVectorXf{statistics[mit->child_statistic].value},
                        VectorVectorXf{*mgit});
                    statistic_gradients[mit->parent_statistic] += result.first[0];
                    statistic_gradients[mit->child_statistic] += result.second[0];
                    mit++;
                    mgit++;
                }
                break;
        }
    }
}

}
}