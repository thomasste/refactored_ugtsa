#include "refactored_ugtsa/algorithms/ugtsa_algorithm.h"

#include <iostream>

namespace ugtsa {
namespace algorithms {

std::array<long long int, 2> UGTSAAlgorithm::Seed() {
    return { distribution(generator), distribution(generator) };
}

UGTSAAlgorithm::UGTSAAlgorithm(GameState *game_state, unsigned seed, int grow_factor, TensorflowWrapper *tensorflow_wrapper, bool training)
        : MCTSAlgorithm(game_state, seed, grow_factor), tensorflow_wrapper(tensorflow_wrapper), training(training) {}

std::string UGTSAAlgorithm::DebugString() {
    return MCTSAlgorithm::DebugString() + "\n" +
        "UGTSA statistics size: " + std::to_string(statistics.size()) + "\n" +
        "UGTSA updates size: " + std::to_string(updates.size()) + "\n" +
        "UGTSA move_rates size: " + std::to_string(move_rates.size()) + "\n" +
        "UGTSA zeros - statistics: " + std::to_string(zero_statistic_outputs_count) + "/" + std::to_string(zero_statistic_gradients_count) + "/" + std::to_string(statistics_count) + "\n" +
        "UGTSA zeros - updates: " + std::to_string(zero_update_outputs_count) + "/" + std::to_string(zero_update_gradients_count) + "/" + std::to_string(updates_count) + "\n" +
        "UGTSA zeros - modified_statistics: " + std::to_string(zero_modified_statistic_outputs_count) + "/" + std::to_string(zero_modified_statistic_gradients_count) + "/" + std::to_string(modified_statistics_count) + "\n" +
        "UGTSA zeros - modified_updates: " + std::to_string(zero_modified_update_outputs_count) + "/" + std::to_string(zero_modified_update_gradients_count) + "/" + std::to_string(modified_updates_count) + "\n" +
        "UGTSA zeros - move_rates: " + std::to_string(zero_move_rate_outputs_count) + "/" + std::to_string(zero_move_rate_gradients_count) + "/" + std::to_string(move_rates_count);
}

Eigen::VectorXf UGTSAAlgorithm::Value(int move_rate) {
    return move_rates[move_rate].value;
}

int UGTSAAlgorithm::Statistic() {
    order.push_back(Type::STATISTIC);
    boards.push_back(game_state->Board());
    game_state_infos.push_back(game_state->Info());
    statistics.push_back({Type::STATISTIC, Seed(), {}, (int) boards.size() - 1, (int) game_state_infos.size() - 1});
    statistics.back().value = tensorflow_wrapper->Statistic(
        statistics.back().seed,
        training,
        VectorMatrixXf{boards.back()},
        VectorVectorXf{game_state_infos.back()})[0];
    // if ((statistics.back().value.array() == 0.).all()) zero_statistic_outputs_count++;
    return statistics.size() - 1;
}

int UGTSAAlgorithm::Update() {
    order.push_back(Type::UPDATE);
    payoffs.push_back(game_state->LightPlayoutPayoff());
    updates.push_back({Type::UPDATE, Seed(), {}, (int) payoffs.size() - 1, -100});
    updates.back().value = tensorflow_wrapper->Update(
        updates.back().seed,
        training,
        VectorVectorXf{payoffs.back()})[0];
    // if ((updates.back().value.array() == 0.).all()) zero_update_outputs_count++;
    return updates.size() - 1;
}

int UGTSAAlgorithm::ModifiedStatistic(int statistic, int update) {
    order.push_back(Type::MODIFIED_STATISTIC);
    statistics.push_back({Type::MODIFIED_STATISTIC, Seed(), {}, statistic, update});
    statistics.back().value = tensorflow_wrapper->ModifiedStatistic(
        statistics.back().seed,
        training,
        VectorVectorXf{statistics[statistic].value},
        std::vector<VectorVectorXf>{{updates[update].value}})[0];
    // if ((statistics.back().value.array() == 0.).all()) zero_modified_statistic_outputs_count++;
    return statistics.size() - 1;
}

int UGTSAAlgorithm::ModifiedUpdate(int update, int statistic) {
    order.push_back(Type::MODIFIED_UPDATE);
    updates.push_back({Type::MODIFIED_UPDATE, Seed(), {}, update, statistic});
    updates.back().value = tensorflow_wrapper->ModifiedUpdate(
        updates.back().seed,
        training,
        {updates[update].value},
        {statistics[statistic].value})[0];
    // if ((updates.back().value.array() == 0.).all()) zero_modified_update_outputs_count++;
    return updates.size() - 1;
}

int UGTSAAlgorithm::MoveRate(int parent_statistic, int child_statistic) {
    order.push_back(Type::MOVE_RATE);
    move_rates.push_back({Type::MOVE_RATE, Seed(), {}, parent_statistic, child_statistic});
    move_rates.back().value = tensorflow_wrapper->MoveRate(
        move_rates.back().seed,
        training,
        {statistics[parent_statistic].value},
        {statistics[child_statistic].value})[0];
    // if ((move_rates.back().value.array() == 0.).all()) zero_move_rate_outputs_count++;
    return move_rates.size() - 1;
}

VectorVectorXf UGTSAAlgorithm::UntrackedMoveRates(const std::vector<int> &parent_statistics, const std::vector<int> &child_statistics) {
    auto parent_statistic_values = VectorVectorXf();
    auto child_statistic_values = VectorVectorXf();
    for (auto x : parent_statistics) parent_statistic_values.push_back(statistics[x].value);
    for (auto x : child_statistics) child_statistic_values.push_back(statistics[x].value);
    return tensorflow_wrapper->MoveRate(
        Seed(),
        false,
        parent_statistic_values,
        child_statistic_values);
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
                {
                    statistics_count++;
                    // if (((sgit->array() != 0.).any())) {
                        tensorflow_wrapper->BackpropagateStatistic(
                            sit->seed,
                            training,
                            VectorMatrixXf{boards[sit->board]},
                            VectorVectorXf{game_state_infos[sit->game_state_info]},
                            VectorVectorXf{*sgit});
                    // } else {
                    //     zero_statistic_gradients_count++;
                    // }
                    sit++;
                    sgit++;
                }
                break;
            case Type::UPDATE:
                {
                    updates_count++;
                    // if (((ugit->array() != 0.).any())) {
                        tensorflow_wrapper->BackpropagateUpdate(
                            uit->seed,
                            training,
                            VectorVectorXf{payoffs[uit->payoff]},
                            VectorVectorXf{*ugit});
                    // } else {
                    //     zero_update_gradients_count++;
                    // }
                    uit++;
                    ugit++;
                }
                break;
            case Type::MODIFIED_STATISTIC:
                {
                    modified_statistics_count++;
                    // if (((sgit->array() != 0.).any())) {
                        auto result = tensorflow_wrapper->BackpropagateModifiedStatistic(
                            sit->seed,
                            training,
                            VectorVectorXf{statistics[sit->statistic].value},
                            std::vector<VectorVectorXf>{{updates[sit->update].value}},
                            VectorVectorXf{*sgit});
                        statistic_gradients[sit->statistic] += result.first[0];
                        update_gradients[sit->update] += result.second[0][0];
                    // } else {
                    //     zero_modified_statistic_gradients_count++;
                    // }
                    sit++;
                    sgit++;
                }
                break;
            case Type::MODIFIED_UPDATE:
                {
                    modified_updates_count++;
                    // if (((ugit->array() != 0.).any())) {
                        auto result = tensorflow_wrapper->BackpropagateModifiedUpdate(
                            uit->seed,
                            training,
                            VectorVectorXf{updates[uit->update].value},
                            VectorVectorXf{statistics[uit->statistic].value},
                            VectorVectorXf{*ugit});
                        update_gradients[uit->update] += result.first[0];
                        statistic_gradients[uit->statistic] += result.second[0];
                    // } else {
                    //     zero_modified_update_gradients_count++;
                    // }
                    uit++;
                    ugit++;
                }
                break;
            case Type::MOVE_RATE:
                {
                    move_rates_count++;
                    // if (((mgit->array() != 0.).any())) {
                        auto result = tensorflow_wrapper->BackpropagateMoveRate(
                            mit->seed,
                            training,
                            VectorVectorXf{statistics[mit->parent_statistic].value},
                            VectorVectorXf{statistics[mit->child_statistic].value},
                            VectorVectorXf{*mgit});
                        statistic_gradients[mit->parent_statistic] += result.first[0];
                        statistic_gradients[mit->child_statistic] += result.second[0];
                    // } else {
                    //     zero_move_rate_gradients_count++;
                    // }
                    mit++;
                    mgit++;
                }
                break;
        }
    }
}

}
}