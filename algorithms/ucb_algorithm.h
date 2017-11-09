#pragma once

#include "refactored_ugtsa/algorithms/mcts_algorithm.h"
#include "refactored_ugtsa/common/typedefs.h"

using namespace ugtsa::common;

namespace ugtsa {
namespace algorithms {

class UCBAlgorithm : public MCTSAlgorithm {
private:
    struct Statistic_ {
        int number_of_visits;
        std::vector<int> wins;
    };

    float exploration_factor;

    std::vector<Statistic_> statistics;
    VectorVectorXf updates;
    VectorVectorXf move_rates;

    float UCB(float pn, float cn, float w);

public:
    UCBAlgorithm(GameState *game_state, unsigned seed, int grow_factor, float exploration_factor);

    std::string DebugString();
    Eigen::VectorXf Value(int move_rate);
    int Statistic();
    int Update();
    int ModifiedStatistic(int statistic, int update);
    int ModifiedUpdate(int update, int statistic);
    int MoveRate(int parent_statistic, int child_statistic);
    VectorVectorXf UntrackedMoveRates(const std::vector<int> &parent_statistics, const std::vector<int> &child_statistics);
};

}
}