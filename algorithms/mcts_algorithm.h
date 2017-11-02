#pragma once

#include "refactored_ugtsa/algorithms/algorithm.h"

namespace ugtsa {
namespace algorithms {

class MCTSAlgorithm : public Algorithm {
private:
    struct Node {
        int number_of_visits;
        int children[2];
        int statistic;
    };

protected:
    std::default_random_engine generator;

private:
    int grow_factor;

    std::vector<Node> tree;

public:
    MCTSAlgorithm(GameState *game_state, unsigned seed, int grow_factor);

    void Improve();
    std::vector<int> MoveRates();

    virtual std::string DebugString();
    virtual Eigen::VectorXf Value(int move_rate) = 0;
    virtual int Statistic() = 0;
    virtual int Update() = 0;
    virtual int ModifiedStatistic(int statistic, int update) = 0;
    virtual int ModifiedUpdate(int update, int statistic) = 0;
    virtual int MoveRate(int parent_statistic, int child_statistic) = 0;
    virtual Eigen::VectorXf UntrackedMoveRate(int parent_statistic, int child_statistic) = 0;
};

}
}