#pragma once

#include "refactored_ugtsa/algorithms/mcts_algorithm.h"
#include "refactored_ugtsa/common/tensorflow_wrapper.h"
#include "refactored_ugtsa/common/typedefs.h"

using namespace ugtsa::common;

namespace ugtsa {
namespace algorithms {

class UGTSAAlgorithm : public MCTSAlgorithm {
private:
    enum Type {
        INITIAL,
        MODIFIED
    };

    struct Statistic_ {
        Type type;

        std::array<long long int, 2> seed;
        Eigen::VectorXf value;

        union {
            int board;
            int statistic;
        };

        union {
            int game_state_info;
            int update;
        };
    };

    struct Update_ {
        Type type;

        std::array<long long int, 2> seed;
        Eigen::VectorXf value;

        union {
            int payoff;
            int update;
        };

        int statistic;
    };

    struct MoveRate_ {
        std::array<long long int, 2> seed;
        Eigen::VectorXf value;

        int parent_statistic;
        int child_statistic;
    };

    TensorflowWrapper *tensorflow_wrapper;
    bool training;

    VectorMatrixXf boards;
    VectorVectorXf game_state_infos;
    VectorVectorXf payoffs;

    std::vector<Statistic_> statistics;
    std::vector<Update_> updates;
    std::vector<MoveRate_> move_rates;

    std::uniform_int_distribution<long long int> distribution;

    std::array<long long int, 2> Seed();

public:
    UGTSAAlgorithm(GameState *game_state, unsigned seed, int grow_factor, TensorflowWrapper *tensorflow_wrapper, bool training);

    Eigen::VectorXf Value(int move_rate);
    int Statistic();
    int Update();
    int ModifiedStatistic(int statistic, int update);
    int ModifiedUpdate(int update, int statistic);
    int MoveRate(int parent_statistic, int child_statistic);
};

}
}