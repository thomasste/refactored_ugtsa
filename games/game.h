#pragma once

#include "third_party/eigen3/Eigen/Core"

#include <random>

namespace ugtsa {
namespace games {

class GameState {
public:
    std::default_random_engine generator;
    int player_count;
    int player;

    GameState(unsigned seed, int player_count, int player);

    Eigen::VectorXf LightPlayoutPayoff();
    void MoveToRandomState();

    virtual int MoveCount() = 0;
    virtual void ApplyMove(int i) = 0;
    virtual void UndoMove() = 0;

    virtual bool IsFinal() = 0;

    virtual Eigen::MatrixXf Board() = 0;
    virtual Eigen::VectorXf Info() = 0;
    virtual Eigen::VectorXf Payoff() = 0;

    virtual void PrintDebugInfo() = 0;
};

}
}