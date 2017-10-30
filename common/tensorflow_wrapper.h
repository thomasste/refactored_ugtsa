#pragma once

#include "tensorflow/core/public/session.h"
#include "refactored_ugtsa/common/typedefs.h"

#include <string>

namespace ugtsa {
namespace common {

namespace tf = tensorflow;

class TensorflowWrapper {
public:
    int worker_count;

    static const std::string STEP_TENSOR;

private:
    static const std::string GRAPHS_PATH;
    static const std::string MODELS_PATH;
    static const std::string LOAD_OP;
    static const std::string SAVE_OP;
    static const std::string PATH_TENSOR;
    static const std::string WORKER_COUNT;

    static const std::string UNTRAINABLE_MODEL;
    static const std::string SET_UNTRAINABLE_MODEL_OP;
    static const std::string SET_UNTRAINABLE_MODEL_INPUT;

    static const std::string ZERO_GRADIENT_ACCUMULATORS_OP;
    static const std::string APPLY_GRADIENTS_OP;

    static const std::string STATISTIC_SEED;
    static const std::string STATISTIC_TRAINING;
    static const std::string STATISTIC_BOARD;
    static const std::string STATISTIC_GAME_STATE_INFO;
    static const std::string STATISTIC_OUTPUT;
    static const std::string STATISTIC_OUTPUT_GRADIENT;
    static const std::string STATISTIC_UPDATE_GRADIENT_ACCUMULATORS_OP;

    static const std::string UPDATE_SEED;
    static const std::string UPDATE_TRAINING;
    static const std::string UPDATE_PAYOFF;
    static const std::string UPDATE_OUTPUT;
    static const std::string UPDATE_OUTPUT_GRADIENT;
    static const std::string UPDATE_UPDATE_GRADIENT_ACCUMULATORS_OP;

    static const std::string MODIFIED_STATISTIC_SEED;
    static const std::string MODIFIED_STATISTIC_TRAINING;
    static const std::string MODIFIED_STATISTIC_STATISTIC;
    static const std::string MODIFIED_STATISTIC_UPDATES_COUNT;
    static const std::string MODIFIED_STATISTIC_UPDATES;
    static const std::string MODIFIED_STATISTIC_OUTPUT;
    static const std::string MODIFIED_STATISTIC_OUTPUT_GRADIENT;
    static const std::string MODIFIED_STATISTIC_STATISTIC_GRADIENT;
    static const std::string MODIFIED_STATISTIC_UPDATES_GRADIENT;
    static const std::string MODIFIED_STATISTIC_UPDATE_GRADIENT_ACCUMULATORS_OP;

    static const std::string MODIFIED_UPDATE_SEED;
    static const std::string MODIFIED_UPDATE_TRAINING;
    static const std::string MODIFIED_UPDATE_UPDATE;
    static const std::string MODIFIED_UPDATE_STATISTIC;
    static const std::string MODIFIED_UPDATE_OUTPUT;
    static const std::string MODIFIED_UPDATE_OUTPUT_GRADIENT;
    static const std::string MODIFIED_UPDATE_UPDATE_GRADIENT;
    static const std::string MODIFIED_UPDATE_STATISTIC_GRADIENT;
    static const std::string MODIFIED_UPDATE_UPDATE_GRADIENT_ACCUMULATORS_OP;

    static const std::string MOVE_RATE_SEED;
    static const std::string MOVE_RATE_TRAINING;
    static const std::string MOVE_RATE_PARENT_STATISTIC;
    static const std::string MOVE_RATE_CHILD_STATISTIC;
    static const std::string MOVE_RATE_OUTPUT;
    static const std::string MOVE_RATE_OUTPUT_GRADIENT;
    static const std::string MOVE_RATE_PARENT_STATISTIC_GRADIENT;
    static const std::string MOVE_RATE_CHILD_STATISTIC_GRADIENT;
    static const std::string MOVE_RATE_UPDATE_GRADIENT_ACCUMULATORS_OP;

    static const std::string COST_FUNCTION_SEED;
    static const std::string COST_FUNCTION_TRAINING;
    static const std::string COST_FUNCTION_LOGITS;
    static const std::string COST_FUNCTION_LABELS;
    static const std::string COST_FUNCTION_OUTPUT;
    static const std::string COST_FUNCTION_OUTPUT_GRADIENT;
    static const std::string COST_FUNCTION_LOGITS_GRADIENT;
    static const std::string COST_FUNCTION_UPDATE_GRADIENT_ACCUMULATORS_OP;

    tf::Session* session;
    std::string graph_name;

    std::string GraphPath();
    std::string ModelPath(std::string model_name);
    std::string ModelName(int version);
    void LoadGraph();
    void LoadModel(int version);

    tf::Tensor VectorVectorXfToTensor(VectorVectorXf v);
    VectorVectorXf TensorToVectorVectorXf(tf::Tensor t);
    tf::Tensor VectorXfToTensor(Eigen::VectorXf v);
    Eigen::VectorXf TensorToVectorXf(tf::Tensor t);
    std::pair<tf::Tensor, tf::Tensor> VectorVectorVectorXfToTensors(std::vector<VectorVectorXf> v, int size);
    std::vector<VectorVectorXf> TensorToVectorVectorVectorXf(tf::Tensor t, std::vector<VectorVectorXf> s);
    tf::Tensor VectorMatrixXfToTensor(VectorMatrixXf v);
    tf::Tensor BoolToTensor(bool b);
    tf::Tensor FloatToTensor(float f);
    float TensorToFloat(tf::Tensor t);
    tf::Tensor SeedToTensor(std::array<long long int, 2> seed);

public:
    TensorflowWrapper(std::string graph_name, int version);
    ~TensorflowWrapper();

    void SaveModel();
    int EvalIntScalar(std::string name);

    Eigen::VectorXf GetUntrainableModel();
    void SetUntrainableModel(Eigen::VectorXf model);

    void ZeroGradientAccumulators();
    void ApplyGradients();

    VectorVectorXf Statistic(std::array<long long int, 2> seed, bool training, VectorMatrixXf board, VectorVectorXf game_state_info);
    void BackpropagateStatistic(std::array<long long int, 2> seed, bool training, VectorMatrixXf board, VectorVectorXf game_state_info, VectorVectorXf output_gradient);
    VectorVectorXf Update(std::array<long long int, 2> seed, bool training, VectorVectorXf payoff);
    void BackpropagateUpdate(std::array<long long int, 2> seed, bool training, VectorVectorXf payoff, VectorVectorXf output_gradient);
    VectorVectorXf ModifiedStatistic(std::array<long long int, 2> seed, bool training, VectorVectorXf statistic, std::vector<VectorVectorXf> updates);
    std::pair<VectorVectorXf, std::vector<VectorVectorXf>> BackpropagateModifiedStatistic(std::array<long long int, 2> seed, bool training, VectorVectorXf statistic, std::vector<VectorVectorXf> updates, VectorVectorXf output_gradient);
    VectorVectorXf ModifiedUpdate(std::array<long long int, 2> seed, bool training, VectorVectorXf update, VectorVectorXf statistic);
    std::pair<VectorVectorXf, VectorVectorXf> BackpropagateModifiedUpdate(std::array<long long int, 2> seed, bool training, VectorVectorXf update, VectorVectorXf statistic, VectorVectorXf output_gradient);
    VectorVectorXf MoveRate(std::array<long long int, 2> seed, bool training, VectorVectorXf parent_statistic, VectorVectorXf child_statistic);
    std::pair<VectorVectorXf, VectorVectorXf> BackpropagateMoveRate(std::array<long long int, 2> seed, bool training, VectorVectorXf parent_statistic, VectorVectorXf child_statistic, VectorVectorXf output_gradient);
    float CostFunction(VectorVectorXf logits, VectorVectorXf labels);
    VectorVectorXf BackpropagateCostFunction(VectorVectorXf logits, VectorVectorXf labels);
};

}
}