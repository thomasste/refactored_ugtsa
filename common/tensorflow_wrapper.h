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

    tf::Tensor VectorVectorXfToTensor(const std::vector<Eigen::VectorXf*> &v);
    tf::Tensor VectorVectorXfToTensor(const VectorVectorXf &v);
    VectorVectorXf TensorToVectorVectorXf(const tf::Tensor &t);
    tf::Tensor VectorXfToTensor(const Eigen::VectorXf &v);
    Eigen::VectorXf TensorToVectorXf(const tf::Tensor &t);
    std::pair<tf::Tensor, tf::Tensor> VectorVectorVectorXfToTensors(const std::vector<std::vector<Eigen::VectorXf*>> &v, int size);
    std::pair<tf::Tensor, tf::Tensor> VectorVectorVectorXfToTensors(const std::vector<VectorVectorXf> &v, int size);
    std::vector<VectorVectorXf> TensorToVectorVectorVectorXf(const tf::Tensor &t, const std::vector<std::vector<Eigen::VectorXf*>> &s);
    std::vector<VectorVectorXf> TensorToVectorVectorVectorXf(const tf::Tensor &t, const std::vector<VectorVectorXf> &s);
    tf::Tensor VectorMatrixXfToTensor(const std::vector<Eigen::MatrixXf*> &v);
    tf::Tensor VectorMatrixXfToTensor(const VectorMatrixXf &v);
    tf::Tensor BoolToTensor(bool b);
    tf::Tensor FloatToTensor(float f);
    float TensorToFloat(const tf::Tensor &t);
    tf::Tensor SeedToTensor(const std::array<long long int, 2> &seed);

public:
    TensorflowWrapper(std::string graph_name, int version);
    ~TensorflowWrapper();

    void SaveModel();
    int EvalIntScalar(std::string name);

    Eigen::VectorXf GetUntrainableModel();
    void SetUntrainableModel(const Eigen::VectorXf &model);

    void ZeroGradientAccumulators();
    void ApplyGradients();

    VectorVectorXf Statistic(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::MatrixXf*> &board, const std::vector<Eigen::VectorXf*> &game_state_info);
    VectorVectorXf Statistic(const std::array<long long int, 2> &seed, bool training, const VectorMatrixXf &board, const VectorVectorXf &game_state_info);
    void BackpropagateStatistic(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::MatrixXf*> &board, const std::vector<Eigen::VectorXf*> &game_state_info, const std::vector<Eigen::VectorXf*> &output_gradient);
    void BackpropagateStatistic(const std::array<long long int, 2> &seed, bool training, const VectorMatrixXf &board, const VectorVectorXf &game_state_info, const VectorVectorXf &output_gradient);
    VectorVectorXf Update(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::VectorXf*> &payoff);
    VectorVectorXf Update(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &payoff);
    void BackpropagateUpdate(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::VectorXf*> &payoff, const std::vector<Eigen::VectorXf*> &output_gradient);
    void BackpropagateUpdate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &payoff, const VectorVectorXf &output_gradient);
    VectorVectorXf ModifiedStatistic(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::VectorXf*> &statistic, const std::vector<std::vector<Eigen::VectorXf*>> &updates);
    VectorVectorXf ModifiedStatistic(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &statistic, const std::vector<VectorVectorXf> &updates);
    std::pair<VectorVectorXf, std::vector<VectorVectorXf>> BackpropagateModifiedStatistic(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::VectorXf*> &statistic, const std::vector<std::vector<Eigen::VectorXf*>> &updates, const std::vector<Eigen::VectorXf*> &output_gradient);
    std::pair<VectorVectorXf, std::vector<VectorVectorXf>> BackpropagateModifiedStatistic(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &statistic, const std::vector<VectorVectorXf> &updates, const VectorVectorXf &output_gradient);
    VectorVectorXf ModifiedUpdate(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::VectorXf*> &update, const std::vector<Eigen::VectorXf*> &statistic);
    VectorVectorXf ModifiedUpdate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &update, const VectorVectorXf &statistic);
    std::pair<VectorVectorXf, VectorVectorXf> BackpropagateModifiedUpdate(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::VectorXf*> &update, const std::vector<Eigen::VectorXf*> &statistic, const std::vector<Eigen::VectorXf*> &output_gradient);
    std::pair<VectorVectorXf, VectorVectorXf> BackpropagateModifiedUpdate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &update, const VectorVectorXf &statistic, const VectorVectorXf &output_gradient);
    VectorVectorXf MoveRate(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::VectorXf*> &parent_statistic, const std::vector<Eigen::VectorXf*> &child_statistic);
    VectorVectorXf MoveRate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &parent_statistic, const VectorVectorXf &child_statistic);
    std::pair<VectorVectorXf, VectorVectorXf> BackpropagateMoveRate(const std::array<long long int, 2> &seed, bool training, const std::vector<Eigen::VectorXf*> &parent_statistic, const std::vector<Eigen::VectorXf*> &child_statistic, const std::vector<Eigen::VectorXf*> &output_gradient);
    std::pair<VectorVectorXf, VectorVectorXf> BackpropagateMoveRate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &parent_statistic, const VectorVectorXf &child_statistic, const VectorVectorXf &output_gradient);
    float CostFunction(const std::vector<Eigen::VectorXf*> &logits, const std::vector<Eigen::VectorXf*> &labels);
    float CostFunction(const VectorVectorXf &logits, const VectorVectorXf &labels);
    VectorVectorXf BackpropagateCostFunction(const std::vector<Eigen::VectorXf*> &logits, const std::vector<Eigen::VectorXf*> &labels);
    VectorVectorXf BackpropagateCostFunction(const VectorVectorXf &logits, const VectorVectorXf &labels);
};

}
}