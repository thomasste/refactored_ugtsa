#include "tensorflow/core/public/session.h"
#include "refactored_ugtsa/common/typedefs.h"

#include <string>

namespace ugtsa {
namespace common {

namespace tf = tensorflow;

class TensorflowWrapper {
public:
    static const std::string STEP_TENSOR;

private:
    static const std::string GRAPHS_PATH;
    static const std::string MODELS_PATH;
    static const std::string LOAD_OP;
    static const std::string SAVE_OP;
    static const std::string PATH_TENSOR;

    static const std::string UNTRAINABLE_MODEL;
    static const std::string SET_UNTRAINABLE_MODEL_OP;
    static const std::string SET_UNTRAINABLE_MODEL_INPUT;

    static const std::string ZERO_GRADIENT_ACCUMULATORS_OP;
    static const std::string APPLY_GRADIENTS_OP;

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

    float CostFunction(VectorVectorXf logits, VectorVectorXf labels);
    VectorVectorXf BackpropagateCostFunction(VectorVectorXf logits, VectorVectorXf labels);
};

}
}