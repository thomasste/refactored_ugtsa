#include "refactored_ugtsa/common/tensorflow_wrapper.h"
#include "boost/filesystem.hpp"

#include <regex>

namespace ugtsa {
namespace common {

namespace fs = boost::filesystem;

const std::string TensorflowWrapper::STEP_TENSOR = "global_step:0";

const std::string TensorflowWrapper::GRAPHS_PATH = "graphs/";
const std::string TensorflowWrapper::MODELS_PATH = "models/";
const std::string TensorflowWrapper::LOAD_OP = "save/restore_all";
const std::string TensorflowWrapper::SAVE_OP = "save/control_dependency:0";
const std::string TensorflowWrapper::PATH_TENSOR = "save/Const";

const std::string TensorflowWrapper::UNTRAINABLE_MODEL = "collected_untrainable_variables:0";
const std::string TensorflowWrapper::SET_UNTRAINABLE_MODEL_OP = "set_untrainable_variables";
const std::string TensorflowWrapper::SET_UNTRAINABLE_MODEL_INPUT = "set_untrainable_variables_input:0";

const std::string TensorflowWrapper::ZERO_GRADIENT_ACCUMULATORS_OP = "zero_gradient_accumulators";
const std::string TensorflowWrapper::APPLY_GRADIENTS_OP = "apply_gradients";

const std::string TensorflowWrapper::COST_FUNCTION_SEED = "cost_function/seed:0";
const std::string TensorflowWrapper::COST_FUNCTION_TRAINING = "cost_function/training:0";
const std::string TensorflowWrapper::COST_FUNCTION_LOGITS = "cost_function/logits:0";
const std::string TensorflowWrapper::COST_FUNCTION_LABELS = "cost_function/labels:0";
const std::string TensorflowWrapper::COST_FUNCTION_OUTPUT = "cost_function/output:0";
const std::string TensorflowWrapper::COST_FUNCTION_OUTPUT_GRADIENT = "cost_function/output_gradient:0";
const std::string TensorflowWrapper::COST_FUNCTION_LOGITS_GRADIENT = "cost_function/logits_gradient:0";
const std::string TensorflowWrapper::COST_FUNCTION_UPDATE_GRADIENT_ACCUMULATORS_OP = "cost_function/update_gradient_accumulators";

std::string TensorflowWrapper::GraphPath() {
    return GRAPHS_PATH + graph_name + ".pb";
}

std::string TensorflowWrapper::ModelPath(std::string model_name) {
    return MODELS_PATH + model_name;
}

std::string TensorflowWrapper::ModelName(int version) {
    return graph_name + "." + std::to_string(version);
}

void TensorflowWrapper::LoadGraph() {
    auto graph_def = tf::GraphDef();
    TF_CHECK_OK(tf::ReadBinaryProto(tf::Env::Default(), GraphPath(), &graph_def));
    TF_CHECK_OK(session->Create(graph_def));
}

void TensorflowWrapper::LoadModel(int version) {
    if (version == -1) {
        std::smatch match;
        std::regex regex("^" + graph_name + ".([0-9]+).index$");
        for (fs::directory_iterator it(MODELS_PATH); it != fs::directory_iterator(); ++it) {
            if (std::regex_search(it->path().filename().string(), match, regex)) {
                int v = std::atoi(match[1].str().c_str());
                if (v > version) version = v;
            }
        }
    }

    auto model_path = tf::Tensor(tf::DT_STRING, tf::TensorShape({1, 1}));
    model_path.matrix<std::string>()(0, 0) = ModelPath(ModelName(version));
    TF_CHECK_OK(session->Run({{PATH_TENSOR, model_path}}, {}, {LOAD_OP}, nullptr));
}

tf::Tensor TensorflowWrapper::VectorVectorXfToTensor(VectorVectorXf v) {
    auto t = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({v.size(), v[0].size()}));
    auto view = t.matrix<float>();
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[0].size(); j++) {
            view(i, j) = v[i](j);
        }
    }
    return t;
}

VectorVectorXf TensorflowWrapper::TensorToVectorVectorXf(tf::Tensor t) {
    auto v = VectorVectorXf();
    auto view = t.matrix<float>();
    for (int i = 0; i < t.dim_size(0); i++) {
        Eigen::VectorXf row = Eigen::VectorXf::Zero(t.dim_size(1));
        for (int j = 0; j < t.dim_size(1); j++) {
            row(j) = view(i, j);
        }
        v.push_back(row);
    }
    return v;
}

tf::Tensor TensorflowWrapper::VectorXfToTensor(Eigen::VectorXf v) {
    auto t = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({v.size()}));
    auto view = t.vec<float>();
    for (int i = 0; i < v.size(); i++) {
        view(i) = v(i);
    }
    return t;
}

Eigen::VectorXf TensorflowWrapper::TensorToVectorXf(tf::Tensor t) {
    Eigen::VectorXf v = Eigen::VectorXf::Zero(t.dim_size(0));
    auto view = t.vec<float>();
    for (int i = 0; i < t.dim_size(0); i++) {
        v(i) = view(i);
    }
    return v;
}

tf::Tensor TensorflowWrapper::BoolToTensor(bool b) {
    auto t = tf::Tensor(tf::DT_BOOL, tf::TensorShape({}));
    t.scalar<bool>()(0) = b;
    return t;
}

tf::Tensor TensorflowWrapper::FloatToTensor(float f) {
    auto t = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({}));
    t.scalar<float>()(0) = f;
    return t;
}

float TensorflowWrapper::TensorToFloat(tf::Tensor t) {
    return t.scalar<float>()(0);
}

tf::Tensor TensorflowWrapper::SeedToTensor(std::array<long long int, 2> seed) {
    auto t = tf::Tensor(tf::DT_INT64, {2});
    auto view = t.vec<long long int>();
    view(0) = seed[0];
    view(1) = seed[1];
    return t;
}

TensorflowWrapper::TensorflowWrapper(std::string graph_name, int version)
        : graph_name(graph_name) {
    auto session_options = tf::SessionOptions();
    TF_CHECK_OK(tf::NewSession(session_options, &session));
    LoadGraph();
    LoadModel(version);
}

TensorflowWrapper::~TensorflowWrapper() {
    TF_CHECK_OK(session->Close());
}

void TensorflowWrapper::SaveModel() {
    auto model_path = tf::Tensor(tf::DT_STRING, tf::TensorShape({1, 1}));
    model_path.matrix<std::string>()(0, 0) = ModelPath(ModelName(EvalIntScalar(STEP_TENSOR)));
    TF_CHECK_OK(session->Run({{PATH_TENSOR, model_path}}, {}, {SAVE_OP}, nullptr));
}

int TensorflowWrapper::EvalIntScalar(std::string name) {
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(session->Run({}, {name}, {}, &outputs));
    return outputs[0].scalar<int>()(0);
}

Eigen::VectorXf TensorflowWrapper::GetUntrainableModel() {
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(session->Run({}, {UNTRAINABLE_MODEL}, {}, &outputs));
    return TensorToVectorXf(outputs[0]);
}

void TensorflowWrapper::SetUntrainableModel(Eigen::VectorXf model) {
    TF_CHECK_OK(session->Run({{SET_UNTRAINABLE_MODEL_INPUT, VectorXfToTensor(model)}}, {}, {SET_UNTRAINABLE_MODEL_OP}, nullptr));
}

void TensorflowWrapper::ZeroGradientAccumulators() {
    TF_CHECK_OK(session->Run({}, {}, {ZERO_GRADIENT_ACCUMULATORS_OP}, nullptr));
}

void TensorflowWrapper::ApplyGradients() {
    TF_CHECK_OK(session->Run({}, {}, {APPLY_GRADIENTS_OP}, nullptr));
}

float TensorflowWrapper::CostFunction(VectorVectorXf logits, VectorVectorXf labels) {
    std::vector<tf::Tensor> outputs;
    TF_CHECK_OK(
        session->Run({
                {COST_FUNCTION_SEED, SeedToTensor({0, 0})},
                {COST_FUNCTION_TRAINING, BoolToTensor(false)},
                {COST_FUNCTION_LOGITS, VectorVectorXfToTensor(logits)},
                {COST_FUNCTION_LABELS, VectorVectorXfToTensor(labels)}},
            {COST_FUNCTION_OUTPUT},
            {},
            &outputs));
    return TensorToFloat(outputs[0]);
}

VectorVectorXf TensorflowWrapper::BackpropagateCostFunction(VectorVectorXf logits, VectorVectorXf labels) {
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(
        session->Run({
                {COST_FUNCTION_SEED, SeedToTensor({0, 0})},
                {COST_FUNCTION_TRAINING, BoolToTensor(false)},
                {COST_FUNCTION_LOGITS, VectorVectorXfToTensor(logits)},
                {COST_FUNCTION_LABELS, VectorVectorXfToTensor(labels)},
                {COST_FUNCTION_OUTPUT_GRADIENT, FloatToTensor(1.)}},
            {COST_FUNCTION_LOGITS_GRADIENT},
            {COST_FUNCTION_UPDATE_GRADIENT_ACCUMULATORS_OP},
            &outputs));
    return TensorToVectorVectorXf(outputs[0]);
}

}
}