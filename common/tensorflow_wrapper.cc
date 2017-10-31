#include "refactored_ugtsa/common/tensorflow_wrapper.h"
#include "boost/filesystem.hpp"

#include <regex>

namespace ugtsa {
namespace common {

namespace fs = boost::filesystem;

const std::string TensorflowWrapper::STEP_TENSOR = "global_step:0";

const std::string TensorflowWrapper::WORKER_COUNT = "worker_count:0";
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

const std::string TensorflowWrapper::STATISTIC_SEED = "statistic/seed:0";
const std::string TensorflowWrapper::STATISTIC_TRAINING = "statistic/training:0";
const std::string TensorflowWrapper::STATISTIC_BOARD = "statistic/board:0";
const std::string TensorflowWrapper::STATISTIC_GAME_STATE_INFO = "statistic/game_state_info:0";
const std::string TensorflowWrapper::STATISTIC_OUTPUT = "statistic/output:0";
const std::string TensorflowWrapper::STATISTIC_OUTPUT_GRADIENT = "statistic/output_gradient:0";
const std::string TensorflowWrapper::STATISTIC_UPDATE_GRADIENT_ACCUMULATORS_OP = "statistic/update_gradient_accumulators";

const std::string TensorflowWrapper::UPDATE_SEED = "update/seed:0";
const std::string TensorflowWrapper::UPDATE_TRAINING = "update/training:0";
const std::string TensorflowWrapper::UPDATE_PAYOFF = "update/payoff:0";
const std::string TensorflowWrapper::UPDATE_OUTPUT = "update/output:0";
const std::string TensorflowWrapper::UPDATE_OUTPUT_GRADIENT = "update/output_gradient:0";
const std::string TensorflowWrapper::UPDATE_UPDATE_GRADIENT_ACCUMULATORS_OP = "update/update_gradient_accumulators";

const std::string TensorflowWrapper::MODIFIED_STATISTIC_SEED = "modified_statistic/seed:0";
const std::string TensorflowWrapper::MODIFIED_STATISTIC_TRAINING = "modified_statistic/training:0";
const std::string TensorflowWrapper::MODIFIED_STATISTIC_STATISTIC = "modified_statistic/statistic:0";
const std::string TensorflowWrapper::MODIFIED_STATISTIC_UPDATES_COUNT = "modified_statistic/updates_count:0";
const std::string TensorflowWrapper::MODIFIED_STATISTIC_UPDATES = "modified_statistic/updates:0";
const std::string TensorflowWrapper::MODIFIED_STATISTIC_OUTPUT = "modified_statistic/output:0";
const std::string TensorflowWrapper::MODIFIED_STATISTIC_OUTPUT_GRADIENT = "modified_statistic/output_gradient:0";
const std::string TensorflowWrapper::MODIFIED_STATISTIC_STATISTIC_GRADIENT = "modified_statistic/statistic_gradient:0";
const std::string TensorflowWrapper::MODIFIED_STATISTIC_UPDATES_GRADIENT = "modified_statistic/updates_gradient:0";
const std::string TensorflowWrapper::MODIFIED_STATISTIC_UPDATE_GRADIENT_ACCUMULATORS_OP = "modified_statistic/update_gradient_accumulators";

const std::string TensorflowWrapper::MODIFIED_UPDATE_SEED = "modified_update/seed:0";
const std::string TensorflowWrapper::MODIFIED_UPDATE_TRAINING = "modified_update/training:0";
const std::string TensorflowWrapper::MODIFIED_UPDATE_UPDATE = "modified_update/update:0";
const std::string TensorflowWrapper::MODIFIED_UPDATE_STATISTIC = "modified_update/statistic:0";
const std::string TensorflowWrapper::MODIFIED_UPDATE_OUTPUT = "modified_update/output:0";
const std::string TensorflowWrapper::MODIFIED_UPDATE_OUTPUT_GRADIENT = "modified_update/output_gradient:0";
const std::string TensorflowWrapper::MODIFIED_UPDATE_UPDATE_GRADIENT = "modified_update/updates_gradient:0";
const std::string TensorflowWrapper::MODIFIED_UPDATE_STATISTIC_GRADIENT = "modified_update/statistic_gradient:0";
const std::string TensorflowWrapper::MODIFIED_UPDATE_UPDATE_GRADIENT_ACCUMULATORS_OP = "modified_update/update_gradient_accumulators";

const std::string TensorflowWrapper::MOVE_RATE_SEED = "move_rate/seed:0";
const std::string TensorflowWrapper::MOVE_RATE_TRAINING = "move_rate/training:0";
const std::string TensorflowWrapper::MOVE_RATE_PARENT_STATISTIC = "move_rate/parent_statistic:0";
const std::string TensorflowWrapper::MOVE_RATE_CHILD_STATISTIC = "move_rate/child_statistic:0";
const std::string TensorflowWrapper::MOVE_RATE_OUTPUT = "move_rate/output:0";
const std::string TensorflowWrapper::MOVE_RATE_OUTPUT_GRADIENT = "move_rate/output_gradient:0";
const std::string TensorflowWrapper::MOVE_RATE_PARENT_STATISTIC_GRADIENT = "move_rate/parent_statistic_gradient:0";
const std::string TensorflowWrapper::MOVE_RATE_CHILD_STATISTIC_GRADIENT = "move_rate/child_statistic_gradient:0";
const std::string TensorflowWrapper::MOVE_RATE_UPDATE_GRADIENT_ACCUMULATORS_OP = "move_rate/update_gradient_accumulators";

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

    auto model_path = tf::Tensor(tf::DT_STRING, {1, 1});
    model_path.matrix<std::string>()(0, 0) = ModelPath(ModelName(version));
    TF_CHECK_OK(session->Run({{PATH_TENSOR, model_path}}, {}, {LOAD_OP}, nullptr));
}

tf::Tensor TensorflowWrapper::VectorVectorXfToTensor(const VectorVectorXf &v) {
    auto t = tf::Tensor(tf::DT_FLOAT, {(int) v.size(), (int) v[0].size()});
    auto view = t.matrix<float>();
    for (uint i = 0; i < v.size(); i++) {
        for (uint j = 0; j < v[0].size(); j++) {
            view(i, j) = v[i](j);
        }
    }
    return t;
}

VectorVectorXf TensorflowWrapper::TensorToVectorVectorXf(const tf::Tensor &t) {
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

tf::Tensor TensorflowWrapper::VectorXfToTensor(const Eigen::VectorXf &v) {
    auto t = tf::Tensor(tf::DT_FLOAT, {v.size()});
    auto view = t.vec<float>();
    for (int i = 0; i < v.size(); i++) {
        view(i) = v(i);
    }
    return t;
}

Eigen::VectorXf TensorflowWrapper::TensorToVectorXf(const tf::Tensor &t) {
    Eigen::VectorXf v = Eigen::VectorXf::Zero(t.dim_size(0));
    auto view = t.vec<float>();
    for (int i = 0; i < t.dim_size(0); i++) {
        v(i) = view(i);
    }
    return v;
}

std::pair<tf::Tensor, tf::Tensor> TensorflowWrapper::VectorVectorVectorXfToTensors(const std::vector<VectorVectorXf> &v, int size) {
    auto t0 = tf::Tensor(tf::DT_INT32, {(int) v.size()});
    auto view0 = t0.vec<int>();
    for (uint i = 0; i < v.size(); i++) {
        view0(i) = v[i].size();
    }

    auto t1 = tf::Tensor(tf::DT_FLOAT, {(int) v.size(), v[0][0].size() * size});
    auto view1 = t1.matrix<float>();
    for (uint i = 0; i < v.size(); i++) {
        for (int j = 0; j < size; j++) {
            for (uint k = 0; k < v[0][0].size(); k++) {
                view1(i, j * size + k) = 0.;
            }
        }
    }
    for (uint i = 0; i < v.size(); i++) {
        for (uint j = 0; j < v[i].size(); j++) {
            for (uint k = 0; k < v[i][j].size(); k++) {
                view1(i, j * size + k) = v[i][j](k);
            }
        }
    }

    return {t0, t1};
}

std::vector<VectorVectorXf> TensorflowWrapper::TensorToVectorVectorVectorXf(const tf::Tensor &t, const std::vector<VectorVectorXf> &s) {
    auto view = t.matrix<float>();
    auto vvv = std::vector<VectorVectorXf>();
    for (uint i = 0; i < s.size(); i++) {
        auto vv = VectorVectorXf();
        for (uint j = 0; j < s[i].size(); j++) {
            Eigen::VectorXf v = Eigen::VectorXf::Zero(s[i][j].size());
            for (int k = 0; k < s[i][j].size(); k++) {
                v(k) = view(i, j * s[i][j].size() + k);
            }
            vv.push_back(v);
        }
        vvv.push_back(vv);
    }
    return vvv;
}

tf::Tensor TensorflowWrapper::VectorMatrixXfToTensor(const VectorMatrixXf &v) {
    auto t = tf::Tensor(tf::DT_FLOAT, {(int) v.size(), v[0].rows(), v[0].cols()});
    auto view = t.tensor<float, 3>();
    for (uint i = 0; i < v.size(); i++) {
        for (uint j = 0; j < v[0].rows(); j++) {
            for (uint k = 0; k < v[0].cols(); k++) {
                view(i, j, k) = v[i](j, k);
            }
        }
    }
    return t;
}

tf::Tensor TensorflowWrapper::BoolToTensor(bool b) {
    auto t = tf::Tensor(tf::DT_BOOL, {});
    t.scalar<bool>()(0) = b;
    return t;
}

tf::Tensor TensorflowWrapper::FloatToTensor(float f) {
    auto t = tf::Tensor(tf::DT_FLOAT, {});
    t.scalar<float>()(0) = f;
    return t;
}

float TensorflowWrapper::TensorToFloat(const tf::Tensor &t) {
    return t.scalar<float>()(0);
}

tf::Tensor TensorflowWrapper::SeedToTensor(const std::array<long long int, 2> &seed) {
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
    worker_count = EvalIntScalar(WORKER_COUNT);
}

TensorflowWrapper::~TensorflowWrapper() {
    TF_CHECK_OK(session->Close());
}

void TensorflowWrapper::SaveModel() {
    auto model_path = tf::Tensor(tf::DT_STRING, {1, 1});
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

void TensorflowWrapper::SetUntrainableModel(const Eigen::VectorXf &model) {
    TF_CHECK_OK(session->Run({{SET_UNTRAINABLE_MODEL_INPUT, VectorXfToTensor(model)}}, {}, {SET_UNTRAINABLE_MODEL_OP}, nullptr));
}

void TensorflowWrapper::ZeroGradientAccumulators() {
    TF_CHECK_OK(session->Run({}, {}, {ZERO_GRADIENT_ACCUMULATORS_OP}, nullptr));
}

void TensorflowWrapper::ApplyGradients() {
    TF_CHECK_OK(session->Run({}, {}, {APPLY_GRADIENTS_OP}, nullptr));
}

VectorVectorXf TensorflowWrapper::Statistic(const std::array<long long int, 2> &seed, bool training, const VectorMatrixXf &board, const VectorVectorXf &game_state_info) {
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(
        session->Run({
                {STATISTIC_SEED, SeedToTensor(seed)},
                {STATISTIC_TRAINING, BoolToTensor(training)},
                {STATISTIC_BOARD, VectorMatrixXfToTensor(board)},
                {STATISTIC_GAME_STATE_INFO, VectorVectorXfToTensor(game_state_info)}},
            {STATISTIC_OUTPUT},
            {},
            &outputs));
    return TensorToVectorVectorXf(outputs[0]);
}

void TensorflowWrapper::BackpropagateStatistic(const std::array<long long int, 2> &seed, bool training, const VectorMatrixXf &board, const VectorVectorXf &game_state_info, const VectorVectorXf &output_gradient) {
    TF_CHECK_OK(
        session->Run({
                {STATISTIC_SEED, SeedToTensor(seed)},
                {STATISTIC_TRAINING, BoolToTensor(training)},
                {STATISTIC_BOARD, VectorMatrixXfToTensor(board)},
                {STATISTIC_GAME_STATE_INFO, VectorVectorXfToTensor(game_state_info)},
                {STATISTIC_OUTPUT_GRADIENT, VectorVectorXfToTensor(output_gradient)}},
            {},
            {STATISTIC_UPDATE_GRADIENT_ACCUMULATORS_OP},
            nullptr));
}

VectorVectorXf TensorflowWrapper::Update(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &payoff) {
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(
        session->Run({
                {UPDATE_SEED, SeedToTensor(seed)},
                {UPDATE_TRAINING, BoolToTensor(training)},
                {UPDATE_PAYOFF, VectorVectorXfToTensor(payoff)}},
            {UPDATE_OUTPUT},
            {},
            &outputs));
    return TensorToVectorVectorXf(outputs[0]);
}

void TensorflowWrapper::BackpropagateUpdate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &payoff, const VectorVectorXf &output_gradient) {
    TF_CHECK_OK(
        session->Run({
                {UPDATE_SEED, SeedToTensor(seed)},
                {UPDATE_TRAINING, BoolToTensor(training)},
                {UPDATE_PAYOFF, VectorVectorXfToTensor(payoff)},
                {UPDATE_OUTPUT_GRADIENT, VectorVectorXfToTensor(output_gradient)}},
            {},
            {UPDATE_UPDATE_GRADIENT_ACCUMULATORS_OP},
            nullptr));
}

VectorVectorXf TensorflowWrapper::ModifiedStatistic(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &statistic, const std::vector<VectorVectorXf> &updates) {
    auto tensors = VectorVectorVectorXfToTensors(updates, worker_count);
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(
        session->Run({
                {MODIFIED_STATISTIC_SEED, SeedToTensor(seed)},
                {MODIFIED_STATISTIC_TRAINING, BoolToTensor(training)},
                {MODIFIED_STATISTIC_STATISTIC, VectorVectorXfToTensor(statistic)},
                {MODIFIED_STATISTIC_UPDATES_COUNT, tensors.first},
                {MODIFIED_STATISTIC_UPDATES, tensors.second}},
            {MODIFIED_STATISTIC_OUTPUT},
            {},
            &outputs));
    return TensorToVectorVectorXf(outputs[0]);
}

std::pair<VectorVectorXf, std::vector<VectorVectorXf>> TensorflowWrapper::BackpropagateModifiedStatistic(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &statistic, const std::vector<VectorVectorXf> &updates, const VectorVectorXf &output_gradient) {
    auto tensors = VectorVectorVectorXfToTensors(updates, worker_count);
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(
        session->Run({
                {MODIFIED_STATISTIC_SEED, SeedToTensor(seed)},
                {MODIFIED_STATISTIC_TRAINING, BoolToTensor(training)},
                {MODIFIED_STATISTIC_STATISTIC, VectorVectorXfToTensor(statistic)},
                {MODIFIED_STATISTIC_UPDATES_COUNT, tensors.first},
                {MODIFIED_STATISTIC_UPDATES, tensors.second}},
            {MODIFIED_STATISTIC_STATISTIC_GRADIENT, MODIFIED_STATISTIC_UPDATES_GRADIENT},
            {MODIFIED_STATISTIC_UPDATE_GRADIENT_ACCUMULATORS_OP},
            &outputs));
    return {TensorToVectorVectorXf(outputs[0]), TensorToVectorVectorVectorXf(outputs[1], updates)};
}

VectorVectorXf TensorflowWrapper::ModifiedUpdate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &update, const VectorVectorXf &statistic) {
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(
        session->Run({
                {MODIFIED_UPDATE_SEED, SeedToTensor(seed)},
                {MODIFIED_UPDATE_TRAINING, BoolToTensor(training)},
                {MODIFIED_UPDATE_UPDATE, VectorVectorXfToTensor(update)},
                {MODIFIED_UPDATE_STATISTIC, VectorVectorXfToTensor(statistic)}},
            {MODIFIED_UPDATE_OUTPUT},
            {},
            &outputs));
    return TensorToVectorVectorXf(outputs[0]);
}

std::pair<VectorVectorXf, VectorVectorXf> TensorflowWrapper::BackpropagateModifiedUpdate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &update, const VectorVectorXf &statistic, const VectorVectorXf &output_gradient) {
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(
        session->Run({
                {MODIFIED_UPDATE_SEED, SeedToTensor(seed)},
                {MODIFIED_UPDATE_TRAINING, BoolToTensor(training)},
                {MODIFIED_UPDATE_UPDATE, VectorVectorXfToTensor(update)},
                {MODIFIED_UPDATE_STATISTIC, VectorVectorXfToTensor(statistic)},
                {MODIFIED_UPDATE_OUTPUT_GRADIENT, VectorVectorXfToTensor(output_gradient)}},
            {MODIFIED_UPDATE_UPDATE_GRADIENT, MODIFIED_UPDATE_STATISTIC_GRADIENT},
            {MODIFIED_UPDATE_UPDATE_GRADIENT_ACCUMULATORS_OP},
            &outputs));
    return {TensorToVectorVectorXf(outputs[0]), TensorToVectorVectorXf(outputs[1])};
}

VectorVectorXf TensorflowWrapper::MoveRate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &parent_statistic, const VectorVectorXf &child_statistic) {
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(
        session->Run({
                {MOVE_RATE_SEED, SeedToTensor(seed)},
                {MOVE_RATE_TRAINING, BoolToTensor(training)},
                {MOVE_RATE_PARENT_STATISTIC, VectorVectorXfToTensor(parent_statistic)},
                {MOVE_RATE_CHILD_STATISTIC, VectorVectorXfToTensor(child_statistic)}},
            {MOVE_RATE_OUTPUT},
            {},
            &outputs));
    return TensorToVectorVectorXf(outputs[0]);
}

std::pair<VectorVectorXf, VectorVectorXf> TensorflowWrapper::BackpropagateMoveRate(const std::array<long long int, 2> &seed, bool training, const VectorVectorXf &parent_statistic, const VectorVectorXf &child_statistic, const VectorVectorXf &output_gradient) {
    auto outputs = std::vector<tf::Tensor>();
    TF_CHECK_OK(
        session->Run({
                {MOVE_RATE_SEED, SeedToTensor(seed)},
                {MOVE_RATE_TRAINING, BoolToTensor(training)},
                {MOVE_RATE_PARENT_STATISTIC, VectorVectorXfToTensor(parent_statistic)},
                {MOVE_RATE_CHILD_STATISTIC, VectorVectorXfToTensor(child_statistic)},
                {MOVE_RATE_OUTPUT_GRADIENT, VectorVectorXfToTensor(output_gradient)}},
            {MOVE_RATE_PARENT_STATISTIC_GRADIENT, MOVE_RATE_CHILD_STATISTIC_GRADIENT},
            {MOVE_RATE_UPDATE_GRADIENT_ACCUMULATORS_OP},
            &outputs));
    return {TensorToVectorVectorXf(outputs[0]), TensorToVectorVectorXf(outputs[1])};
}

float TensorflowWrapper::CostFunction(const VectorVectorXf &logits, const VectorVectorXf &labels) {
    auto outputs = std::vector<tf::Tensor>();
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

VectorVectorXf TensorflowWrapper::BackpropagateCostFunction(const VectorVectorXf &logits, const VectorVectorXf &labels) {
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