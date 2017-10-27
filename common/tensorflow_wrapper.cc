#include "refactored_ugtsa/common/tensorflow_wrapper.h"
#include "boost/filesystem.hpp"

#include <regex>

namespace ugtsa {
namespace common {

namespace fs = boost::filesystem;
namespace tf = tensorflow;

const std::string TensorflowWrapper::GRAPHS_PATH = "graphs/";
const std::string TensorflowWrapper::MODELS_PATH = "models/";
const std::string TensorflowWrapper::LOAD_OP = "save/restore_all";
const std::string TensorflowWrapper::SAVE_OP = "save/control_dependency:0";
const std::string TensorflowWrapper::PATH_TENSOR = "save/Const";
const std::string TensorflowWrapper::STEP_TENSOR = "global_step:0";

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
    std::vector<tf::Tensor> outputs;
    session->Run({}, {name}, {}, &outputs);
    return outputs[0].scalar<int>()(0);
}

}
}