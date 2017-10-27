#include "tensorflow/core/public/session.h"

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

    tf::Session* session;
    std::string graph_name;

    std::string GraphPath();
    std::string ModelPath(std::string model_name);
    std::string ModelName(int version);
    void LoadGraph();
    void LoadModel(int version);

public:
    TensorflowWrapper(std::string graph_name, int version);
    ~TensorflowWrapper();

    void SaveModel();
    int EvalIntScalar(std::string name);
};

}
}