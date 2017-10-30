#pragma once

#include "third_party/eigen3/Eigen/Core"

#include <vector>

namespace ugtsa {
namespace common {

typedef std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> VectorVectorXf;
typedef std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> VectorMatrixXf;

}
}