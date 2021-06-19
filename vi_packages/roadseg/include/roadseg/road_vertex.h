
#pragma once

#include <remseg/vertex.h>
#include <vector>

#include <minbase/crossplat.h>

THIRDPARTY_INCLUDES_BEGIN
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
THIRDPARTY_INCLUDES_END

using namespace vi::remseg;

namespace roadseg {

class RoadVertex : public Vertex
{
public:
  long double **coordsSumOfSquares;

  struct HelperStats
  {
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigSolver;

    const Eigen::Vector3d& eigenvalues() const { return eigSolver.eigenvalues(); }
  };

  RoadVertex() :
    Vertex()
  , coordsSumOfSquares(nullptr)
  , needToUpdate(false)
  {}

  RoadVertex(const RoadVertex* v);

  ~RoadVertex();

  void Initialize(int _channelsNum) override;
  void update(const uint8_t * pix) override;
  void absorb(Vertex *to_be_absorbed) override;

  const HelperStats & getHelperStats() const;

  Json::Value jsonLog() const override;

  static double getMaxFirstEigenVal() { return maxFirstEigValThresh; }
  static void setMaxFirstEigValThresh(double d) { maxFirstEigValThresh = d; }

private:
  mutable HelperStats helperStats;

  mutable bool needToUpdate;
  static double maxFirstEigValThresh;

  void updateHelperStats() const;
};

}	// ns roadseg
