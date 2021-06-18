

#include <roadseg/road_vertex.h>

#include <cmath>

namespace roadseg
{

  double RoadVertex::maxFirstEigValThresh = 1;

  const int COORDS_NUM = 3;

  RoadVertex::RoadVertex(const RoadVertex *v)
      : Vertex(v)
  {
    coordsSumOfSquares = new long double *[COORDS_NUM];
    for (int i = 0; i < COORDS_NUM; i++)
    {
      coordsSumOfSquares[i] = new long double[COORDS_NUM];
      for (int j = 0; j < COORDS_NUM; j++)
        coordsSumOfSquares[i][j] = v->coordsSumOfSquares[i][j];
    }
    helperStats = v->helperStats;
    needToUpdate = v->needToUpdate;
  }

  RoadVertex::~RoadVertex()
  {
    if (coordsSumOfSquares != NULL)
    {
      for (int i = 0; i < COORDS_NUM; i++)
        delete[] coordsSumOfSquares[i];
      delete[] coordsSumOfSquares;
    }
  }

  void RoadVertex::Initialize(int _channelsNum)
  {
    Vertex::Initialize(_channelsNum);

    coordsSumOfSquares = new long double *[COORDS_NUM];
    for (int i = 0; i < COORDS_NUM; i++)
    {
      coordsSumOfSquares[i] = new long double[COORDS_NUM];
      for (int j = 0; j < COORDS_NUM; j++)
        coordsSumOfSquares[i][j] = 0;
    }
  }

  void RoadVertex::update(const uint8_t *pix)
  {
    double pixd[3] = {(double)pix[0], (double)pix[1], (double)pix[2]};
    Vertex::update(pixd);

    Eigen::Vector3d pix_vec(pixd[0], pixd[1], pixd[2]);
    Eigen::Matrix3d pix_sq_mat = pix_vec * pix_vec.transpose();
    for (int i = 0; i < COORDS_NUM; ++i)
      for (int j = 0; j < COORDS_NUM; ++j)
        coordsSumOfSquares[i][j] += (long double)pix_sq_mat(i, j);

    needToUpdate = true;
  }

  void RoadVertex::absorb(Vertex *v)
  {
    Vertex::absorb(v);

    RoadVertex *cv = dynamic_cast<RoadVertex *>(v);
    for (int i = 0; i < COORDS_NUM; i++)
      for (int j = 0; j < COORDS_NUM; j++)
        coordsSumOfSquares[i][j] += cv->coordsSumOfSquares[i][j];

    needToUpdate = true;
  }

  const RoadVertex::HelperStats &RoadVertex::getHelperStats() const
  {
    if (needToUpdate)
      updateHelperStats();
    return helperStats;
  }

  void RoadVertex::updateHelperStats() const
  {
    HelperStats &hs = helperStats;

    // calculate mean and covariance matrix
    {
      Eigen::Vector3d sum(channelsSum[0], channelsSum[1], channelsSum[2]);
      Eigen::Matrix3d sumSquares;
      for (int i = 0; i < COORDS_NUM; ++i)
        for (int j = 0; j < COORDS_NUM; ++j)
          sumSquares(i, j) = coordsSumOfSquares[i][j];

      hs.mean = sum / area;
      hs.covariance = sumSquares / area - hs.mean * hs.mean.transpose();
    }

    // calculate eigen vectors
    hs.eigSolver.compute(hs.covariance, Eigen::EigenvaluesOnly);
    needToUpdate = false;
  }

  Json::Value RoadVertex::jsonLog() const
  {
    Json::Value root = Vertex::jsonLog();

    HelperStats const &hs = getHelperStats();

    root["cov"] = Json::arrayValue;
    for (int i = 0; i < COORDS_NUM; ++i)
    {
      Json::Value cov_row = Json::arrayValue;
      for (int j = 0; j < COORDS_NUM; ++j)
        cov_row.append(hs.covariance(i, j));
      root["cov"].append(cov_row);
    }

    root["values"] = Json::arrayValue;
    for (int i = 0; i < COORDS_NUM; i++)
      root["values"].append(hs.eigenvalues()[i]);

    return root;
  }

}
// ns roadseg
