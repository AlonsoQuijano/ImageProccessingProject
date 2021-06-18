/*
Copyright (c) 2012-2018, Visillect Service LLC. All rights reserved.
Developed for Kharkevich Institute for Information Transmission Problems of the
              Russian Academy of Sciences (IITP RAS).

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of copyright holders.
*/


#include <roadseg/road_distance_func.h>

#include <remseg/distance_func.h>

#include <cassert>
#include <cmath>
#include <algorithm>

THIRDPARTY_INCLUDES_BEGIN
#include <Eigen/Geometry>
THIRDPARTY_INCLUDES_END

namespace roadseg {

double dist_point_to_line(Eigen::Vector3d const& p,
                          Eigen::Vector3d const& v,
                          Eigen::Vector3d const& mean)
{
  Eigen::Vector3d r = mean - p;
  Eigen::Vector3d res = r - v.dot(r) * v;
  return res.norm();
}

double dist_point_to_segment(const Eigen::Vector3d& p,
                             const Eigen::Vector3d& a,
                             const Eigen::Vector3d& b)
{
  if ((b - a).dot(p - a) < 0)
    return (p - a).norm();
  if ((a - b).dot(p - b) < 0)
    return (p - b).norm();
  return dist_point_to_line(p, (b - a).normalized(), a);
}

EdgeValue error(const RoadVertex *v) {
  RoadVertex::HelperStats const & hs = v->getHelperStats();
  double err = hs.eigenvalues()[0];
  return err * v->area;
}

bool isParallel(const RoadVertex *v1, const RoadVertex *v2)
{
  RoadVertex v1cp(v1);
  RoadVertex v2cp(v2);
  v1cp.absorb(&v2cp);
  double firstEigenVal = error(&v1cp);
  return firstEigenVal < RoadVertex::getMaxFirstEigenVal();
}

EdgeValue criteria(const RoadVertex *v1, const RoadVertex *v2)
{
  if (!isParallel(v1, v2))
    return std::numeric_limits<double>::infinity();

  RoadVertex v(v1);
  RoadVertex tmp(v2);
  v.absorb(&tmp);
  return std::sqrt(error(&v) - error(v1) - error(v2));
}

} // ns roadseg
