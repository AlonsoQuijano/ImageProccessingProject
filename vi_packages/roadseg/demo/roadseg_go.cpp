#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iterator>

#include <minbase/crossplat.h>
#include <minimgapi/minimgapi-helpers.hpp>
#include <minimgapi/imgguard.hpp>
#include <minimgio/minimgio.h>
#include <mximg/image.h>
#include <mximg/ocv.h>
#include <vi_cvt/std/exception_macros.hpp>
#include <vi_cvt/ocv/image.hpp>

#include <roadseg/road_distance_func.h>
#include <roadseg/road_vertex.h>
#include <remseg/segmentator.hpp>
#include <remseg/utils.h>

#include <opencv2/opencv.hpp>

THIRDPARTY_INCLUDES_BEGIN
#include <boost/filesystem.hpp>
#include <boost/serialization/vector.hpp>
#include <tclap/CmdLine.h>
THIRDPARTY_INCLUDES_END

namespace bfs = boost::filesystem;

using namespace vi::remseg;
using namespace roadseg;

void prepare_image(mximg::PImage &xyz_mximage, std::string const &depth_img_path)
{
  cv::Mat cv_depth_image = cv::imread(depth_img_path, cv::IMREAD_ANYDEPTH);
  DECLARE_GUARDED_MINIMG(xyz_image);
  // 12-channels hack to write 3-channel 32-bit float image as uint_8 one
  NewMinImagePrototype(&xyz_image, cv_depth_image.cols, cv_depth_image.rows, 12, MinTyp::TYP_UINT8);
  if (cv_depth_image.type() != CV_16U)
  {
    throw std::runtime_error("Wrong depth image format. Expected image's type to be uint16_t");
  }
  auto data_path = bfs::path(depth_img_path).parent_path();
  std::string K_path = data_path.append("K").string();
  std::ifstream ifs(K_path);
  std::istream_iterator<float> start(ifs), end;
  std::vector<float> K_coefs(start, end);
  const auto fx = K_coefs[0];
  const auto cx = K_coefs[2];
  const auto fy = K_coefs[4];
  const auto cy = K_coefs[5];
  for (int i = 0; i < cv_depth_image.rows; ++i)
    for (int j = 0; j < cv_depth_image.cols; ++j)
    {
      auto depth = cv_depth_image.at<ushort>(i, j);
      float x = 0;
      float y = 0;
      float z = 0;
      if (depth > 0)
      {
        z = depth / 1000.;
        x = z * (j - cx) / fx;
        y = z * (i - cy) / fy;
      }
      GetMinImageLineAs<float>(&xyz_image, i)[j * 3] = x;
      GetMinImageLineAs<float>(&xyz_image, i)[j * 3 + 1] = y;
      GetMinImageLineAs<float>(&xyz_image, i)[j * 3 + 2] = z;
    }
  xyz_mximage = mximg::createByOwning(xyz_image);
}


Eigen::Vector3d get_eigen_vector(RoadVertex* vertex) {
  auto hs =  vertex->getHelperStats();
  hs.eigSolver.compute(hs.covariance);
  return hs.eigSolver.eigenvectors().col(0);
}

bool check_angles(RoadVertex* a, RoadVertex* b, float cos_angle_thresh) {
  auto vec_a = get_eigen_vector(a);
  auto vec_b = get_eigen_vector(b);

  return std::abs(vec_a.dot(vec_b)) > cos_angle_thresh;
}

bool check_segment_is_plane(RoadVertex *vertex)
{
  auto hs = vertex->getHelperStats();
  auto eig_vals = hs.eigenvalues();
  const float EPS = 1e-7f;
  return (eig_vals[1] > 1e-7f) && (eig_vals[0] / eig_vals[1] < 0.01); // no points or lines segments to merge
}

void prepare_merge_list(std::vector<std::vector<SegmentID>> &segments_to_merge, Segmentator<RoadVertex> &segmentator, float max_angle) {
  auto image_map = segmentator.getImageMap();
  auto stats = image_map.getSegmentStats();
  std::vector<bool> segments_to_check(image_map.getWidth() * image_map.getHeight(), false);
  for (int i = 0; i < image_map.getHeight(); ++i)
    for (int j = 0; j < image_map.getWidth(); ++j){
      auto id = image_map.getSegment({j, i});
      if (check_segment_is_plane(segmentator.vertexById(id))) {
        segments_to_check[id] = true;
      }
    }
  auto cos_angle_thresh = cos(max_angle);

  auto last_segment_checked = std::find(segments_to_check.begin(), segments_to_check.end(), true);
  while (last_segment_checked != segments_to_check.end())
  {
    int start_id = last_segment_checked - segments_to_check.begin();
    segments_to_merge.push_back({start_id});
    segments_to_check[start_id] = false;

    std::queue<SegmentID> bfs_ids;
    bfs_ids.emplace(start_id);
    while (bfs_ids.size())
    {
      auto checking_id = bfs_ids.front();
      bfs_ids.pop();
      for (auto const& neighbour_id : stats[checking_id].neighbours) {
        if (!segments_to_check[neighbour_id]) {
          continue;
        }
        if (check_angles(segmentator.vertexById(checking_id), segmentator.vertexById(neighbour_id), cos_angle_thresh)) {
          segments_to_merge.back().emplace_back(neighbour_id);
          segments_to_check[neighbour_id] = false;
          bfs_ids.emplace(neighbour_id);
        }
      }
    }
    last_segment_checked = std::find(last_segment_checked, segments_to_check.end(), true);
  }
}

void merge_planes(Segmentator<RoadVertex> &segmentator, float max_angle, i8r::PLogger & dbg)
{
  std::vector<std::vector<SegmentID>> segments_to_merge;
  prepare_merge_list(segments_to_merge, segmentator, max_angle);
  for (size_t i = 0; i < segments_to_merge.size(); ++i) {
    if (segments_to_merge[i].size() < 2) {
      continue;
    }
    auto absorbent_vertex = segmentator.vertexById(segments_to_merge[i][0]);
    for (size_t j = 1; j < segments_to_merge[i].size(); ++j) {
      auto vertex = segmentator.vertexById(segments_to_merge[i][j]);
      segmentator.merge(absorbent_vertex, vertex);
    }
    if (dbg->enabled()) {
      segmentator.updateMapping();
      DECLARE_GUARDED_MINIMG(vis);
      visualize(&vis, segmentator.getImageMap());
      dbg->save(std::to_string(i), "plane_merging", &vis, "");
    }
  }
  segmentator.updateMapping();
}


void save_mask(Segmentator<RoadVertex> const& segmentator, std::string const& basename, std::string const& output) {
  SegmentStat max_area_vertex;
  SegmentID max_id = 0;
  const auto map = segmentator.getImageMap();
  const auto stats = map.getSegmentStats();
  int total_area_sum = 0;
  const int max_area = map.getWidth() * map.getHeight();
  for (auto const& stat : stats) {
    if (!total_area_sum || stat.second.area > max_area_vertex.area) {
      max_id = stat.first;
      max_area_vertex = stat.second;
    }
    total_area_sum += stat.second.area;
    if (total_area_sum >  max_area - max_area_vertex.area) {
      break;
    }
  }

  cv::Mat mask(map.getHeight(), map.getWidth(), CV_8UC1);
  for (int i = 0; i < map.getHeight(); ++i) {
    for (int j = 0; j < map.getWidth(); ++j) {
      mask.at<uchar>(i, j) =  (map.getSegment(j , i) == max_id) ? UINT_MAX : 0;
    }
  }
  std::string const mask_filename = bfs::absolute(basename + "_mask.png", output).string();
  cv::imwrite(mask_filename, mask);
}

int main(int argc, const char *argv[])
{
  i8r::AutoShutdown i8r_shutdown;

  TCLAP::CmdLine cmd("Run Range-Based Region Merge Segmentation on a Single image");
  TCLAP::ValueArg<double> errorLimit("e", "error_limit", "average error limit", false, -1, "double", cmd);
  TCLAP::ValueArg<int> segmentsLimit("n", "segm_limit", "segments limit", false, -1, "int", cmd);
  TCLAP::ValueArg<int> blockSize("b", "block_size", "initial size of regions", false, 3, "int", cmd);
  TCLAP::UnlabeledValueArg<std::string> imagePath("image", "path to source xyz-image", true, "", "string", cmd);
  TCLAP::ValueArg<std::string> output("o", "output", "path to output dir", false, ".", "string", cmd);
  TCLAP::SwitchArg debug("d", "debug", "debug mode", cmd, false);
  TCLAP::ValueArg<int> debugIter("i", "debug_iter", "debug iterations", false, 1, "int", cmd);
  TCLAP::ValueArg<int> maxSegments("s", "max_segments", "max segments for debug output", false, -1, "int", cmd);
  TCLAP::ValueArg<double> maxEigVal("v", "max_eigen_val", "maximum eigen val to consider region as a plane", false, 0.1, "double", cmd);
  TCLAP::ValueArg<double> maxAngle("a", "max_angle", "maximum angle to consider ", false, 0, "double", cmd);

  cmd.parse(argc, argv);

  RoadVertex::setMaxFirstEigValThresh(maxEigVal.getValue());

  if (!bfs::is_directory(output.getValue()))
    throw std::runtime_error("Failed to find output directory " + output.getValue());

  if (debug.getValue())
    i8r::configure(output.getValue());

  std::string const basename = bfs::path(imagePath.getValue()).stem().string();
  std::string const imgres_filename = bfs::absolute(basename + ".png", output.getValue()).string();

  try
  {
    mximg::PImage image;
    prepare_image(image, imagePath.getValue());

    auto dbg = i8r::logger("debug." + basename + ".pointlike");

    Segmentator<RoadVertex> segmentatorPlanes(*image, error, criteria, true);
    // group pixels into blocks
    std::cout << "Prepare blocks" << std::endl;
    for (int i = 0; i < (*image)->height; i += blockSize.getValue())
    {
      for (int j = 0; j < (*image)->width; j += blockSize.getValue())
      {
        std::cout << "Block: " << i << ", " << j << "\n";
        for (int ki = 0; ki < std::min<int>(blockSize.getValue(), (*image)->height - i); ++ki)
        {
          for (int kj = 0; kj < std::min<int>(blockSize.getValue(), (*image)->width - j); ++kj)
          {
            if ((ki == 0) && (kj == 0)) {
              continue;
            }
            auto top_left_block_corner = segmentatorPlanes.pointToVertex({j, i});
            auto block_point = segmentatorPlanes.pointToVertex({j + kj, i + ki});
            segmentatorPlanes.merge(top_left_block_corner, block_point);
          }
        }
      }
    }
    segmentatorPlanes.updateMapping();

    segmentatorPlanes.mergeToLimit(-1, errorLimit.getValue(), segmentsLimit.getValue(),
                                   dbg, debugIter.getValue(), maxSegments.getValue());

    merge_planes(segmentatorPlanes, maxAngle.getValue(), dbg);
    const ImageMap &imageMap = segmentatorPlanes.getImageMap();

    DECLARE_GUARDED_MINIMG(imgres);
    visualize(&imgres, imageMap);
    THROW_ON_MINERR(SaveMinImage(imgres_filename.c_str(), &imgres));
    save_mask(segmentatorPlanes, basename, output.getValue());
  }
  catch (std::exception const &e)
  {
    std::cerr << "Exception caught: " << e.what() << "\n";
    return 1;
  }
  catch (...)
  {
    std::cerr << "UNTYPED exception\n";
    return 2;
  }
  return 0;
}
