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
  cv::Mat cv_xyz_image = cv::Mat::zeros(cv_depth_image.rows, cv_depth_image.cols, CV_32FC3);
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
      if (depth > 0)
      {
        float z = depth / 1000.;
        float x = z * (j - cx) / fx;
        float y = z * (i - cy) / fy;
        auto vec = cv_xyz_image.ptr<float>(i, j);
        vec[0] = x;
        vec[1] = y;
        vec[2] = z;
      }
    }
  cv::normalize(cv_xyz_image, cv_xyz_image, 0, 255, cv::NORM_MINMAX, CV_8UC3);
  cv::imwrite(data_path.parent_path().append("xyz_0_255.png").string(), cv_xyz_image);
  xyz_mximage = mximg::createByCopy(cv_xyz_image);
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
  TCLAP::ValueArg<double> maxEigVal("v", "max_eigen_val", "maximum eigen val to consider region as a plane", false, 1, "double", cmd);

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

    segmentatorPlanes.mergeToLimit(-1, errorLimit.getValue(), segmentsLimit.getValue(),
                                   dbg, debugIter.getValue(), maxSegments.getValue());

    const ImageMap &imageMap = segmentatorPlanes.getImageMap();

    DECLARE_GUARDED_MINIMG(imgres);
    visualize(&imgres, imageMap);
    THROW_ON_MINERR(SaveMinImage(imgres_filename.c_str(), &imgres));
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
