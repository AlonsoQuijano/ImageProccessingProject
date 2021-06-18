#include <algorithm>
#include <cmath>
#include <iostream>

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
#include <tclap/CmdLine.h>
THIRDPARTY_INCLUDES_END

namespace bfs = boost::filesystem;

using namespace vi::remseg;
using namespace roadseg;

void obtainBlockList(std::set<std::pair<int, int>> &blockList, size_t block_size, size_t width, size_t height)
{
  for (size_t i = 0; i < height; i += block_size)
    for (size_t j = 0; j < width; j += block_size)
      blockList.insert({j, i});
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
  TCLAP::ValueArg<double> blockingThresh("g", "blocking_thresh", "blocking threshold value", false, 1, "double", cmd);
  TCLAP::ValueArg<double> maxEigVal("", "max_eigen_val", "maximum eigen val to consider region as a plane", false, 1, "double", cmd);
  // TCLAP::ValueArg<double> glareThresh("", "glare_thresh", "glare threshold", false, 230, "double", cmd);
  // TCLAP::SwitchArg prefilter("p", "prefilter", "use image pre-filtering", cmd, false);

  cmd.parse(argc, argv);

  RoadVertex::setMaxFirstEigValThresh(maxEigVal.getValue());

  if (!bfs::is_directory(output.getValue()))
    throw std::runtime_error("Failed to find output directory " + output.getValue());

  if (debug.getValue())
    i8r::configure(output.getValue());

  std::string const basename = bfs::path(imagePath.getValue()).stem().string();
  std::string const imgres_filename = bfs::absolute(basename + ".png", output.getValue()).string();
  std::string const filtered_filename = bfs::absolute(basename + ".filtered.png", output.getValue()).string();

  try
  {
    mximg::PImage image = mximg::Image::imread(imagePath.getValue().c_str());
    if ((*image)->channels != 3)
      throw std::runtime_error("Image should have exact 3 channels for color segmentation");

    cv::Mat cv_image_filtered;
    auto dbg = i8r::logger("debug." + basename + ".pointlike");

    std::set<std::pair<int, int>> blockList;
    obtainBlockList(blockList, blockSize.getValue(), (*image)->width, (*image)->height);

    ImageMap im_map((*image)->width, (*image)->height);
    Segmentator<RoadVertex> segmentatorPlanes(*image, &im_map, error, criteria, blockList, BLOCK_SEGMENTS, true);
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
