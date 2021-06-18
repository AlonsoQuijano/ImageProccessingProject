treat_warnings_as_errors()

cmake_minimum_required(VERSION 3.0.0)

project(roadseg)

add_library(roadseg
  src/road_distance_func.cpp
  src/road_vertex.cpp
)

target_include_directories(remseg
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${PROJECT_BINARY_DIR}
)

target_link_libraries(roadseg
  mximg
  remseg
)

#----------------demo---------------------------------------

# if (Boost_FOUND AND OpenCV_FOUND)
#   add_executable(roadseg_go demo/roadseg_go.cpp)
#   target_link_libraries(roadseg_go
#     ${Boost_LIBRARIES}
#     ${OpenCV_LIBS}
#     roadseg)
# endif()