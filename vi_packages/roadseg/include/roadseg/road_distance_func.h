#pragma once

#include <roadseg/road_vertex.h>
#include <remseg/edge_heap.h>

namespace roadseg {

const double ME = 0.5;

EdgeValue criteria(const RoadVertex *v1, const RoadVertex *v2);

EdgeValue error(const RoadVertex *v);

bool isUnionLowVariance(const RoadVertex *v1, const RoadVertex *v2);

}	// ns roadseg
