#ifndef __BBOX_UTIL_GPU_H_
#define __BBOX_UTIL_GPU_H_

#include "detection_output.h"

template <typename Dtype>
void CasRegDecodeBBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, Dtype* bbox_data, const Dtype* arm_loc_data);

template <typename Dtype>
void OSPermuteDataGPU(const int nthreads,
          const Dtype* data, const Dtype* arm_data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data, float objectness_score);

#endif
