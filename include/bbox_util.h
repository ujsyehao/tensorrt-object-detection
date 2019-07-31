#ifndef BBOX_UTIL_H
#define BBOX_UTIL_H

#include <iostream>
#include <stdint.h>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "detection_output.h"

using namespace std;

template <typename Dtype>
bool SortScorePairDescend(const pair<float, Dtype>& pair1, const pair<float, Dtype>& pair2);

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



template <typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
      const int top_k, vector<pair<Dtype, int> >* score_index_vec);

template <typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized = true);

template <typename Dtype>
Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2);

template <typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices);

template <typename Dtype>
void ApplySoftNMSFast(const Dtype* bboxes, Dtype* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices);

#endif
