#include "bbox_util.h"

#include <algorithm>
#include <functional>
#include <map>
#include <vector>

// NMS implementation
template <typename Dtype>
bool SortScorePairDescend(const pair<float, Dtype>& pair1,
                          const pair<float, Dtype>& pair2) {
  return pair1.first > pair2.first;
}

template bool SortScorePairDescend(const pair<float, int>& pair1,
                                   const pair<float, int>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> >& pair1,
                                   const pair<float, pair<int, int> >& pair2);

template <typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
      const int top_k, vector<pair<Dtype, int> >* score_index_vec) {
  // Generate index score pairs.
  // num: 6375
  // threshold: 0.01
  for (int i = 0; i < num; ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::sort(score_index_vec->begin(), score_index_vec->end(),
            SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

template <typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized) {
  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return Dtype(0.);
  } else {
    const Dtype width = bbox[2] - bbox[0];
    const Dtype height = bbox[3] - bbox[1];
    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1) * (height + 1);
    }
  }
}

template <typename Dtype>
Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
      bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else {
    const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
    const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
    const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
    const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

    const Dtype inter_width = inter_xmax - inter_xmin;
    const Dtype inter_height = inter_ymax - inter_ymin;
    const Dtype inter_size = inter_width * inter_height;

    const Dtype bbox1_size = BBoxSize<Dtype>(bbox1);
    const Dtype bbox2_size = BBoxSize<Dtype>(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

template <typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices) {
  // Get top_k scores (with corresponding indices). -> pair<score, index>
  vector<pair<Dtype, int> > score_index_vec;
  GetMaxScoreIndex<Dtype>(scores, num, score_threshold, top_k, &score_index_vec);

  // Do nms.
  // score_index_vec -> score, index
  // indices -> keep bbox index
  float adaptive_threshold = nms_threshold; // nms_threshold: 0.45
  indices->clear();

  // please refer to SSD source code note
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    // idx bbox IOUs must be less than thresholds with all boxes
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap<Dtype>(bboxes + idx * 4, bboxes + kept_idx * 4);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
  }
}

template
void ApplyNMSFast(const float* bboxes, const float* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices);