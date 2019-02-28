#include "bbox_util_gpu.h"

#include <algorithm>
#include <functional>
#include <map>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>
#include <cudnn.h>

#include <glog/logging.h>

#include "detection_output.h"

#define CUDA_CHECK(condition) \
    do { \
        cudaError_t error = condition; \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void DecodeBBoxesKernel(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, Dtype* bbox_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % 4;

    //printf("###################################\n");
    //printf("i value: %d \n", i);
    //printf("num_loc_classes value: %d\n", num_loc_classes);
    //printf("num_priors value: %d\n", num_priors);

    const int c = (index / 4) % num_loc_classes;
    const int d = (index / 4 / num_loc_classes) % num_priors;
    
    if (!share_location && c == background_label_id) {
      // Ignore background class if not share_location.
      return;
    }
    const int pi = d * 4;
    const int vi = pi + num_priors * 4;
    if (code_type == CodeType::CORNER) {
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index];
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i];
      }
    } else if (code_type == CodeType::CENTER_SIZE) {
      //printf("test test test\n");
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      //printf("pi %d x1 %f y1 %f x2 %f y2 %f\n", pi, p_xmin, p_ymin, p_xmax, p_ymax);
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
      const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

      const Dtype xmin = loc_data[index - i];
      const Dtype ymin = loc_data[index - i + 1];
      const Dtype xmax = loc_data[index - i + 2];
      const Dtype ymax = loc_data[index - i + 3];

      Dtype decode_bbox_center_x, decode_bbox_center_y;
      Dtype decode_bbox_width, decode_bbox_height;
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to retore the offset
        // predictions.
        decode_bbox_center_x = xmin * prior_width + prior_center_x;
        decode_bbox_center_y = ymin * prior_height + prior_center_y;
        decode_bbox_width = exp(xmax) * prior_width;
        decode_bbox_height = exp(ymax) * prior_height;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        decode_bbox_center_x =
          prior_data[vi] * xmin * prior_width + prior_center_x;
        decode_bbox_center_y =
          prior_data[vi + 1] * ymin * prior_height + prior_center_y;
        decode_bbox_width =
          exp(prior_data[vi + 2] * xmax) * prior_width;
        decode_bbox_height =
          exp(prior_data[vi + 3] * ymax) * prior_height;
        //printf("x %f , y %f, w %f, h %f\n" , decode_bbox_center_x, decode_bbox_center_y, decode_bbox_width, decode_bbox_height);
      }

      switch (i) {
        case 0:
          bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;
          break;
        case 1:
          bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;
          break;
        case 2:
          bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;
          break;
        case 3:
          bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;
          break;
      }
    } else if (code_type == CodeType::CORNER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      Dtype p_size;
      if (i == 0 || i == 2) {
        p_size = prior_width;
      } else {
        p_size = prior_height;
      }
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index] * p_size;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i] * p_size;
      }
    } else {
      // Unknown code type.
    }
    if (clip_bbox) {
      bbox_data[index] = max(min(bbox_data[index], Dtype(1.)), Dtype(0.));
    }
  }
}

template <typename Dtype>
__global__ void CasRegDecodeBBoxesKernel(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, Dtype* bbox_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % 4;
    const int c = (index / 4) % num_loc_classes;
    const int d = (index / 4 / num_loc_classes) % num_priors;
    if (!share_location && c == background_label_id) {
      // Ignore background class if not share_location.
      return;
    }
    const int pi = d * 4;
    const int vi = pi + num_priors * 4;
    if (code_type == CodeType::CORNER) {
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = bbox_data[index] + loc_data[index];
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          bbox_data[index] + loc_data[index] * prior_data[vi + i];
      }
    } else if (code_type == CodeType::CENTER_SIZE) {
      // get normalized bbox_data coordinate [x1, y1, x2, y2]
      const Dtype p_xmin = bbox_data[index - i];
      const Dtype p_ymin = bbox_data[index - i + 1];
      const Dtype p_xmax = bbox_data[index - i + 2];
      const Dtype p_ymax = bbox_data[index - i + 3];
      // convert to x_ctr, y_ctr, w, h
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
      const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

      // get odm_loc prediction coordinate: [tx, ty, tw, th]
      const Dtype xmin = loc_data[index - i];
      const Dtype ymin = loc_data[index - i + 1];
      const Dtype xmax = loc_data[index - i + 2];
      const Dtype ymax = loc_data[index - i + 3];

      Dtype decode_bbox_center_x, decode_bbox_center_y;
      Dtype decode_bbox_width, decode_bbox_height;
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to retore the offset
        // predictions.
        decode_bbox_center_x = xmin * prior_width + prior_center_x;
        decode_bbox_center_y = ymin * prior_height + prior_center_y;
        decode_bbox_width = exp(xmax) * prior_width;
        decode_bbox_height = exp(ymax) * prior_height;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        decode_bbox_center_x =
          prior_data[vi] * xmin * prior_width + prior_center_x;
        decode_bbox_center_y =
          prior_data[vi + 1] * ymin * prior_height + prior_center_y;
        decode_bbox_width =
          exp(prior_data[vi + 2] * xmax) * prior_width;
        decode_bbox_height =
          exp(prior_data[vi + 3] * ymax) * prior_height;
      }

      switch (i) {
        case 0:
          bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;
          break;
        case 1:
          bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;
          break;
        case 2:
          bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;
          break;
        case 3:
          bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;
          break;
      }
    } else if (code_type == CodeType::CORNER_SIZE) {
      const Dtype p_xmin = bbox_data[index - i];
      const Dtype p_ymin = bbox_data[index - i + 1];
      const Dtype p_xmax = bbox_data[index - i + 2];
      const Dtype p_ymax = bbox_data[index - i + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      Dtype p_size;
      if (i == 0 || i == 2) {
        p_size = prior_width;
      } else {
        p_size = prior_height;
      }
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = bbox_data[index] + loc_data[index] * p_size;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          bbox_data[index] + loc_data[index] * prior_data[vi + i] * p_size;
      }
    } else {
      // Unknown code type.
    }
    if (clip_bbox) {
      bbox_data[index] = max(min(bbox_data[index], Dtype(1.)), Dtype(0.));
    }
  }
}

template <typename Dtype>
void CasRegDecodeBBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, Dtype* bbox_data, const Dtype* arm_loc_data) {
  // loc_data -> odm_loc
  // prior_data -> arm_priorbox format: [1, 2, layer_height * layer_width * num_priors * 4]
  //            -> which stores normalized [x1, y1, x2, y2]
  // arm_loc_data -> arm_loc
  // loc_count -> odm_loc count()
  // bbox_data -> odm_loc data()

  // note: bbox_data stores normalized decoded coordinates[Predict_x1, Predict_y1, Predict_x2, Predict_y2]

  

  DecodeBBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, arm_loc_data, prior_data, code_type,
      variance_encoded_in_target, num_priors, share_location, num_loc_classes,
      background_label_id, clip_bbox, bbox_data);
  CUDA_POST_KERNEL_CHECK;




  CasRegDecodeBBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, loc_data, prior_data, code_type,
      variance_encoded_in_target, num_priors, share_location, num_loc_classes,
      background_label_id, clip_bbox, bbox_data);
  CUDA_POST_KERNEL_CHECK;

  // test code
  /*float* tmp = new float[100];
  cudaMemcpy(tmp, bbox_data, 100 * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "!!!!!! bbox_data !!!!!!!!" << std::endl;
  for (int i = 0; i < 100; i++)
    std::cout << tmp[i] << " ";
  std::cout << std::endl;*/
}

template void CasRegDecodeBBoxesGPU(const int nthreads,
  const float* loc_data, const float* prior_data,
  const CodeType code_type, const bool variance_encoded_in_target,
  const int num_priors, const bool share_location,
  const int num_loc_classes, const int background_label_id,
  const bool clip_bbox, float* bbox_data, const float* arm_loc_data);

template <typename Dtype>
__global__ void OSPermuteDataKernel(const int nthreads,
          const Dtype* data, const Dtype* arm_data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data, float objectness_score) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % num_dim;
    const int c = (index / num_dim) % num_classes;
    const int d = (index / num_dim / num_classes) % num_data;
    const int n = index / num_dim / num_classes / num_data;
    const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;

    const int arm_index = ((n * num_data + d) * 2 + 1) * num_dim + i;
    if (arm_data[arm_index] < objectness_score) {
      if (c == 0)
        new_data[new_index] = 1.0;
      else        
        new_data[new_index] = 0.0;
    } else {
      new_data[new_index] = data[index];
    }
  }
}

template <typename Dtype>
void OSPermuteDataGPU(const int nthreads,
          const Dtype* data, const Dtype* arm_data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data, float objectness_score) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  OSPermuteDataKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, data, arm_data, num_classes, num_data,
      num_dim, new_data, objectness_score);
  CUDA_POST_KERNEL_CHECK;
}

template void OSPermuteDataGPU(const int nthreads,
  const float* data, const float* arm_data, const int num_classes, const int num_data,
  const int num_dim, float* new_data, float objectness_score);

