#include <pybind11/pybind11.h>
// must include pybind11/eigen.h if using eigen matrix as arguments.
// must include pybind11/stl.h if using containers in STL in arguments.
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
 #include <vector>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <sparsehash/dense_hash_map>
#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
namespace py = pybind11;
using namespace pybind11::literals;
using google::dense_hash_map;      // namespace where class lives by default

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")

template <typename DType, int NDim>
int points_to_bev_kernel(py::array_t<DType> points,
                         std::vector<DType> voxel_size,
                         std::vector<DType> coors_range,
                         py::array_t<DType> bev,
                         int scale)
{
  auto points_rw = points.template mutable_unchecked<2>();
  auto N = points_rw.shape(0);
  auto bev_rw = bev.template mutable_unchecked<NDim>();

  int zdim_minus_1 = bev.shape(0)-1;
  int zdim_minus_2 = bev.shape(0)-2;

  constexpr int ndim_minus_1 = NDim - 1;
  bool failed = false;
  int coor[NDim];
  int c;
  int grid_size[NDim];
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  DType intensity;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed)
      continue;
    bev_rw(coor[0], coor[1], coor[2])=1;
    bev_rw(zdim_minus_1, coor[1], coor[2])+=1;
    intensity = points_rw(i, 3);
    if (intensity > bev_rw(zdim_minus_2, coor[1], coor[2]))
        bev_rw(zdim_minus_2, coor[1], coor[2])=intensity;

  }
  return 1;
}

template <typename DType>
py::array_t<bool> points_bound_kernel(py::array_t<DType> points,
                        std::vector<DType> lower_bound,
                        std::vector<DType> upper_bound
                        ){

    auto points_ptr = points.template mutable_unchecked<2>();

    int N = points_ptr.shape(0);
    int ndim = points_ptr.shape(1);
    auto keep = py::array_t<bool>(N);

    auto keep_ptr = keep.mutable_unchecked<1>();

    bool success = 0;
    for (int i = 0; i < N; i++){
        success = 1;
        for (int j=0; j<(ndim-1); j++){
            if(points_ptr(i, j) < lower_bound[j] || points_ptr(i, j) >= upper_bound[j]){
                success = 0;
                break;
            }
        }
        keep_ptr(i) = success;
    }
    return keep;
}

py::array_t<bool> coordinates_unique(py::array_t<int> coordinates,
                                     std::vector<int> grid_size){
    auto coordinates_ptr = coordinates.mutable_unchecked<2>();

    std::unordered_map<int, int> flag;
    int N = coordinates.shape(0);

    auto keep = py::array_t<bool>(N);
    auto keep_rw = keep.mutable_unchecked<1>();

    for (int i = 0; i < N; ++i){
      int idx = coordinates_ptr(i, 0) * grid_size[0] * grid_size[1] + \
                    coordinates_ptr(i, 1) * grid_size[0] + coordinates_ptr(i, 2);

      flag[idx] = i;
      keep_rw[i] = 0;
    }

    std::unordered_map<int, int>::iterator iter;
    for (iter = flag.begin(); iter != flag.end(); ++iter){
      keep_rw[iter->second] = 1;
    }

    return keep;
}

template <typename DType>
std::unordered_map<int, int> points_to_coordinates(py::array_t<DType>& points,
                                                   std::vector<DType> lower_bound,
                                                   std::vector<DType> upper_bound,
                                                   std::vector<DType> voxel_size,
                                                   std::vector<int> grid_size){
    auto points_ptr = points.template mutable_unchecked<2>();
    int N = points_ptr.shape(0);

    std::unordered_map<int, int> flag;

    bool success = 0;
    for (int i = 0; i < N; i++){
        success = 1;
        for (int j=0; j<3; j++){
            if(points_ptr(i, j) < lower_bound[j] || points_ptr(i, j) >= upper_bound[j]){
                success = 0;
                break;
            }
        }
        if (success){
             // calculate coordinates
            int w = (int)((points_ptr(i, 0) - lower_bound[0]) / voxel_size[0]);
            int l = (int)((points_ptr(i, 1) - lower_bound[1]) / voxel_size[1]);
            int h = (int)((points_ptr(i, 2) - lower_bound[2]) / voxel_size[2]);

            int indice = h * grid_size[0] * grid_size[1] + l * grid_size[0] + w;
            flag[indice] = i;
        }
    }

    return flag;
}


int get_coordinates(py::array_t<int>& coordinates,
                    std::unordered_map<int, int>& flag,
                    std::vector<int> grid_size){

    auto coordinates_rw = coordinates.mutable_unchecked<2>();

    int cnt = 0;
    std::unordered_map<int, int>::iterator iter;
    for (iter = flag.begin(); iter != flag.end(); ++iter){
        int indice = iter->first;

        coordinates_rw(cnt, 0) = indice / grid_size[0] / grid_size[1];
        coordinates_rw(cnt, 1) = (indice / grid_size[0]) % grid_size[1];
        coordinates_rw(cnt, 2) = indice % grid_size[0];

        cnt++;
    }
    return cnt;
}

int get_voxels(py::array_t<float>& points,
               py::array_t<float>& voxels,
               std::unordered_map<int, int> flag){

    auto points_ptr = points.template mutable_unchecked<2>();
    const int ndim = points_ptr.shape(1);

    auto voxels_rw = voxels.template mutable_unchecked<2>(); // x y z i

    int cnt = 0;
    std::unordered_map<int, int>::iterator iter;
    for (iter = flag.begin(); iter != flag.end(); ++iter){
        int idx = iter->second;

        for (int j = 0; j < ndim; j++){
            voxels_rw(cnt, j) = points_ptr(idx, j);
        }
        cnt++;
    }

    return cnt;
}


int pt_in_box3d_cpu(float x, float y, float z, float cx, float cy, float bottom_z, float w, float l, float h, float angle){
    float max_dis = 10.0, x_rot, y_rot, cosa, sina, cz;
    int in_flag;
    cz = bottom_z + h / 2.0;
    if ((fabsf(x - cx) > max_dis) || (fabsf(z - cz) > h / 2.0) || (fabsf(y - cy) > max_dis)){
        return 0;
    }
    cosa = cos(angle); sina = sin(angle);
    x_rot = (x - cx) * cosa + (y - cy) * (-sina);
    y_rot = (x - cx) * sina + (y - cy) * cosa;

    in_flag = (x_rot >= -w / 2.0) & (x_rot <= w / 2.0) & (y_rot >= -l / 2.0) & (y_rot <= l / 2.0);
    return in_flag;
}

int pts_in_boxes3d_cpu(at::Tensor pts, at::Tensor boxes3d, at::Tensor pts_flag, at::Tensor reg_target){
    // param pts: (N, 3)
    // param boxes3d: (M, 7)  [x, y, z, h, w, l, ry]
    // param pts_flag: (M, N)
    // param reg_target: (N, 3), center offsets

    CHECK_CONTIGUOUS(pts_flag);
    CHECK_CONTIGUOUS(pts);
    CHECK_CONTIGUOUS(boxes3d);
    CHECK_CONTIGUOUS(reg_target);

    long boxes_num = boxes3d.size(0);
    long pts_num = pts.size(0);

    int * pts_flag_flat = pts_flag.data<int>();
    float * pts_flat = pts.data<float>();
    float * boxes3d_flat = boxes3d.data<float>();
    float * reg_target_flat = reg_target.data<float>();

//    memset(assign_idx_flat, -1, boxes_num * pts_num * sizeof(int));
//    memset(reg_target_flat, 0, pts_num * sizeof(float));

    int i, j, cur_in_flag;
    for (i = 0; i < boxes_num; i++){
        for (j = 0; j < pts_num; j++){
            cur_in_flag = pt_in_box3d_cpu(pts_flat[j * 3], pts_flat[j * 3 + 1], pts_flat[j * 3 + 2], boxes3d_flat[i * 7],
                                          boxes3d_flat[i * 7 + 1], boxes3d_flat[i * 7 + 2], boxes3d_flat[i * 7 + 3],
                                          boxes3d_flat[i * 7 + 4], boxes3d_flat[i * 7 + 5], boxes3d_flat[i * 7 + 6]);
            pts_flag_flat[i * pts_num + j] = cur_in_flag;
            if(cur_in_flag==1){
                reg_target_flat[j*3] = pts_flat[j*3] - boxes3d_flat[i*7];
                reg_target_flat[j*3+1] = pts_flat[j*3+1] - boxes3d_flat[i*7+1];
                reg_target_flat[j*3+2] = pts_flat[j*3+2] - (boxes3d_flat[i*7+2] + boxes3d_flat[i*7+3] / 2.0);
            }
        }
    }
    return 1;
}

template <typename DType>
py::array_t<bool> points_in_bbox3d_np(py::array_t<DType> points, py::array_t<DType> boxes3d)
{

    //const DType* points_ptr = static_cast<DType*>(points.request().ptr);
    //const DType* boxes3d_ptr = static_cast<DType*>(boxes3d.request().ptr);
    //const DType* boxes3d_ptr = boxes3d.data();

    auto points_ptr = points.template mutable_unchecked<2>();
    auto boxes3d_ptr = boxes3d.template mutable_unchecked<2>();

    int N = points.shape(0);
    int M = boxes3d.shape(0);

    auto keep = py::array_t<bool>({N,M});

    //int * keep_ptr = keep.mutable_data();
    //int * keep_ptr = static_cast<int*>(keep.request().ptr);
    auto keep_ptr = keep.mutable_unchecked<2>();

    int i, j, cur_in_flag;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            cur_in_flag = pt_in_box3d_cpu(
                points_ptr(j, 0), points_ptr(j, 1), points_ptr(j, 2),
                boxes3d_ptr(i, 0), boxes3d_ptr(i, 1), boxes3d_ptr(i, 2),
                boxes3d_ptr(i, 3), boxes3d_ptr(i, 4), boxes3d_ptr(i, 5),
                boxes3d_ptr(i, 6)
            );

            keep_ptr(j, i) = (bool) cur_in_flag;
        }
    }

    return keep;
}

template <typename DType>
int create_unique_voxels_input(py::array_t<DType>& points,
                               py::array_t<int>& coordinates,
                               std::vector<DType> lower_bound,
                               std::vector<DType> upper_bound,
                               std::vector<DType> voxel_size,
                               std::vector<int> grid_size){

    auto points_ptr = points.template mutable_unchecked<2>();

    int N = points_ptr.shape(0);
    int ndim = points_ptr.shape(1);

    auto coordinates_ptr = coordinates.mutable_unchecked<2>();

//    typedef std::unordered_map<int, int> M;
    typedef dense_hash_map<int, int> M;
    M flag;
    flag.set_empty_key(NULL);

    int cnt = 0;
    bool success = 0;
    for (int i = 0; i < N; i++){
        success = 1;

        for (int j=0; j<3; j++){
            if(points_ptr(i, j) < lower_bound[j] || points_ptr(i, j) >= upper_bound[j]){
                success = 0;
                break;
            }
        }

       if (success){
           // calculate coordinates
           int x_idx = (int)((points_ptr(i, 0) - lower_bound[0]) / voxel_size[0]);
           int y_idx = (int)((points_ptr(i, 1) - lower_bound[1]) / voxel_size[1]);
           int z_idx = (int)((points_ptr(i, 2) - lower_bound[2]) / voxel_size[2]);

           int indice = z_idx * grid_size[0] * grid_size[1] + y_idx * grid_size[0] + x_idx;
           std::pair<M::iterator, bool> const& r = flag.insert(M::value_type(indice, cnt));

           if (r.second) {
              // value are inserted
               coordinates_ptr(cnt, 0) = z_idx;
               coordinates_ptr(cnt, 1) = y_idx;
               coordinates_ptr(cnt, 2) = x_idx;

               for (int j = 0; j < ndim; ++j){
                    points_ptr(cnt, j) = points_ptr(i, j);
               }

               cnt++;
           }
       }
    }

    return cnt;
}

template <typename DType>
int create_mean_voxels_input(py::array_t<DType>& voxels,
                             py::array_t<int>& coords,
                             py::array_t<int>& counts,
                             std::vector<DType> lower_bound,
                             std::vector<DType> upper_bound,
                             std::vector<DType> voxel_size,
                             std::vector<int> grid_size) {

    auto voxels_ptr = voxels.template mutable_unchecked<2>();
    int N = voxels_ptr.shape(0);
    int ndim = voxels_ptr.shape(1);

    auto coords_ptr = coords.mutable_unchecked<2>();
    auto counts_ptr = counts.mutable_unchecked<1>();
//  typedef std::unordered_map<int, int> M;
    typedef dense_hash_map<int, int> M;

    M flag;
    flag.set_empty_key(NULL);

    int cnt = 0;
    bool success = 0;
    for (int i = 0; i < N; i++){
      success = 1;

      for (int j=0; j<3; j++){
        if(voxels_ptr(i, j) < lower_bound[j] || voxels_ptr(i, j) >= upper_bound[j]) {
          success = 0;
          break;
        }
      }

      if (success) {
         // calculate coords
        int x_idx = (int)((voxels_ptr(i, 0) - lower_bound[0]) / voxel_size[0]);
        int y_idx = (int)((voxels_ptr(i, 1) - lower_bound[1]) / voxel_size[1]);
        int z_idx = (int)((voxels_ptr(i, 2) - lower_bound[2]) / voxel_size[2]);

        x_idx = std::max(0, std::min(x_idx, grid_size[0] - 1));
        y_idx = std::max(0, std::min(y_idx, grid_size[1] - 1));
        z_idx = std::max(0, std::min(z_idx, grid_size[2] - 1));

        int indice = z_idx * grid_size[0] * grid_size[1] + y_idx * grid_size[0] + x_idx;
        std::pair<M::iterator, bool> const& r = flag.insert(M::value_type(indice, cnt));
        if (r.second) {
          // value are inserted
          coords_ptr(cnt, 0) = z_idx;
          coords_ptr(cnt, 1) = y_idx;
          coords_ptr(cnt, 2) = x_idx;

          for (int j=0; j<ndim; ++j)
            voxels_ptr(cnt, j) = voxels_ptr(i, j);
          counts_ptr(cnt) = 1;
          cnt++;
        }
        else {
          // key exists, sum
          for (int j=0; j<ndim; ++j)
            voxels_ptr(r.first->second, j) += voxels_ptr(i, j);
          counts_ptr(r.first->second) += 1;
        }
      }
    }
//    for (int i = 0; i < cnt; ++i){
//      for (int j=0; j< ndim; ++j)
//        voxels_ptr(i, j) /= temp_num_ptr(i);
//    }
    return cnt;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "pybind11 example plugin";      // module doc string
  // m.def("points_to_bev_kernel",                              // function name
        // &points_to_bev_kernel<float,3>,                               // function pointer
        // "function of converting points to voxel" //function doc string
       // );
  m.def("points_bound_kernel",                              // function name
        &points_bound_kernel<float>,                               // function pointer
        py::return_value_policy::reference,
        "function of filtering points" //function doc string
       );
  m.def("coordinates_unique",                              // function name
       &coordinates_unique,                             // function pointer
       py::return_value_policy::reference,
       "function of filtering points" //function doc string
       );
  m.def("points_to_voxel_3d_unique",
        &create_unique_voxels_input<float>,
        py::return_value_policy::reference,
        "convert points to voxels and unique coordinates"
  );
  m.def("points_to_voxel_3d_mean",
        &create_mean_voxels_input<float>,
        py::return_value_policy::reference,
        "convert points to voxels with mean points within the same voxels and unique coordinates"
  );
//  m.def("get_coordinates",
//       &get_coordinates,
//       py::return_value_policy::reference,
//       "fetch coordinates from map dict"
//  );
//  m.def("get_voxels",
//       &get_voxels,
//       py::return_value_policy::reference,
//       "fetch voxels feature from map dict"
//  );
  m.def("pts_in_boxes3d", &pts_in_boxes3d_cpu, "points in boxes3d (CPU)");
  m.def("points_in_bbox3d_np", &points_in_bbox3d_np<float>, "points in boxes3d using numpy (CPU)");
}
