import numpy as np
from . import points_op

class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._grid_size = grid_size

    def generate(self, points):
        counts = np.zeros(points.shape[0], dtype=np.int32)

        voxels = points.copy()
        coordinates = np.zeros((voxels.shape[0], 3), dtype=np.int32)  # zyx
        cnt = points_op.points_to_voxel_3d_mean(voxels, coordinates, counts,
                                                self.point_cloud_range[:3], self.point_cloud_range[3:],
                                                self.voxel_size, self.grid_size)
        coordinates = coordinates[:cnt, :]
        voxels = voxels[:cnt, :]
        counts = counts[:cnt]
        voxels /= counts.reshape(-1, 1)
        res = dict(
            voxels=voxels,
            points=points,  # zyx
            coordinates=coordinates,
        )

        return res

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
