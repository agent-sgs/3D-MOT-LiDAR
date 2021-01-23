import sys
sys.path.append('..')
import pathlib, pickle
import numpy as np
from SA_SSD.tools.viz_utils import draw_lidar

data_root = '/opt/projects/data/kitti/track/'
data_root = pathlib.Path(data_root)
data_path = data_root / 'training'

def get_points(info):
    v_path = info['velodyne_path']
    v_path = pathlib.Path(data_path) / 'velodyne' / v_path
    points_v = np.fromfile(
        str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
    points_v = points_v[np.random.choice(len(points_v), size=len(points_v) // 2)]
    return points_v


def main():

    train_info_path = data_root / 'kitti_infos_trainval.pkl'

    with open(train_info_path, 'rb') as f:
        kitti_infos = pickle.load(f)

    seq_info = kitti_infos[0]

    select = [1, 10, 23]
    info = seq_info[select[0]]
    Tr_velo_to_imu = np.linalg.inv(info['calib/Tr_imu_to_velo'])
    Tr_0_inv = np.linalg.inv(info['oxts'])

    points_list = []
    points = get_points(info)
    points[:, 3] = 1
    points_list.append(points @ Tr_velo_to_imu.T)

    for idx in select:
        info = seq_info[idx]
        Tr_velo_to_imu = np.linalg.inv(info['calib/Tr_imu_to_velo'])
        imu_to_world = Tr_0_inv @ info['oxts']

        points = get_points(info)

        points[:, 3] = 1
        points = points @ (Tr_velo_to_imu.T @ imu_to_world.T)
        points_list.append(points)


    fig = draw_lidar(points_list[0], fgcolor=(1., 0., 0.))
    fig = draw_lidar(points_list[1], fig, fgcolor=(0., 1., 0.), show=True)
    fig = draw_lidar(points_list[2], fig, fgcolor=(0., 0., 1.), show=True)

if __name__ == '__main__':
    main()