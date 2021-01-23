import numpy as np, pathlib
from collections import namedtuple

# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

class PoseGenerator(object):
    """Generator to read OXTS ground truth data.

       Poses are given in an East-North-Up coordinate system
       whose origin is the first GPS position.
    """
    def __init__(self, oxts_file):
        with open(oxts_file, 'r') as f:
            lines = f.readlines()
        self.content = [line.strip().split(' ') for line in lines]

        # Scale for Mercator projection (from first lat value)
        self.scale = None
        # Origin of the global coordinate system (first GPS position)
        self.Tr_0_inv = None

    def __len__(self):
        return len(self.content)

    def get(self, idx):
        line = self.content[idx]
        # Last five entries are flags and counts
        line[:-5] = [float(x) for x in line[:-5]]
        line[-5:] = [int(float(x)) for x in line[-5:]]

        packet = OxtsPacket(*line)
        if self.scale is None:
            self.scale = np.cos(packet.lat * np.pi / 180.)

        R, t = self.pose_from_oxts_packet(packet, self.scale)
        pose = self.transform_from_rot_trans(R, t)
        # if self.Tr_0_inv is None:
        #     self.Tr_0_inv = np.linalg.inv(pose)
        # return self.Tr_0_inv @ pose
        return pose

    def pose_from_oxts_packet(self, packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet.
        """
        er = 6378137.  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * \
             np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        Rx = rotx(packet.roll)
        Ry = roty(packet.pitch)
        Rz = rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t

    def transform_from_rot_trans(self, R, t):
        """Transforation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.hstack([R, t])

if __name__ == '__main__':
    root_path = pathlib.Path('/opt/projects/data/kitti/track/training')
    Pose = PoseGenerator(root_path / 'oxts' / ('0001' + '.txt'))

    all_data = []
    for idx in range(177, 181):
        oxts = Pose.get(idx)
        all_data.append(oxts.reshape((1, 3, 4)))

    all_data = np.vstack(all_data).astype(np.float32)
    np.save(root_path / "0001_177_180.npy", all_data)
    print()