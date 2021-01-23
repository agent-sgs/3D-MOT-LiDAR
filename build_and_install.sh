# package for detector
pip install tensorboardX
pip install pybind11
pip install terminaltables
pip install tqdm
pip install pyyaml
pip install easydict
conda install numba
pip install cython # for cocoapttools
pip install scikit-image
pip install mmcv

# package for tracker
pip install filterpy==1.4.5
pip install pillow==6.2.2
#pip install opencv-python==3.4.3.18
pip install glob2==0.6
pip install joblib

cd tracker/lib/ops/
cd iou3d && python setup.py build_ext --inplace
cd ../iou3d_nms && python setup.py build_ext --inplace
cd -


