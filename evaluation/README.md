# kitti-object/mot-eval-python
**Note**: The kitti-object-eval-python is borrowed from [traveller59/kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python)

The kitti-mot-eval-python is builded on top of [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT/tree/master/evaluation)

## Dependencies
Only support python 3.6+

```
pip install tqdm scipy scikit-image
conda install numba
```

## Usage
* evaluate 2D/2D BEV/3D MOT for predicted 3D trajectories on KITTI 3D dataset:
```
# python evaluation/evaluate_kitti3dmot.py result_path eval_type
python evaluation/evaluate_kitti3dmot.py results/PVRCNN/ 2D
python evaluation/evaluate_kitti3dmot.py results/PVRCNN/ BEV
python evaluation/evaluate_kitti3dmot.py results/PVRCNN/ 3D
```
* 3D Object evaluation tools :
```Python
# python evaluation/evaluate_kitti3dobject.py split result_path 
python evaluation/evaluate_kitti3dobject.py val results/PVRCNN/data/ --class_name Car
```
