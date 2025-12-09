- Install the required packages
```
pip install "numpy>=1.23.5,<2"          # numpy-1.26.4
pip install omegaconf==2.2.3            # omegaconf-2.2.3
pip install opencv-python==4.8.1.78     # opencv-python-4.8.1.78

pip install --no-cache-dir -U pip setuptools wheel
pip install moviepy decord              # moviepy-2.2.1, decord-0.6.0
pip install --no-cache-dir -U openmim   # openmim-0.3.9
mim install mmengine                    # mmengine-0.10.7

# mim install "mmcv==2.1.0"               # mmcv-2.1.0, may fail
# mim uninstall mmcv -y
cd ComfyUI/custom_nodes/
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv && git checkout v2.1.0
pip install -r requirements/optional.txt
gcc --version                                                   # Check the gcc version (requires 5.4+)
MMCV_WITH_OPS=1 MAX_JOBS=$(nproc) python setup.py build_ext     # Build the C++ and CUDA extensions, may take a while
MMCV_WITH_OPS=1 MAX_JOBS=$(nproc) python setup.py develop       # Install mmcv with the C++ and CUDA extensions, in-place
# pip install -e . -v                                           # Install mmcv in editable mode
python .dev_scripts/check_installation.py                       # Verify the mmcv installation
cd ../

mim install "mmdet>=3.1.0"              # mmdet-3.3.0
pip install mmpose                      # mmpose-1.3.2

# Quick smoke test
python - <<'PY'
import mmcv, mmpose
from mmpose.apis import inference_topdown, init_model
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.structures import merge_data_samples
print("mmcv", mmcv.__version__, "mmpose", mmpose.__version__)
PY
```

- Download the pretrained weights
```
cd ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess
wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth -O steadydancer/pretrained_weights/dwpose/dw-ll_ucoco_384.pth
wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth -O steadydancer/pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth
```