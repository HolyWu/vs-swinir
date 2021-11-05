# SwinIR
SwinIR function for VapourSynth, based on https://github.com/JingyunLiang/SwinIR.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchvision` and `torchaudio` are not required and hence can be omitted from the command.
- [VapourSynth](http://www.vapoursynth.com/)


## Installation
```
pip install --upgrade vsswinir
python -m vsswinir
```


## Usage
```python
from vsswinir import SwinIR

ret = SwinIR(clip)
```

See `__init__.py` for the description of the parameters.
