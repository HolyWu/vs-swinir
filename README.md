# SwinIR
Image Restoration Using Swin Transformer, based on https://github.com/JingyunLiang/SwinIR.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started) 1.13
- [VapourSynth](http://www.vapoursynth.com/) R55+


## Installation
```
pip install -U vsswinir
python -m vsswinir
```


## Usage
```python
from vsswinir import swinir

ret = swinir(clip)
```

See `__init__.py` for the description of the parameters.
