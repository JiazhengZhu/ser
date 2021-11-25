import pytest
from math import isclose

from torchvision.transforms.functional import normalize

from ser.transforms import _flip, _normalize, _transforms, transforms_by_option
from PIL import Image, ImageOps
import numpy as np
from numpy import asarray
from torchvision.transforms import ToTensor


@pytest.fixture(autouse=True)
def img():
    return Image.open('test/4.2.03.tiff').convert('L')

# flip is concordant with PIL flip


def test_flip(img):
    output = _flip()(img)
    ground = ImageOps.flip(ImageOps.mirror(img))
    assert np.allclose(np.array(output), asarray(
        ground).astype('float32')/255, atol=1e-4)

# normalising gives correct mean and std


def test_norm(img):
    n = _normalize()
    output = n(ToTensor()(img))
    pixels = asarray(output)
    ground = (np.array(img)/255 - 0.5)/0.5
    assert np.allclose(pixels, ground, atol=1e-4)

# test transforms


def test_trans(img):
    t = _transforms(_flip, _normalize)
    output = t(img)
    ground = (asarray(ImageOps.mirror(ImageOps.flip(img)))/255 - 0.5) / 0.5
    assert np.allclose(asarray(output), ground, atol=1e-4)

# correct handling of names


def test_names(img):
    with pytest.raises(NameError):
        transforms_by_option(['fflip'])
    with pytest.raises(NameError):
        transforms_by_option(['flip', 'normalize'])
    output = transforms_by_option(['flip', 'normalise'])(img)
    ground = (asarray(ImageOps.mirror(ImageOps.flip(img)))/255 - 0.5) / 0.5
    assert np.allclose(asarray(output), ground, atol=1e-4)
