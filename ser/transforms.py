from torchvision import transforms as torch_transforms


def _transforms(*stages):
    return torch_transforms.Compose(
        [
            *(stage() for stage in stages),
        ]
    )


def _normalize():
    """
    Normalize a tensor to have a mean of 0.5 and a std dev of 0.5
    """
    return torch_transforms.Normalize((0.5,), (0.5,))


def _flip():
    """
    Flip a tensor both vertically and horizontally
    """
    return torch_transforms.Compose(
        [
            torch_transforms.RandomHorizontalFlip(p=1.0),
            torch_transforms.RandomVerticalFlip(p=1.0),
            torch_transforms.ToTensor(),
        ]
    )


def transforms_by_option(lst):
    allowed_transforms = {'flip': _flip,
                          'normalise': _normalize}
    ts_to_remove = [t for t in lst if t not in allowed_transforms.keys()]
    if len(ts_to_remove) != 0:
        raise NameError(f'Unsupported transforms: {ts_to_remove}.')
    ts = []
    for k in allowed_transforms.keys():
        if k in lst:
            ts.append(allowed_transforms[k])
    return _transforms(*ts)
