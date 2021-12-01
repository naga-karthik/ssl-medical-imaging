import torchvision.transforms as T
import torch
import random

state1 = None


class SimpleRandomRotation:
    """Rotate by one of the given angles."""

    def __init__(self):
        self.angles = [0, 45, 90, 135, 180, 225, 270, 315]

    def __call__(self, x):
        if random.randint(0, 5) != 0:
            return x
        angle = random.choice(self.angles)
        transform = T.RandomRotation((angle, angle))
        return transform(x)


class ElasticDeformation:
    """Rotate by one of the given angles."""

    def __init__(self, control_point_spacing, sigma):
        self.spacing = (control_point_spacing,) * 2  # 100
        self.sigmas = [sigma] * 2  # 10.0
        self.interpolation = "nearest"
        self.fill = None

    def __call__(self, img):

        torch.set_printoptions(threshold=10_000)
        """
        global state1
        if state1 is None:
            state1 = torch.get_rng_state()
        else:
            print(torch.eq(state1, torch.get_rng_state()))
            torch.set_rng_state(state1)
            state1 = None
        """
        shape = img.shape[-2:]

        control_points = tuple(
            max(1, int(round(float(shape[d]) / self.spacing[d])))
            for d in range(2)
        )

        control_point_offsets = torch.zeros((2,) + control_points)
        for d in range(2):
            if self.sigmas[d] > 0:
                control_point_offsets[d] = torch.randn(size=control_points) * self.sigmas[d] * 1 / (0.5 * shape[d])
        displacement = T.functional_tensor.resize(control_point_offsets, shape,
                                                  interpolation="bicubic").unsqueeze(-1).transpose(0,
                                                                                                   -1)  # 1 x H x W x 2

        hw_space = [torch.linspace((-s + 1) / s, (s - 1) / s, s) for s in shape]
        grid_y, grid_x = torch.meshgrid(hw_space, indexing='ij')
        identity_grid = torch.stack([grid_x, grid_y], -1).unsqueeze(0)  # 1 x H x W x 2
        grid = identity_grid.to(img.device) + displacement.to(img.device)
        return T.functional_tensor._apply_grid_transform(img, grid, self.interpolation, self.fill)


class CustomCompose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ]).

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img = t(img)
            mask = t(mask)
        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string