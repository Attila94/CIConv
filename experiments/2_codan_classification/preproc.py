import torch

EPS = 1e-5

class log(object):
    """
    Log-scaling and channel-wise normalisation.

    """
    def __call__(self, tensor):
        tensor = torch.log(tensor + EPS)
        return tensor

class norm(object):
    """
    Channel-wise normalisation.

    """
    def __call__(self, tensor):
        tensor -= torch.mean(tensor, dim=0, keepdim=True)
        tensor /= torch.std(tensor, dim=0, keepdim=True) + EPS
        return tensor

class nrgb(object):
    """
    Normalized RGB.

    """
    def __call__(self, tensor):
        return tensor / (torch.sum(tensor, dim=0, keepdim=True) + EPS)

class comp(object):
    """
    Comprehensive Colour Image Normalization, Finlayson et al.

    """
    def __call__(self, tensor, n_iter=6):
        for i in range(n_iter):
            tensor = tensor / (torch.sum(tensor, dim=0, keepdim=True) + EPS)
            tensor = tensor / (torch.mean(tensor, dim=(1,2), keepdim=True) + EPS)

        return tensor

class alvarez(object):
    """
    Road Detection based on Illuminant Invariance
    Alvarez and Lopez (2011)
    """

    def log_app(x, a=5000):
        # computes logaritmic approximation
        return 5000 * (x ** (1/a) - 1)

    def __call__(self, tensor, angle=0.45):
        angle = torch.tensor(angle)
        tensor = torch.clamp(tensor, min=EPS)
        tensor = torch.cos(angle)*torch.log(tensor[0,:,:] / (tensor[2,:,:] + EPS)) + torch.sin(angle)*torch.log(tensor[1,:,:] / (tensor[2,:,:]+EPS))
        tensor = tensor[None,:,:]
        ts = tensor.shape
        tensor = tensor.expand(3,ts[1],ts[2]).clone()
        return tensor

class maddern(object):
    """
    Illumination Invariant Imaging: Applications in Robust Vision-based
    Localisation, Mapping and Classiﬁcation for Autonomous Vehicles
    Maddern et al. (2009)
    """

    def log_app(x, a=5000):
        # computes logaritmic approximation
        return 5000 * (x ** (1/a) - 1)

    def __call__(self, tensor, angle=0.45):
        angle = torch.tensor(angle)
        tensor = torch.clamp(tensor, min=EPS)
        tensor = 0.5 + torch.log(tensor[1,:,:]) - angle * torch.log(tensor[2,:,:]) - (1 - angle) * torch.log(tensor[0,:,:])
        tensor = tensor[None,:,:]
        ts = tensor.shape
        tensor = tensor.expand(3,ts[1],ts[2]).clone()
        return tensor
