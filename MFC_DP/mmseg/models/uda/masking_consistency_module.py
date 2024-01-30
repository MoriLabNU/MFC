import random
import torch
import numpy as np
from torch.nn import Module
from mmseg.ops import resize
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform


def get_gaussian_kernel_1d(kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def get_gaussian_kernel_2d(kernel_size, sigma, dtype, device):
    kernel_1d_x = get_gaussian_kernel_1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel_1d_y = get_gaussian_kernel_1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    return torch.mm(kernel_1d_y[:, None], kernel_1d_x[None, :])


class MFCMaskGenerator:
    def __init__(self, mask_ratio, mask_block_size, preserved_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.preserved_size = preserved_size

    @torch.no_grad()
    def shift(self, x, axes=(2, 3), inverse=False):
        if not torch.is_tensor(x):
            raise TypeError("Input must be a tensor.")

        if axes is None:
            axes = tuple(range(x.ndim))
            shift = [(-dim // 2 if inverse else dim // 2) for dim in x.shape]
        elif isinstance(axes, int):
            shift = -(x.shape[axes] // 2) if inverse else x.shape[axes] // 2
        else:
            shift = [(-x.shape[axis] // 2 if inverse else x.shape[axis] // 2) for axis in axes]
        return torch.roll(x, shift, axes)

    @torch.no_grad()
    def fft_mask(self, x):
        B, _, H, W = x.shape
        mshape = B, 1, round(H / abs(self.mask_block_size)), round(W / abs(self.mask_block_size))
        input_mask = (torch.rand(mshape, device=x.device) > abs(self.mask_ratio)).float()
        input_mask2 = torch.zeros(B, 1, H, W, device=x.device)
        input_mask2[:, :, H // 2 - self.preserved_size: H // 2 + self.preserved_size,
        W // 2 - self.preserved_size: W // 2 + self.preserved_size] = 1
        fft_mask = resize(input_mask, size=(H, W)) + input_mask2

        x_fft = torch.fft.fftn(x, dim=(2, 3), norm='ortho')
        x_fft_shift = self.shift(x_fft)
        x_fft_filter = fft_mask * x_fft_shift
        x_ifft = torch.fft.ifftn(self.shift(x_fft_filter, inverse=True), dim=(2, 3), norm='ortho').abs()
        return x_ifft


class MaskingConsistencyModule(Module):
    def __init__(self, cfg):
        super(MaskingConsistencyModule, self).__init__()
        self.max_iters = cfg['max_iters']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_gen = MFCMaskGenerator(**cfg['mask_generator'])
        self.aug = True

        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def __call__(self, model, img, img_metas, gt_semantic_seg, target_img, target_img_metas, valid_pseudo_mask,
                 pseudo_label, pseudo_weight):
        if pseudo_label is None or pseudo_weight is None:
            raise ValueError("Pseudo label and weight must not be None.")

        masked_img = target_img.clone()
        masked_lbl = pseudo_label.unsqueeze(1)
        masked_seg_weight = pseudo_weight

        if self.aug:
            dev = img.device
            means, stds = get_mean_std(img_metas, dev)
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_s,
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
            masked_img, _ = strong_transform(strong_parameters, data=masked_img.clone())

        masked_img = self.mask_gen.fft_mask(masked_img)

        masked_loss = model.forward_train(masked_img, img_metas, masked_lbl, seg_weight=masked_seg_weight)
        return masked_loss
