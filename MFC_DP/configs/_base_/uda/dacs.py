# Obtained from: https://github.com/lhoyer/HRDA

uda = dict(
    type='DACS',
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=0.75,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    mask_generator=None,
    debug_img_interval=0,
    print_grad_magnitude=False,
)
use_ddp_wrapper = True
