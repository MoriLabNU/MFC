_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_i2ib_to_Cholec_256x256.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]
seed = 2
model = dict(
    type='HRDAEncoderDecoder',
    decode_head=dict(
        type='HRDAHead',
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1),
    scales=[1, 0.5],
    hr_crop_size=(256, 256),
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[128, 128],
        crop_size=[256, 256]))
uda = dict(
    mask_generator=dict(mask_ratio=0.7, mask_block_size=32, preserved_size=8, _delete_=True))
optimizer_config = None
optimizer = dict(
    lr=3e-04,
)
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=1)
evaluation = dict(interval=10000, metric='mIoU')
name = 'SF_MFC_T2R'
exp = 'segmentation'
