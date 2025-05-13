launcher = 'none'
load_from = 'checkpoints/convnext-base_3rdparty_in21k_20220124-13b83eec.pth'
model = dict(
    head=dict(
        dropout_rate=0.5,
        loss=dict(label_smooth_val=0.2, type='LabelSmoothLoss'),
        num_classes=5))
optim_wrapper = dict(
    clip_grad=None, optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05))
optimizer = dict(lr=0.0001, type='AdamW', weight_decay=0.05)
param_scheduler = [
    dict(begin=0, by_epoch=True, end=10, start_factor=0.01, type='LinearLR'),
    dict(T_max=30, begin=10, end=40, eta_min=1e-06, type='CosineAnnealingLR'),
]
train_cfg = dict(max_epochs=40)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/train.txt',
        classes=[
            'daisy',
            'dandelion',
            'roses',
            'sunflowers',
            'tulips',
        ],
        data_root='data/flower_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=224, type='RandomResizedCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                type='ColorJitter'),
            dict(ratio=(
                0.05,
                0.2,
            ), type='RandomErasing'),
            dict(type='PackInputs'),
        ],
        type='ImageNet'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='RandomResizedCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(brightness=0.4, contrast=0.4, saturation=0.4, type='ColorJitter'),
    dict(ratio=(
        0.05,
        0.2,
    ), type='RandomErasing'),
    dict(type='PackInputs'),
]
work_dir = 'work_dirs/flower_mbv3'
