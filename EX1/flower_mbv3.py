# configs/my_configs/flower_mbv3.py

_base_ = [
     '../_base_/models/mobilenet_v3/mobilenet_v3_small_imagenet.py',
    '../_base_/default_runtime.py',
]

# 1. 模型配置
model = dict(
    head=dict(
        num_classes=5,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
        topk=(1,)
    )
)

# 2. 数据集配置
data_root = 'data/flower_dataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        data_prefix='train',
        ann_file='train.txt',
        classes='data/flower_dataset/classes.txt',
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        data_prefix='val',
        ann_file='val.txt',
        classes='data/flower_dataset/classes.txt',
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='Accuracy', topk=(1,))

# 3. 优化器与学习率策略
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=20, eta_min=1e-5, by_epoch=True, begin=5, end=100)
]

# 4. 训练配置
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()


# 5. 自动学习率缩放（如果更改了 batch_size）
auto_scale_lr = dict(base_batch_size=256)

# 6. 加载预训练模型（ImageNet 上训练好的 MobileNetV3 Small）
load_from = 'checkpoints/mobilenet-v3-small_8xb128_in1k_20221114-bd1bfcde.pth'
