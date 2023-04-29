

model = dict(
    backbone='unet_backbone',
    num_classes=4,
    in_channels=3,
)

train = dict(
    num_epochs=20,
    # max_interval=10000,

)

setting = dict(
    optimizer=dict(type='Adam', lr=1e-3, weight_decay=0.00005),
    loss_fn=dict(type='Cross')
)






