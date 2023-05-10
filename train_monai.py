import torch

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference



# Define model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    kernel_size=(3,3,3)
).cuda()
print(model)

# Define loss function and optimizer
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Define metrics
dice_metric = DiceMetric(include_background=True, reduction="mean")

# Train the model
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    for batch_idx in range(10):
        inputs, targets = (
            torch.randn(2,1,96,160,160).cuda(), 
            torch.round(torch.randn(2,1,96,160,160)).cuda()
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        dice_metric(y_pred=outputs, y=targets)
        epoch_dice += dice_metric.aggregate().item()
        dice_metric.reset()
        print(f"Loss: {loss.item()}")