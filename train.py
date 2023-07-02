from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path
import torch
from layers import num_to_groups
from train_utils import *
from unet import *
from torchvision.utils import save_image
from torch.optim import Adam
import os
import shutil
import torchvision.transforms as T
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms
import torchvision


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# load dataset from the hub
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
image_size = 32
channels = 1
batch_size = 64
from PIL import Image


def sample_to_pil(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


torch.manual_seed(0)

inference_transform, reverse_transform, dataset_transform = get_transforms(image_size)
preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)


to_pil = T.ToPILImage()
# to_pil(ds_sample['pixel_values']).save('sample.png')

# create dataloader
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)


results_folder = Path("./results")
shutil.rmtree("./results")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 200


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = UNet2DModel(
    sample_size=image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

epochs = 30

losses = []

for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Get the model prediction
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        losses.append(loss.item())

        # Update the model parameters with the optimizer
        optimizer.step()
        optimizer.zero_grad()

    loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
    print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

    # save generated images
    # Random starting point (8 random images):
    s = torch.randn(8, 3, 32, 32).to(device)

    for i, t in enumerate(noise_scheduler.timesteps):
        # Get model pred
        with torch.no_grad():
            residual = model(s, t).sample

        # Update sample with step
        s = noise_scheduler.step(residual, t, s).prev_sample

    img = sample_to_pil(s)
    img.save(f"results/epoch_{epoch}.png")
