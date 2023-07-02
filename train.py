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

from PIL import Image

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# load dataset from the hub
dataset = load_dataset("nielsr/CelebA-faces", split="train")
image_size = 128
num_channels = 3
batch_size = 32


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
        transforms.CenterCrop(size=image_size),
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)


def transform(examples):
    mode = "RGB" if num_channels == 3 else "L"
    images = [preprocess(image.convert(mode)) for image in examples["image"]]
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

model = Unet(
    dim=image_size,
    channels=num_channels,
    dim_mults=(
        1,
        2,
        4,
    ),
)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=100, beta_schedule="squaredcos_cap_v2"
)

epochs = 600

losses = []

sample_every = 50

for epoch in range(epochs):
    num_batches = len(train_dataloader)
    for i, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        # for i in range(len(noisy_images)):
        #     to_pil(noisy_images[i]).save(f'results/test_{i}_ts_{timesteps[i]}_noisy.png')
        #     to_pil(clean_images[i]).save(f'results/test_{i}_ts_{timesteps[i]}_clean.png')

        # Get the model prediction
        noise_pred = model(noisy_images, timesteps)

        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        losses.append(loss.item())
        print(f"loss for batch {i} / {num_batches}: {loss.item()}")

        # Update the model parameters with the optimizer
        optimizer.step()
        optimizer.zero_grad()

        if i % sample_every == 0:
            print("Sampling...")
            gen_batch_size = 8
            # save generated images
            # Random starting point (8 random images):
            s = torch.randn(gen_batch_size, num_channels, 32, 32).to(device)
            for j, t in enumerate(noise_scheduler.timesteps):
                # Get model pred
                with torch.no_grad():
                    t = t.repeat(gen_batch_size).to(device)
                    residual = model(s, t)

                # Update sample with step
                t = t[0].item()
                s = noise_scheduler.step(residual, t, s).prev_sample

            img = sample_to_pil(s)
            img.save(f"results/epoch_{epoch}_batch_{i}.png")

    loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
    print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
