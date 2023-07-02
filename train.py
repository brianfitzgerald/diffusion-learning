from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path
import torch
from model.layers import num_to_groups
from train_utils import *
from model.unet import *
from torchvision.utils import save_image
from torch.optim import Adam
import os
import shutil
import torchvision.transforms as T
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# load dataset from the hub
dataset = load_dataset("nielsr/CelebA-faces", split="train")
image_size = 48
num_channels = 3
batch_size = 128
epochs = 600
sample_every = 200
save_every = 1000
clip_value = 1


torch.manual_seed(0)

preprocess = transforms.Compose(
    [
        transforms.CenterCrop(size=128),
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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = Unet(
    dim=image_size,
    channels=num_channels,
    dim_mults=(
        1,
        2,
        4,
        8,
    ),
)

model = model.to(device)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


model.apply(init_weights)


optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=500, beta_schedule="squaredcos_cap_v2"
)


losses = []


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
        #     to_pil(noisy_images[i]).save(
        #         f"results/test_{i}_ts_{timesteps[i]}_noisy.png"
        #     )
        #     to_pil(clean_images[i]).save(
        #         f"results/test_{i}_ts_{timesteps[i]}_clean.png"
        #     )

        # Get the model prediction
        noise_pred = model(noisy_images, timesteps)

        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
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
            s = torch.randn(gen_batch_size, num_channels, image_size, image_size).to(device)
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
