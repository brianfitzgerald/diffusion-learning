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


# load dataset from the hub
dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128

torch.manual_seed(0)

inference_transform, reverse_transform, dataset_transform = get_transforms(image_size)


# define function
def dataset_transform_batch(examples):
    examples["pixel_values"] = [
        dataset_transform(image.convert("L")) for image in examples["image"]
    ]
    del examples["image"]

    return examples


transformed_dataset = dataset.with_transform(dataset_transform_batch).remove_columns(
    "label"
)

ds_sample = transformed_dataset["train"][0]
to_pil = T.ToPILImage()
# to_pil(ds_sample['pixel_values']).save('sample.png')

# create dataloader
dataloader = DataLoader(
    transformed_dataset["train"], batch_size=batch_size, shuffle=True
)


results_folder = Path("./results")
shutil.rmtree("./results")
results_folder.mkdir(exist_ok=True)
save_and_sample_every = 200


device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(
        1,
        2,
        4,
    ),
)
model.to(device)

print("using", device)

optimizer = Adam(model.parameters(), lr=1e-3)

scheduler = get_scheduler(200)

epochs = 100

for epoch in range(epochs):
    print(f"Epoch {epoch}:")
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        # set of timesteps that we are sampling
        t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, scheduler, batch, t, loss_type="huber")

        if step % 100 == 0:
            print(f"Loss for step {step}:", loss.item())

        loss.backward()
        optimizer.step()

        # save generated images
    batches = num_to_groups(4, batch_size)
    all_images_list = list(
        map(
            lambda n: sample(model, scheduler, image_size, n, channels),
            batches,
        )
    )
    all_images = torch.cat(all_images_list, dim=0)
    all_images = (all_images + 1) * 0.5
    save_image(
        all_images,
        str(results_folder / f"sample-{epoch}.png"),
        nrow=6,
    )
    print(f"Sampled for step {step}")
