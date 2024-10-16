import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


REPO_ID = "Ryan-sjtu/celebahq-caption"
CACHE_DIR = f"data/{REPO_ID.lower().replace('/', '-')}"
SEED = 24
BUFFER_SIZE = 100
BATCH_SIZE = 12
IMAGE_SIZE = 256    
NUM_WORKERS = 2


class CustomResizeAndCrop:
    def __init__(self, target_size=IMAGE_SIZE):
        self.target_size = target_size

    def __call__(self, image):
        width, height = image.size
        if width > height:
            scale = self.target_size / width
        else:
            scale = self.target_size / height

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized_image = transforms.Resize((new_height, new_width))(image)
        final_image = transforms.CenterCrop(self.target_size)(resized_image)

        return final_image

transform_list = transforms.Compose([
    CustomResizeAndCrop(target_size=IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])


os.makedirs(CACHE_DIR, exist_ok=True)


def preprocess(samples, indices):
    # Convert text to lowercase
    samples['text'] = [i.lower() for i in samples['text']]
    samples['index'] = indices
    
    return samples

def collate_fn(batch):
    images = []
    texts = []
    indices = []

    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else "Main process"

    for item in batch:
        image = item['image']
        text = item['text'] 
        index = item['index'] 

        if transform_list:
            image = transform_list(image)

        images.append(image)
        texts.append(text)
        indices.append(index)
    
    print(f"Worker ID: {worker_id}  |  Batch size: {torch.stack(images).shape} | indices: {indices}")

    return images, texts, indices


if __name__ == "__main__":
    ds = load_dataset(REPO_ID, 
                    split='train', 
                    streaming=True, 
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR,)

    shuffled_ds = ds.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)\
                    .map(preprocess, 
                        batch_size=BATCH_SIZE,
                        with_indices=True,
                        batched=True)


    dataloader = DataLoader(
        dataset=shuffled_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,  
    )

    for i, (images, texts, indices) in enumerate(dataloader):
        print(f"Batch {i}, done")
      