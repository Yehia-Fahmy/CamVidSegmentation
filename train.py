import os
import tarfile
import urllib.request
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def download_and_extract_camvid(dest_dir="data"):
    url = "https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz"
    archive_path = os.path.join(dest_dir, "camvid.tgz")
    extract_path = os.path.join(dest_dir, "CamVid")

    os.makedirs(dest_dir, exist_ok=True)

    # Download
    if not os.path.exists(archive_path):
        print("Downloading CamVid dataset...")
        urllib.request.urlretrieve(url, archive_path)
        print("Download complete.")
    else:
        print("Archive already downloaded.")

    # Extract
    if not os.path.exists(extract_path):
        print("Extracting CamVid dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

class CamVidDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

if __name__ == "__main__":
    # Download and extract CamVid if needed
    download_and_extract_camvid("data")

    # Update these paths as needed
    train_images_dir = 'data/CamVid/train'
    train_masks_dir = 'data/CamVid/train_labels'

    # Create dataset and dataloader
    train_dataset = CamVidDataset(train_images_dir, train_masks_dir)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    print(f"Number of training samples: {len(train_dataset)}")
    # Print a sample batch shape
    for images, masks in train_loader:
        print(f"Batch of images: {type(images)}, Batch of masks: {type(masks)}")
        break 