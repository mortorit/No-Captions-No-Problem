import torch
from torch.utils.data import Dataset
import json
import trimesh
import os
from PIL import Image, ImageChops

class Pix3DDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None, background_option=False):
        self.transform = transform
        self.background_option = background_option
        self.root_dir = root_dir

        # Load JSON file
        with open(json_path, 'r') as file:
            self.data = json.load(file)

        # Preprocess data: load models and sample point clouds
        self.models = []
        for item in self.data:
            model_path = os.path.join(self.root_dir, item['model'])
            try:
                loaded = trimesh.load(str(model_path), force='mesh')
                # If the loaded object is a scene, convert it to a mesh
                if isinstance(loaded, trimesh.Scene):
                    mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
                else:
                    mesh = loaded

                point_cloud = mesh.sample(2048)
                self.models.append(point_cloud)
            except Exception as e:
                print(f"Error loading model at {model_path}: {e}")
                self.models.append(None)

        # Store image paths and categories
        self.img_paths = [item['img'] for item in self.data]
        self.categories = [item['category'] for item in self.data]
        self.masks = [item['mask'] for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.img_paths[idx])
        img = Image.open(img_path).convert('RGBA')

        if self.background_option:
            # Apply mask to remove background
            mask_path = os.path.join(self.root_dir, self.masks[idx])
            mask = Image.open(mask_path).convert('RGBA')
            img = ImageChops.multiply(img, mask).convert('RGB')


        # Get point cloud and category
        point_cloud = self.models[idx]
        category = self.categories[idx]

        if self.transform:
            img_emb = self.transform(img)

        return img_emb.squeeze(), self.normalize_pc(point_cloud), category

    def normalize_pc(self, point_cloud):
        point_cloud = torch.Tensor(point_cloud)
        centroid = point_cloud.mean(dim=0)
        point_cloud -= centroid
        furthest_distance = point_cloud.norm(p=2, dim=1).max()
        point_cloud /= furthest_distance
        return point_cloud



# Example of how to use the dataset
if __name__ == "__main__":
    dataset = Pix3DDataset(json_path='path/to/json/file', root_dir='path/to/root/directory', background_option=True)
