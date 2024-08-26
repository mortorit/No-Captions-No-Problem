import os

import torch
from torch.utils.data import Dataset
import numpy as np


embs_names = ['000', '180','012', '192', '024', '204', '036', '216', '048', '228', '060', '240',
              '072', '252', '084', '264', '096', '276', '108', '288', '120', '300', '132', '312',
              '144', '324', '156', '336', '168', '348']

class ShapenetPC(Dataset):
    def __init__(self, point_cloud_root, embeddings_root, labels_root, split='train', val_split=0.2, subsample=2048,
                 annotation_type='image_sim', remove_extra_cat= False):
        """
        The ShapenetPC dataset class. Each root directory should contain subdirectories for each category,
        which in turn contain subdirectories for each sample.
        :param point_cloud_root:
        :param embeddings_root:
        :param labels_root:
        :param split:
        :param val_split:
        """
        self.point_cloud_root = point_cloud_root
        self.embeddings_root = embeddings_root
        self.labels_root = labels_root
        self.split = split
        self.val_split = val_split
        self.subsample = subsample
        self.annotation_type = annotation_type

        # Get list of categories
        self.categories = [d for d in sorted(os.listdir(self.point_cloud_root)) if
                           os.path.isdir(os.path.join(self.point_cloud_root, d))]

        self.remove_extra_cat = remove_extra_cat

        #remove annotations for category 02992529, which is a duplicate in ULIP 2 dataset
        if self.annotation_type == 'text_sims' or self.remove_extra_cat:
            self.categories.remove('02992529')

        # Split the categories into training and validation sets, using the val_split parameter
        # to determine the proportion of samples in each category to use for validation
        self.train_files = []
        self.val_files = []
        self.train_point_clouds = []
        self.val_point_clouds = []
        self.cat_start_idxs_train = {}
        self.cat_start_idxs_val = {}
        self.cat_train_len = {}
        self.samples_categories_train = []
        self.samples_categories_val = []
        for i, category in enumerate(self.categories):
            self.cat_start_idxs_train[i] = len(self.train_files)

            samples = [d for d in sorted(os.listdir(os.path.join(self.point_cloud_root, category))) if
                       os.path.isdir(os.path.join(self.point_cloud_root, category, d))]
            split_idx = int(len(samples) * (1 - self.val_split))
            self.train_files.extend(s for s in samples[:split_idx])

            self.cat_start_idxs_val[i] = len(self.val_files)
            self.cat_train_len[i] = len(samples[:split_idx])

            self.val_files.extend([s for s in samples[split_idx:]])
            self.samples_categories_train.extend([i] * len(samples[:split_idx]))
            self.samples_categories_val.extend([i] * len(samples[split_idx:]))

        self.cats_similarities = {}
        if labels_root:
            for i, category in enumerate(self.categories):
                self.cats_similarities[i] = np.load(os.path.join(self.labels_root, category + ('_similarity_matrix.npy' if annotation_type=='image_sim' else '_distance_matrix.npy')))

    def __len__(self):
        if self.split == 'train':
            return len(self.train_files)
        elif self.split == 'val':
            return len(self.val_files)
        else:
            raise ValueError("Invalid split parameter. Must be either 'train' or 'val'.")

    def __getitem__(self, idx):
        if self.split == 'train':
            cat = self.categories[self.samples_categories_train[idx]]
            files = self.train_files
        elif self.split == 'val':
            cat = self.categories[self.samples_categories_val[idx]]
            files = self.val_files
        else:
            raise ValueError("Invalid split parameter. Must be either 'train' or 'val'.")

        pc_path = os.path.join(self.point_cloud_root, cat, files[idx], '20k_sample.pth')
        pc = torch.load(pc_path)
        pc = self.subsample_pc(pc, self.subsample)
        pc = self.augment_pc(pc)

        emb_name = embs_names[np.random.randint(0, len(embs_names))]
        emb_path = os.path.join(self.embeddings_root, cat, files[idx], f"{cat}-{files[idx]}_r_{emb_name}.pth")
        embedding = torch.load(emb_path)

        return (pc.float(), embedding.float(), self.samples_categories_train[idx]
        if self.split == 'train' else self.samples_categories_val[idx], idx)


    def get_couple_labels(self, idx_1, idx_2, category, split, cat_similarity_matrix):
        start_idx = self.cat_start_idxs_train[category.item()] if split == 'train' else self.cat_start_idxs_val[category.item()]
        sample_1_idx = idx_1 - start_idx + self.cat_train_len[category.item()] if split == 'val' else idx_1 - start_idx
        sample_2_idx = idx_2 - start_idx + self.cat_train_len[category.item()] if split == 'val' \
            else idx_2 - start_idx
        label = cat_similarity_matrix[sample_1_idx, sample_2_idx]
        return label

    def create_category_similarity_matrix(self, cats):
        # Create a 2D grid of category comparisons
        similarity_matrix = cats.unsqueeze(1) == cats.unsqueeze(0)

        # Extract the upper triangular part of the matrix
        triangular_similarity_matrix = torch.triu(similarity_matrix, diagonal=1)

        return triangular_similarity_matrix

    def get_labels(self, cats, idxs, split='train', negative_label=0, add_diagonal=0):
        labels = torch.zeros(len(idxs), len(idxs))
        are_same_category = self.create_category_similarity_matrix(cats)

        # Iterate over these elements
        for i in range(len(idxs)):
            cat_sim_matrix = self.cats_similarities[cats[i].item()]
            for j in range(i, len(idxs)):
                if j == i:
                    if self.annotation_type == 'image_sim':
                        labels[i,j] = 1 + add_diagonal
                    else:
                        labels[i, j] = 1
                    continue
                if are_same_category[i][j]:
                    label = self.get_couple_labels(idxs[i], idxs[j], cats[i], split, cat_sim_matrix)
                    if self.annotation_type == 'image_sim':
                        labels[i, j] = labels[j, i] = label
                    else:
                        labels[i, j] = labels[j, i] = 1 / (1 + label)
                else:
                    labels[i, j] = labels[j, i] = negative_label
        return labels

    def get_weights(self, cats, idxs, split='train', alpha=1, negative_add = 0.15):
        labels = torch.zeros(len(idxs), len(idxs))
        are_same_category = self.create_category_similarity_matrix(cats)
        # Iterate over these elements
        for i in range(len(idxs)):
            cat_sim_matrix = self.cats_similarities[cats[i].item()]
            for j in range(i, len(idxs)):
                if j == i:
                    if self.annotation_type == 'image_sim':
                        labels[i,j] = alpha
                    else:
                        labels[i, j] = alpha
                    continue
                if are_same_category[i][j]:
                    label = self.get_couple_labels(idxs[i], idxs[j], cats[i], split, cat_sim_matrix)
                    if self.annotation_type == 'image_sim':
                        labels[i, j] = labels[j, i] = (label + 1) / 2
                    else:
                        labels[i, j] = labels[j, i] = 1 / (1 + label)
                else:
                    labels[i, j] = labels[j, i] = 0 + negative_add

        #convert to weights
        #define a matrix with 0 in the diagonal and 1 in the other cells
        ones = torch.ones_like(labels)
        mask = ones - torch.eye(labels.shape[0])
        #calculate the mean for each row, with the mask
        mean = (labels * mask).mean(dim=1) #similarity mean for each sample without considering itself
        #calculate the weights for each cell as their value divided by the mean of their row
        weights = labels / mean.unsqueeze(1)
        #reset the diagonal to 1
        weights = weights * mask + torch.eye(labels.shape[0])

        return weights

    def random_rotate_pc(self, point_cloud, max_alpha=45, max_beta=45, max_gamma=45):
        """Rotates the given point cloud by the given angles around the x, y, and z axes."""
        # Generate random angles
        alpha = np.random.uniform(-max_alpha, max_alpha)
        beta = np.random.uniform(-max_beta, max_beta)
        gamma = np.random.uniform(-max_gamma, max_gamma)

        # Convert angles from degrees to radians
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)

        # Ensure the rotation matrices are of type Float
        R_x = torch.tensor([[1, 0, 0],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha), np.cos(alpha)]], dtype=torch.float32)

        R_y = torch.tensor([[np.cos(beta), 0, np.sin(beta)],
                            [0, 1, 0],
                            [-np.sin(beta), 0, np.cos(beta)]], dtype=torch.float32)

        R_z = torch.tensor([[np.cos(gamma), -np.sin(gamma), 0],
                            [np.sin(gamma), np.cos(gamma), 0],
                            [0, 0, 1]], dtype=torch.float32)

        # Combined rotation matrix
        R = torch.matmul(R_z, torch.matmul(R_y, R_x))

        # Apply rotation to each point in the point cloud
        # Ensure the point cloud tensor is also Float if not already
        rotated_pc = torch.matmul(point_cloud.float(), R.T)

        return rotated_pc

    def subsample_pc(self, point_cloud, num_samples):
        indices = np.random.choice(point_cloud.shape[0], num_samples, replace=False)
        return point_cloud[indices]

    def normalize_pc(self, point_cloud):
        centroid = point_cloud.mean(dim=0)
        point_cloud -= centroid
        furthest_distance = point_cloud.norm(p=2, dim=1).max()
        point_cloud /= furthest_distance
        return point_cloud

    def augment_pc(self, point_cloud):
        point_cloud = self.normalize_pc(point_cloud)
        return self.random_rotate_pc(point_cloud)

    def get_image_paths(self, cats, idxs, split='train'):
        paths = []
        for i in range(len(idxs)):
            cat = self.categories[cats[i].item()]
            idx = idxs[i]
            if split == 'train':
                paths.append(os.path.join(self.point_cloud_root, cat, self.train_files[idx], f"{cat}-{self.train_files[idx]}_r_000.png"))
            else:
                paths.append(os.path.join(self.point_cloud_root, cat, self.val_files[idx], f"{cat}-{self.val_files[idx]}_r_000.png"))
        return paths
