import numpy as np
import torch
from tqdm import trange

from data.modelnet40 import ModelNet
from models.D3Clip import D3CLIP


modelnet_categories = [
    'airplane',
    'bathtub',
    'bed',
    'bench',
    'bookshelf',
    'bottle',
    'bowl',
    'car',
    'chair',
    'cone',
    'cup',
    'curtain',
    'desk',
    'door',
    'dresser',
    'flower pot',
    'glass box',
    'guitar',
    'keyboard',
    'lamp',
    'laptop',
    'mantel',
    'monitor',
    'night stand',
    'person',
    'piano',
    'plant',
    'radio',
    'range hood',
    'sink',
    'sofa',
    'stairs',
    'stool',
    'table',
    'tent',
    'toilet',
    'tv stand',
    'vase',
    'wardrobe',
    'xbox'
]


texts = []

for cat in modelnet_categories:
    texts.append('a point cloud model of a ' + cat)

def main(data_root, model_path, cuda):
    acc_1 = 0
    acc_5 = 0
    total = 0
    with torch.no_grad():
        device = 'cuda' if cuda else 'cpu'
        model = D3CLIP(device)
        model.load_state_dict(torch.load(model_path))
        model.load_model()
        model = model.to(device)
        model.eval()
        dataset = ModelNet(data_root, 'test')
        text_embs = model.encode_text(texts)
        text_embs = text_embs / text_embs.norm(p=2, dim=1, keepdim=True)
        for i in trange(len(dataset)):
            data = dataset[i]
            pc, cat_idx, cat = data
            pc = pc.to(device).unsqueeze(0)
            pc_embs = model.encode_point_cloud(pc).squeeze()
            pc_embs = pc_embs / pc_embs.norm(p=2, dim=0, keepdim=True)
            similarity = text_embs @ pc_embs.T
            similarity = similarity * model.temp + model.b
            
            closest = torch.argsort(similarity, descending=True)[:5]

            if cat_idx in closest[0]:
                acc_1 += 1
            if cat_idx in closest[:5]:
                acc_5 += 1
            total += 1
        print('Accuracy@1:', acc_1/total, 'Accuracy@5:', acc_5/total)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--cuda', default=True)
    args = parser.parse_args()
    main(**vars(args))
