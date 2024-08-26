import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.D3Clip import D3CLIP
from utils.train_utils import adjust_learning_rate, create_datasets, CustomWeightedDenominatorCrossEntropyLoss, \
    get_weights


def main(similarity_name, point_cloud_root, embeddings_root, labels_root, labels_text_root, batch_size, epochs, cuda,
         save_path, wandb_name, wandb_project, wandb_entity):
    if similarity_name not in ['I2I', '(I2L)2', 'AVG']:
        raise ValueError('Invalid similarity type. Choose from I2I, (I2L)2, AVG')

    name = wandb_name
    wandb.login()
    wandb.init(project=wandb_project, entity=wandb_entity, name=name)

    if not os.path.exists(save_path):
        # Create the directory
        os.makedirs(save_path)

    initial_lr = 1e-7  # Starting LR for warmup
    final_lr = 0.001  # Final LR after warmup, also the starting LR for cosine annealing
    warmup_epochs = 30  # Number of epochs for the linear warmup

    device = 'cuda' if cuda else 'cpu'

    model = D3CLIP(device)

    dataset, val_dataset, dataset_texts, val_dataset_texts = create_datasets(similarity_name, point_cloud_root,
                                                                             embeddings_root, labels_root,
                                                                             labels_text_root)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0.01)
    total_steps = len(data_loader)
    model.to(device)

    criterion = CustomWeightedDenominatorCrossEntropyLoss()

    best_loss = float('inf')
    wandb.watch(model, log='all')
    # training and validation loop
    for epoch in range(epochs):
        model.train()
        train_ep_losses = []
        val_ep_losses = []

        with tqdm(data_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch: {epoch}")
                adjust_learning_rate(optimizer, epoch, i, total_steps, warmup_epochs, epochs, initial_lr, final_lr)
                pc, im_emb, cat, idx = data
                pc_emb = model(pc.to(device))
                im_emb = im_emb.squeeze().to(device)

                # compute similarity between all pairs of embeddings
                pc_emb = pc_emb / pc_emb.norm(p=2, dim=1, keepdim=True)
                im_emb = im_emb / im_emb.norm(p=2, dim=1, keepdim=True)
                similarity = im_emb @ pc_emb.T
                similarity = similarity * model.temp + model.b

                # get weights
                weights = get_weights(similarity_name, dataset, dataset_texts, cat, idx, 'train').cuda()

                # define targets on the diagonal
                labels = torch.arange(0, pc_emb.shape[0]).cuda()

                # contrastive loss
                loss = (criterion(similarity, labels, weights) + criterion(similarity.T, labels, weights)) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t_loss = loss.item()
                train_ep_losses.append(t_loss)

                tepoch.set_postfix(training_loss=t_loss)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_data_loader):
                pc, im_emb, cat, idx = data
                pc_emb = model(pc.to(device))
                im_emb = im_emb.to(device).squeeze()
                pc_emb = pc_emb / pc_emb.norm(p=2, dim=1, keepdim=True)
                im_emb = im_emb / im_emb.norm(p=2, dim=1, keepdim=True)
                similarity = im_emb @ pc_emb.T
                similarity = similarity * model.temp + model.b

                # get weights
                weights = get_weights(similarity_name, val_dataset, val_dataset_texts, cat, idx, 'val').cuda()

                # define targets on the diagonal
                labels = torch.arange(0, pc_emb.shape[0]).cuda()
                # contrastive loss
                loss = (criterion(similarity, labels, weights) + criterion(similarity.T, labels, weights)) / 2

                t_loss = loss.item()
                val_ep_losses.append(t_loss)

                if t_loss < best_loss:
                    best_loss = t_loss
                    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

                torch.save(model.state_dict(), os.path.join(save_path, 'last_model.pth'))

        wandb.log({'train_loss': sum(train_ep_losses) / len(train_ep_losses),
                   'val_loss': sum(val_ep_losses) / len(val_ep_losses), 'lr': optimizer.param_groups[0]['lr']})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--similarity_name', type=str, help='Similarity to use for training (I2I, (I2L)2 or AVG)')
    parser.add_argument('--point_cloud_root', type=str, help='Path to the ShapeNet point cloud root folder')
    parser.add_argument('--embeddings_root', type=str, help='Path to the CLIP image embeddings root folder')
    parser.add_argument('--labels_root', type=str, help='Path to the precomputed I2I similarities root folder')
    parser.add_argument('--labels_text_root', type=str, help='Path to the precomputed (I2L^2) similarities root folder')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train for.')
    parser.add_argument('--cuda', default='True', choices=['True', 'False'], help='Whether to use gpu or not.')
    parser.add_argument('--save_path', type=str, default='.', help='Directory to store models and figures.')
    parser.add_argument('--wandb_name', type=str, help='Name for wandb experiment')
    parser.add_argument('--wandb_project', type=str, help='Name for wandb project')
    parser.add_argument('--wandb_entity', type=str, help='Entity for wandb project')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
