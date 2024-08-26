import argparse
import numpy as np
import open_clip
import random
import torch
from torch.utils.data import DataLoader

from data.pix3d import Pix3DDataset
from models.D3Clip import D3CLIP

#set all seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip = clip.cuda()

def process_image(image):
    with torch.no_grad(), torch.cuda.amp.autocast():
        image = preprocess(image).unsqueeze(0)
    return image

def evaluate_retrieval_with_categories(model, dataset, k=5, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize counters for overall accuracy
    overall_image_to_shape_top1_correct = 0
    overall_image_to_shape_topk_correct = 0
    overall_shape_to_image_top1_correct = 0
    overall_shape_to_image_topk_correct = 0
    total = len(dataset)
    categories = dataset.categories
    # Initialize dictionaries to track per-category accuracy
    category_correct_top1 = {category: 0 for category in set(categories)}
    category_correct_topk = {category: 0 for category in set(categories)}
    category_correct_shape_to_image_top1 = {category: 0 for category in set(categories)}
    category_correct_shape_to_image_topk = {category: 0 for category in set(categories)}
    category_count = {category: 0 for category in set(categories)}

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.eval()
    with torch.no_grad():
        # Process each image for image-to-shape and shape-to-image retrieval
        for data in dataloader:
            im, pc, cat = data
            pc = pc.to(device)
            im_emb = clip.encode_image(im.to(device))
            pc_emb = model.encode_point_cloud(pc)
            im_emb = im_emb.squeeze()
            similarity = model.cosine_similarity(im_emb, pc_emb)
            for row in range(similarity.size(0)):
                sorted_indices = similarity[row].argsort(descending=True)
                correct_index = row
                category = cat[row]

                # Update overall and per-category counters
                if correct_index == sorted_indices[0]:
                    overall_image_to_shape_top1_correct += 1
                    category_correct_top1[category] += 1
                if correct_index in sorted_indices[:k]:
                    overall_image_to_shape_topk_correct += 1
                    category_correct_topk[category] += 1

                category_count[category] += 1

            # Calculate shape-to-image retrieval
            for col in range(similarity.size(1)):
                sorted_indices = similarity[:, col].argsort(descending=True)
                correct_index = col
                category = cat[col]

                # Update overall and per-category counters
                if correct_index == sorted_indices[0]:
                    overall_shape_to_image_top1_correct += 1
                    category_correct_shape_to_image_top1[category] += 1
                if correct_index in sorted_indices[:k]:
                    overall_shape_to_image_topk_correct += 1
                    category_correct_shape_to_image_topk[category] += 1

    # Calculate overall accuracies
    overall_top1_accuracy = overall_image_to_shape_top1_correct / total
    overall_topk_accuracy = overall_image_to_shape_topk_correct / total
    overall_shape_to_image_top1_accuracy = overall_shape_to_image_top1_correct / total
    overall_shape_to_image_topk_accuracy = overall_shape_to_image_topk_correct / total

    # Calculate per-category accuracies
    category_top1_accuracy = {category: (correct / category_count[category]) for category, correct in
                              category_correct_top1.items()}

    category_topk_accuracy = {category: (correct / category_count[category]) for category, correct in
                              category_correct_topk.items()}

    category_shape_to_image_top1_accuracy = {category: (correct / category_count[category]) for category, correct in
                                             category_correct_shape_to_image_top1.items()}

    category_shape_to_image_topk_accuracy = {category: (correct / category_count[category]) for category, correct in
                                             category_correct_shape_to_image_topk.items()}

    return (overall_top1_accuracy, overall_topk_accuracy,
            overall_shape_to_image_top1_accuracy, overall_shape_to_image_topk_accuracy,
            category_top1_accuracy, category_topk_accuracy,
            category_shape_to_image_top1_accuracy, category_shape_to_image_topk_accuracy)


def main(model_path, json_path, root_dir, remove_background):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Pix3DDataset(json_path=json_path, root_dir=root_dir, background_option=remove_background,
                           transform=process_image)
    model = D3CLIP(device=device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    (overall_top1, overall_topk, overall_shape_to_image_top1, overall_shape_to_image_topk, cat_top1, cat_topk,
     cat_shape_to_image_top1, cat_shape_to_image_topk) = evaluate_retrieval_with_categories(model, dataset)
    print(f"Overall Image-to-Shape Top-1 Accuracy: {overall_top1 * 100:.2f}%")
    print(f"Overall Image-to-Shape Top-k Accuracy: {overall_topk * 100:.2f}%")
    print("Per-Category Image-to-Shape Top-1 Accuracy:")
    for category, accuracy in cat_top1.items():
        print(f"{category}: {accuracy * 100:.2f}%")
    print("Per-Category Image-to-Shape top-k Accuracy:")
    for category, accuracy in cat_topk.items():
        print(f"{category}: {accuracy * 100:.2f}%")
    print('-'*50)
    print(f"Overall Shape-to-Image Top-1 Accuracy: {overall_shape_to_image_top1 * 100:.2f}%")
    print(f"Overall Shape-to-Image top-k Accuracy: {overall_shape_to_image_topk * 100:.2f}%")
    print("Per-Category Shape-to-Image Top-1 Accuracy:")
    for category, accuracy in cat_shape_to_image_top1.items():
        print(f"{category}: {accuracy * 100:.2f}%")
    print("Per-Category Shape-to-Image top-k Accuracy:")
    for category, accuracy in cat_shape_to_image_topk.items():
        print(f"{category}: {accuracy * 100:.2f}%")
    return (overall_top1, overall_topk, overall_shape_to_image_top1, overall_shape_to_image_topk, cat_top1, cat_topk,
     cat_shape_to_image_top1, cat_shape_to_image_topk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the Pix3D JSON file")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the Pix3D root directory")
    parser.add_argument("--remove_background", default=True, help="Whether to remove background from images")
    args = parser.parse_args()
    main(args.model_path, args.json_path, args.root_dir, args.background_option)
