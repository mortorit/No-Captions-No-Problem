import math
import open_clip
import torch

from models.pointnext import PointNEXT

class D3CLIP(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.clip = None
        self.preprocess = None
        self.tokenizer = None
        self.point_encoder = PointNEXT()
        self.temp = torch.nn.Parameter(torch.Tensor([math.log(10)]))
        self.b = torch.nn.Parameter(torch.Tensor([-10]))
        self.device = device

    def load_model(self):
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip = self.clip.to(self.device)

    def encode_text(self, texts):
        texts = self.tokenizer(texts).to(self.device)
        text_features = self.clip.encode_text(texts)
        return text_features

    def encode_image(self, images):
        # preprocess images
        images = torch.stack([self.preprocess(image) for image in images])
        return self.clip.encode_image(images.to(self.device))

    def encode_point_cloud(self, point_cloud):
        emb = self.point_encoder(point_cloud)
        return self.point_proj(emb) if self.proj else emb

    def cosine_similarity(self, a, b):
        return (a @ b.T) / (a.norm(dim=-1)[:, None] * b.norm(dim=-1))

    def forward(self, point_cloud):
        return self.encode_point_cloud(point_cloud)
