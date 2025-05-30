import torch
import torch.nn as nn
import open_clip
import os
from basicsr.utils.registry import ARCH_REGISTRY
from PIL import Image

@ARCH_REGISTRY.register()
class DACLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess  = open_clip.create_model_from_pretrained(
            'daclip_ViT-L-14', pretrained="/swim-pool/yicheng/NYCU_VRDP_FinalProject/weights/wild-daclip_ViT-L-14.pt"
        )

    def encode(self, lq_path, device):
        batch_images = []
        for img_path in lq_path:
            image = Image.open(img_path)
            preprocess_image = self.preprocess(image)
            batch_images.append(preprocess_image)
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            image_features, degra_features = self.model.encode_image(batch_tensor, control=True)
        return image_features, degra_features

    def forward(self, x, device):
        return self.encode(x, device)
