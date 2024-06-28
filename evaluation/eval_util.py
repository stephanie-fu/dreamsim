from torchvision import transforms
import glob
import os
from scripts.util import rescale

IMAGE_EXTENSIONS = ["jpg", "png", "JPEG", "jpeg"]

norms = {
    "dino": transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "mae": transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "open_clip": transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "synclr": transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "resnet": transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

dreamsim_transform = transforms.Compose([
        transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

dino_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            norms['dino'],
        ])

dinov2_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            norms['dino'],
        ])

mae_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            norms['mae'],
        ])

simclrv2_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
        ])

synclr_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            norms['synclr'],
        ])

clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    lambda x: x.convert('RGB'),
    transforms.ToTensor(),
    norms['clip'],
])

# https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
resnet_transform = transforms.Compose([
    transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    lambda x: x.convert('RGB'),
    transforms.ToTensor(),
    rescale,
    norms['resnet'],
])

open_clip_transform = clip_transform

vanilla_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

def get_val_transform(model_type):
    if "dino" in model_type:
        return dino_transform
    elif "mae" in model_type:
        return mae_transform
    elif "clip" in model_type:
        return clip_transform
    elif "open_clip" in model_type:
        return open_clip_transform
    else:
        return vanilla_transform


def get_train_transform(model_type):
    if "mae" in model_type:
        norm = norms["mae"]
    elif "clip" in model_type:
        norm = norms["clip"]
    elif "open_clip" in model_type:
        norm = norms["open_clip"]
    else:
        norm = norms["dino"]

    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        norm,
    ])


def get_paths(path):
    all_paths = []
    for ext in IMAGE_EXTENSIONS:
        all_paths += glob.glob(os.path.join(path, f"**.{ext}"))
    return all_paths
