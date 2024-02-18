from PIL import Image
from lightning_fabric import seed_everything
from torch.utils.data import DataLoader

from dataset.dataset import TwoAFCDataset
from dreamsim import dreamsim
from torchvision import transforms
import torch
import os
from tqdm import tqdm

from util.train_utils import seed_worker
from util.utils import get_preprocess
import numpy as np
import pickle as pkl
from sklearn.decomposition import PCA

seed = 1234
dataset_root = './dataset/nights'
model_type = 'dino_vitb16,clip_vitb16,open_clip_vitb16'
num_workers = 8
batch_size = 32

seed_everything(seed)
g = torch.Generator()
g.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = dreamsim(pretrained=True, device=device, cache_dir='./models_new')

train_dataset = TwoAFCDataset(root_dir=dataset_root, split="train", preprocess=get_preprocess(model_type))
val_dataset = TwoAFCDataset(root_dir=dataset_root, split="val", preprocess=get_preprocess(model_type))
test_dataset = TwoAFCDataset(root_dir=dataset_root, split="test", preprocess=get_preprocess(model_type))
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                          worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

data = {'train': {}, 'val': {}, 'test': {}}
all_embeds = []

with torch.no_grad():
    for split, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        for img_ref, img_left, img_right, p, id in tqdm(loader):
            img_ref = img_ref.to(device)
            img_left = img_left.to(device)
            img_right = img_right.to(device)

            embed_ref, embed_0, d0 = model(img_ref, img_left)
            _, embed_1, d1 = model(img_ref, img_right)
            #
            # if split == 'train':
            #     all_embeds.append(embed_ref)
            #     all_embeds.append(embed_0)
            #     all_embeds.append(embed_1)

            for i in range(len(id)):
                curr_id = id[i].item()
                data[split][curr_id] = [d0[i].item(), d1[i].item()]

# all_embeds = torch.cat(all_embeds).cpu()
# principal = PCA(n_components=512)
# principal.fit(all_embeds)

with open('precomputed_sims.pkl', 'wb') as f:
    pkl.dump(data, f)

# with open('precomputed_embeds.pkl', 'wb') as f:
#     pkl.dump(principal, f)