import torch
import os
from tqdm import tqdm
import logging
import numpy as np
import json
import torch.nn.functional as F

def score_nights_dataset(model, test_loader, device):
    logging.info("Evaluating NIGHTS dataset.")
    d0s = []
    d1s = []
    targets = []
    with torch.no_grad():
        for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img_ref, img_left, img_right, target = img_ref.to(device), img_left.to(device), \
                img_right.to(device), target.to(device)

            dist_0 = model(img_ref, img_left)
            dist_1 = model(img_ref, img_right)

            if len(dist_0.shape) < 1:
                dist_0 = dist_0.unsqueeze(0)
                dist_1 = dist_1.unsqueeze(0)
            dist_0 = dist_0.unsqueeze(1)
            dist_1 = dist_1.unsqueeze(1)
            target = target.unsqueeze(1)

            d0s.append(dist_0)
            d1s.append(dist_1)
            targets.append(target)

    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores, dim=0)
    print(f"2AFC score: {str(twoafc_score)}")
    return twoafc_score

def score_things_dataset(model, test_loader, device):
    logging.info("Evaluating Things dataset.")
    count = 0
    with torch.no_grad():
        for i, (img_1, img_2, img_3) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img_1, img_2, img_3 = img_1.to(device), img_2.to(device), img_3.to(device)

            dist_1_2 = model(img_1, img_2)
            dist_1_3 = model(img_1, img_3)
            dist_2_3 = model(img_2, img_3)

            le_1_3 = torch.le(dist_1_2, dist_1_3)
            le_2_3 = torch.le(dist_1_2, dist_2_3)

            count += sum(torch.logical_and(le_1_3, le_2_3))
    count = count.detach().cpu().numpy()
    accs = count / len(full_dataset)
    print(f"Things accs: {str(accs)}")
    return accs

def score_bapps_dataset(model, test_loader, device):
    logging.info("Evaluating BAPPS dataset.")

    d0s = []
    d1s = []
    ps = []
    with torch.no_grad():
        for i, (im_ref, im_left, im_right, p) in tqdm(enumerate(test_loader), total=len(test_loader)):
            im_ref, im_left, im_right, p = im_ref.to(device), im_left.to(device), im_right.to(device), p.to(device)
            d0 = model(im_ref, im_left)
            d1 = model(im_ref, im_right)
            d0s.append(d0)
            d1s.append(d1)
            ps.append(p.squeeze())
    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    ps = torch.cat(ps, dim=0)
    scores = (d0s < d1s) * (1.0 - ps) + (d1s < d0s) * ps + (d1s == d0s) * 0.5
    final_score = torch.mean(scores, dim=0)
    print(f"BAPPS score: {str(final_score)}")
    return final_score

def score_df2_dataset(model, train_loader, test_loader, gt_path, device):

    def extract_feats(model, dataloader):
        embeds = []
        paths = []
        for im, path in tqdm(dataloader):
            im = im.to(device)
            paths.append(path)
            with torch.no_grad():
                out = model.embed(im).squeeze()
                embeds.append(out.to("cpu"))
        embeds = torch.vstack(embeds).numpy()
        paths = np.concatenate(paths)
        return embeds, paths

    train_embeds, train_paths = extract_feats(model, train_loader)
    train_embeds = torch.from_numpy(train_embeds).to('cuda')
    test_embeds, test_paths = extract_feats(model, test_loader)
    test_embeds = torch.from_numpy(test_embeds).to('cuda')

    with open(gt_path, "r") as f:
        gt = json.load(f)

    ks = [1, 3, 5]
    all_results = {}

    relevant = {k: 0 for k in ks}
    retrieved = {k: 0 for k in ks}
    recall = {k: 0 for k in ks}
    
    for i in tqdm(range(test_embeds.shape[0]), total=test_embeds.shape[0]):
        sim = F.cosine_similarity(test_embeds[i, :], train_embeds, dim=-1)
        ranks = torch.argsort(-sim).cpu()

        query_path = test_paths[i]
        total_relevant = len(gt[query_path])
        gt_retrievals = gt[query_path]
        for k in ks:
            if k > 1:
                k_retrieved = int(len([x for x in train_paths[ranks.cpu()[:k]] if x in gt_retrievals]) >0)
            else:
                k_retrieved = int(train_paths[ranks.cpu()[:k]] in gt_retrievals)
    
            relevant[k] += total_relevant
            retrieved[k] += k_retrieved
    
    for k in ks:
        recall[k] = retrieved[k] / test_embeds.shape[0]

    print(f"DF2 recall@k: {str(recall)}")
    return recall



