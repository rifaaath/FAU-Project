import numpy as np
import pickle
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
from tqdm import tqdm
import os

with open(r"hog_features.pkl","rb") as f:
    data=pickle.load(f)
features=data["features"]
labels=data["labels"] 
sample_ids=data["sample_ids"]  

unique_samples = np.unique(sample_ids)
n_train = int(0.7 * len(unique_samples))
train_ids=unique_samples[:n_train]
test_ids=unique_samples[n_train:]
mask_tr=np.isin(sample_ids, train_ids)
mask_te=np.isin(sample_ids, test_ids)
X_train,y_train,sid_train = features[mask_tr], labels[mask_tr], sample_ids[mask_tr]
X_test,y_test,sid_test = features[mask_te], labels[mask_te], sample_ids[mask_te]

# pca
pca = PCA(n_components=0.95, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# miu_c on train set
class_centers = {}
for c in np.unique(y_train):
    class_centers[c] =X_train_pca[y_train == c].mean(axis=0)

# VLAD
def compute_vlad(features, labels, sids, class_centers):
    sample_feats = defaultdict(list)
    sample_lbls = defaultdict(list)

    for f,l, sid in zip(features, labels, sids):
        sample_feats[sid].append(f)
        sample_lbls[sid].append(l)

    encs,order = [],[]
    for sid, flist in tqdm(sample_feats.items(), desc="Computing VLAD"):
        by_cls = defaultdict(list)
        for f, l in zip(flist, sample_lbls[sid]):
            by_cls[l].append(f)

        parts=[]
        for c in sorted(class_centers):
            mu = class_centers[c]
            if c in by_cls:
                # diffs = [mu - f for f in by_cls[c]]
                # v = np.sum(diffs,axis=0)
                diffs = [mu - f for f in by_cls[c]]
                # v = np.mean(diffs, axis=0)
                v = np.sum(diffs, axis=0)
            else:
                v = np.zeros_like(mu)
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v /= norm
            parts.append(v)

        vlad=np.concatenate(parts)
        # power normalize
        vlad=np.sign(vlad)*np.sqrt(np.abs(vlad))
        # L2
        vlad/=np.linalg.norm(vlad)
        encs.append(vlad)
        order.append(sid)

    return np.vstack(encs), np.array(order)

enc_train,order_train=compute_vlad(X_train_pca,y_train,sid_train, class_centers)
enc_test,order_test=compute_vlad(X_test_pca,y_test,sid_test,class_centers)

def majority_label(sids, lbls):
    d = defaultdict(list)
    for sid, l in zip(sids, lbls):
        d[sid].append(l)
    return np.array([max(v, key=v.count) for v in d.values()])

labels_test = majority_label(sid_test, y_test)
cnt= Counter(labels_test)
rare={w for w, c in cnt.items() if c <= 2}
keep=[i for i, l in enumerate(labels_test) if l not in rare]
enc_test_f = enc_test[keep]
labels_f = labels_test[keep]

def distances(encs):
    D = 1.0-encs.dot(encs.T)
    np.fill_diagonal(D, np.finfo(D.dtype).max)
    return D

def evaluate(encs, labels):
    D = distances(encs)
    idx = D.argsort()
    y = LabelEncoder().fit_transform(labels)
    n, corr = len(encs), 0
    APs, APrs = [], []
    for i in range(n):
        rel= 0
        precs=[]
        pr=[]
        R =np.count_nonzero(y==y[i])-1
        for k, g in enumerate(idx[i]):
            if y[g]==y[i]:
                rel+=1
                p=rel/(k+1)
                precs.append(p)
                if k==0:
                    corr+=1
                if rel<=R:
                    pr.append(p)
        APs.append(np.mean(precs))
        APrs.append(np.sum(pr)/R if R>0 else 0)
    return corr/n,np.mean(APs),np.mean(APrs)

top1, mAP, mAPr = evaluate(enc_test_f, labels_f)
print(f"\nTop-1: {top1:.4f}, mAP: {mAP:.4f}, mAP@R: {mAPr:.4f}")
