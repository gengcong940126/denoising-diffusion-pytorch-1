# coding=utf-8
# Taken from:
# https://github.com/google/compare_gan/blob/master/compare_gan/src/prd_score.py
#
# Changes:
#   - default dpi changed from 150 to 300
#   - added handling of cases where P = Q, where precision/recall may be
#     just above 1, leading to errors for the f_beta computation
#
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from score.inception_network import InceptionV3
import os
from score.fid_score import calculate_frechet_distance
from score.inception_score import kl_scores
import matplotlib.pyplot as plt
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel



class precision_recall(object):
    def __init__(self,inception_model, device):
        self.inception_model = inception_model
        self.device = device
        self.disable_tqdm = device.index != 0


    def generate_images(self, gen, d, batch_size):
        zs=torch.randn((batch_size, d)).cuda()
        batch_images = gen(zs)

        return batch_images


    def inception_softmax(self, batch_images):
        with torch.no_grad():
            embeddings, logits = self.inception_model(batch_images)
        return embeddings


    def cluster_into_bins(self, real_embeds, fake_embeds, num_clusters):
        representations = np.vstack([real_embeds, fake_embeds])
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, n_init=10)
        labels = kmeans.fit(representations).labels_

        real_labels = labels[:len(real_embeds)]
        fake_labels = labels[len(real_embeds):]

        real_density = np.histogram(real_labels, bins=num_clusters, range=[0, num_clusters], density=True)[0]
        fake_density = np.histogram(fake_labels, bins=num_clusters, range=[0, num_clusters], density=True)[0]

        return real_density, fake_density


    def compute_PRD(self, real_density, fake_density, num_angles=1001, epsilon=1e-10):
        angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
        slopes = np.tan(angles)

        slopes_2d = np.expand_dims(slopes, 1)

        real_density_2d = np.expand_dims(real_density, 0)
        fake_density_2d = np.expand_dims(fake_density, 0)

        precision = np.minimum(real_density_2d*slopes_2d, fake_density_2d).sum(axis=1)
        recall = precision / slopes

        max_val = max(np.max(precision), np.max(recall))
        if max_val > 1.001:
            raise ValueError('Detected value > 1.001, this should not happen.')
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)

        return precision, recall

    def compute_precision_recall(self, dataloader, gen, num_generate, num_runs, num_clusters,
                                 batch_size,  num_angles=1001):
        dataset_iter = iter(dataloader)
        n_batches = int(math.ceil(float(num_generate) / float(batch_size)))
        for i in tqdm(range(n_batches), disable = self.disable_tqdm):
            real_images = next(dataset_iter).cuda()
            fake_images = gen.sample(batch_size=batch_size)

            real_embed = self.inception_softmax(real_images).detach().cpu().numpy()
            fake_embed = self.inception_softmax(fake_images).detach().cpu().numpy()
            if i == 0:
                real_embeds = np.array(real_embed, dtype=np.float64)
                fake_embeds = np.array(fake_embed, dtype=np.float64)
            else:
                real_embeds = np.concatenate([real_embeds, np.array(real_embed, dtype=np.float64)], axis=0)
                fake_embeds = np.concatenate([fake_embeds, np.array(fake_embed, dtype=np.float64)], axis=0)

        real_embeds = real_embeds[:num_generate]
        fake_embeds = fake_embeds[:num_generate]

        precisions = []
        recalls = []
        for _ in range(num_runs):
            real_density, fake_density = self.cluster_into_bins(real_embeds, fake_embeds, num_clusters)
            precision, recall = self.compute_PRD(real_density, fake_density, num_angles=num_angles)
            precisions.append(precision)
            recalls.append(recall)

        mean_precision = np.mean(precisions, axis=0)
        mean_recall = np.mean(recalls, axis=0)

        return mean_precision, mean_recall

    def compute_fid_pr(self, diffusion, dataloader, gen, num_generate, num_runs, num_clusters,
                                 batch_size, splits,fid_cache,  num_angles=1001):
        dataset_iter = iter(dataloader)
        ys = []
        total_instance = num_generate
        pred_arr = np.empty((total_instance, 2048))
        n_batches = int(math.ceil(float(num_generate) / float(batch_size)))
        for i in tqdm(range(n_batches), disable = self.disable_tqdm):
            start = i * batch_size
            end = start + batch_size
            real_images = next(dataset_iter).cuda()
            fake_images = diffusion.p_sample_loop(gen, real_images.shape, 'cuda')

            real_embed = self.inception_softmax(real_images).detach().cpu().numpy()
            fake_embed = self.inception_softmax(fake_images).detach().cpu().numpy()
            if i == 0:
                real_embeds = np.array(real_embed, dtype=np.float64)
                fake_embeds = np.array(fake_embed, dtype=np.float64)
            else:
                real_embeds = np.concatenate([real_embeds, np.array(real_embed, dtype=np.float64)], axis=0)
                fake_embeds = np.concatenate([fake_embeds, np.array(fake_embed, dtype=np.float64)], axis=0)
            with torch.no_grad():
                embeddings, logits = self.inception_model(fake_images)
                y = torch.nn.functional.softmax(logits, dim=1)
            ys.append(y)

            if total_instance >= batch_size:
                pred_arr[start:end] = embeddings.cpu().data.numpy().reshape(batch_size, -1)
            else:
                pred_arr[start:] = embeddings[:total_instance].cpu().data.numpy().reshape(total_instance, -1)

            total_instance -= fake_images.shape[0]
        real_embeds = real_embeds[:num_generate]
        fake_embeds = fake_embeds[:num_generate]

        precisions = []
        recalls = []
        for _ in range(num_runs):
            real_density, fake_density = self.cluster_into_bins(real_embeds, fake_embeds, num_clusters)
            precision, recall = self.compute_PRD(real_density, fake_density, num_angles=num_angles)
            precisions.append(precision)
            recalls.append(recall)

        mean_precision = np.mean(precisions, axis=0)
        mean_recall = np.mean(recalls, axis=0)
        with torch.no_grad():
            ys = torch.cat(ys, 0)
        is_scores, is_std = kl_scores(ys[:num_generate], splits=splits)
        m1 = np.mean(pred_arr, axis=0)
        s1 = np.cov(pred_arr, rowvar=False)
        f = np.load(fid_cache)
        m2, s2 = f['mu'][:], f['sigma'][:]
        f.close()
        fid_score = calculate_frechet_distance(m1, s1, m2, s2)
        return is_scores, fid_score, mean_precision, mean_recall
    def compute_fid_pr_animeface(self, dataloader, gen, num_generate, num_runs, num_clusters,
                                 batch_size, splits,num_angles=1001):
        dataset_iter = iter(dataloader)
        ys = []
        total_instance = num_generate
        pred_arr = np.empty((total_instance, 2048))
        pred_ar = np.empty((total_instance, 2048))
        n_batches = int(math.ceil(float(num_generate) / float(batch_size)))
        for i in tqdm(range(n_batches), disable = self.disable_tqdm):
            start = i * batch_size
            end = start + batch_size
            real_images = next(dataset_iter).cuda()
            fake_images = gen.sample(batch_size=batch_size)

            real_embed = self.inception_softmax(real_images).detach().cpu().numpy()
            fake_embed = self.inception_softmax(fake_images).detach().cpu().numpy()
            if i == 0:
                real_embeds = np.array(real_embed, dtype=np.float64)
                fake_embeds = np.array(fake_embed, dtype=np.float64)
            else:
                real_embeds = np.concatenate([real_embeds, np.array(real_embed, dtype=np.float64)], axis=0)
                fake_embeds = np.concatenate([fake_embeds, np.array(fake_embed, dtype=np.float64)], axis=0)
            with torch.no_grad():
                embeddings, logits = self.inception_model(fake_images)
                embeddings_r, logits_r = self.inception_model(real_images)
                y = torch.nn.functional.softmax(logits, dim=1)
            ys.append(y)

            if total_instance >= batch_size:
                pred_arr[start:end] = embeddings.cpu().data.numpy().reshape(batch_size, -1)
                pred_ar[start:end] = embeddings_r.cpu().data.numpy().reshape(batch_size, -1)
            else:
                pred_arr[start:] = embeddings[:total_instance].cpu().data.numpy().reshape(total_instance, -1)
                pred_ar[start:] = embeddings_r[:total_instance].cpu().data.numpy().reshape(total_instance, -1)


            total_instance -= fake_images.shape[0]
        real_embeds = real_embeds[:num_generate]
        fake_embeds = fake_embeds[:num_generate]

        precisions = []
        recalls = []
        for _ in range(num_runs):
            real_density, fake_density = self.cluster_into_bins(real_embeds, fake_embeds, num_clusters)
            precision, recall = self.compute_PRD(real_density, fake_density, num_angles=num_angles)
            precisions.append(precision)
            recalls.append(recall)

        mean_precision = np.mean(precisions, axis=0)
        mean_recall = np.mean(recalls, axis=0)
        with torch.no_grad():
            ys = torch.cat(ys, 0)
        is_scores, is_std = kl_scores(ys[:num_generate], splits=splits)
        m1 = np.mean(pred_arr, axis=0)
        s1 = np.cov(pred_arr, rowvar=False)
        m2 = np.mean(pred_ar, axis=0)
        s2 = np.cov(pred_ar, rowvar=False)
        fid_score = calculate_frechet_distance(m1, s1, m2, s2)
        return is_scores, fid_score, mean_precision, mean_recall
    def compute_f_beta(self, precision, recall, beta=1, epsilon=1e-10):
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + epsilon)


def calculate_f_beta_score(diffusion,dataloader, gen, num_generate, num_runs, num_clusters, beta,splits,fid_cache,device):
    inception_model = InceptionV3().cuda()
    inception_model.eval()

    batch_size = dataloader.batch_size
    PR = precision_recall(inception_model, device=device)
    # precision, recall = PR.compute_precision_recall(dataloader, gen,
    #                                 num_generate, num_runs, num_clusters, batch_size)
    is_scores,fid_score,precision, recall = PR.compute_fid_pr(diffusion,dataloader, gen,
                                    num_generate, num_runs, num_clusters, batch_size,splits,fid_cache)
    if not ((precision >= 0).all() and (precision <= 1).all()):
        raise ValueError('All values in precision must be in [0, 1].')
    if not ((recall >= 0).all() and (recall <= 1).all()):
        raise ValueError('All values in recall must be in [0, 1].')
    if beta <= 0:
        raise ValueError('Given parameter beta %s must be positive.' % str(beta))

    f_beta = np.max(PR.compute_f_beta(precision, recall, beta=beta))
    f_beta_inv = np.max(PR.compute_f_beta(precision, recall, beta=1/beta))
    return is_scores,fid_score, precision, recall, f_beta, f_beta_inv
def calculate_f_beta_score_animeface(dataloader, gen, num_generate, num_runs, num_clusters, beta,splits):
    inception_model = InceptionV3().cuda()
    inception_model.eval()

    batch_size = dataloader.batch_size
    PR = precision_recall(inception_model, device='gpu')
    # precision, recall = PR.compute_precision_recall(dataloader, gen,
    #                                 num_generate, num_runs, num_clusters, batch_size)
    is_scores,fid_score,precision, recall = PR.compute_fid_pr_animeface(dataloader, gen,
                                    num_generate, num_runs, num_clusters, batch_size,splits)
    if not ((precision >= 0).all() and (precision <= 1).all()):
        raise ValueError('All values in precision must be in [0, 1].')
    if not ((recall >= 0).all() and (recall <= 1).all()):
        raise ValueError('All values in recall must be in [0, 1].')
    if beta <= 0:
        raise ValueError('Given parameter beta %s must be positive.' % str(beta))

    f_beta = np.max(PR.compute_f_beta(precision, recall, beta=beta))
    f_beta_inv = np.max(PR.compute_f_beta(precision, recall, beta=1/beta))
    return is_scores,fid_score, precision, recall, f_beta, f_beta_inv
def plot_pr_curve(precision, recall, results_dir):



    save_path = os.path.join(results_dir +'/' + "pr_curve.png")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(recall, precision)
    ax.grid(True)
    ax.set_xlabel('Recall (Higher is better)', fontsize=15)
    ax.set_ylabel('Precision (Higher is better)', fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path)
    return fig