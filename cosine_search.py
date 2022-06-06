import numpy as np
import data
import torch
from tqdm import tqdm

# Rarest class: zebra, giraffe, umbrella
TARGET_CLASS = 'umbrella'
SEARCH_RADIUS = 190000 
MODEL_NAME = 'pca200_resnet'


def knn_multiple(target_list, data, k):
    sims = []
    for i, target in enumerate(target_list):
        print("Performing KNN on {}th target".format(i+1))
        cos = torch.nn.CosineSimilarity()
        target = np.expand_dims(target, axis=0)
        sim = cos(torch.Tensor(data), torch.Tensor(target))
        sims.append(np.array(sim))
    print("Done clustering")
    sims = np.array(sims)
    sims = np.max(sims, axis=0)
    print(sims)
    print(sims.shape)
    num_ones = np.count_nonzero(sims==1)
    sims = -sims
    top_ind = np.argpartition(sims, k+num_ones)[:k+num_ones]
    top_ind_sorted = top_ind[np.argsort(sims[top_ind])]
    return top_ind_sorted[num_ones:]


def main():
    print(TARGET_CLASS, SEARCH_RADIUS, MODEL_NAME)

    k = SEARCH_RADIUS
    dataloader = data.DataLoader('/lfs/1/zhyzhang/yt-videos/yt_bb_detection_validation/', '../cluster.db')
    print('loading weights')
    outputs = np.loadtxt('{}_final_weights.txt'.format(MODEL_NAME))
    print(outputs.shape)
    print('done loading weights')

    # Hardcoded for now
    if TARGET_CLASS == 'zebra':
        target_idx = 17
        targets = [41827, 47243, 126844, 152111, 191117]
        total_positives = 510
    elif TARGET_CLASS == 'giraffe':
        target_idx = 8
        targets = [102451, 9271, 82692, 72012, 8570]
        total_positives = 963
    elif TARGET_CLASS == 'umbrella':
        target_idx = 21
        targets = [68737, 62819, 143393, 150118, 69906]
        total_positives = 3881

    targets = [outputs[x] for x in targets]

    top = knn_multiple(targets, outputs, k)
    print(top)
    num_positive = 0
    prec = []
    recall = []
    print("Calculating precision and recall...")
    for i in tqdm(range(len(top))):
        image_id = top[i]
        class_no = dataloader.get_id_class(image_id)
        if class_no == target_idx:
            num_positive += 1
        prec.append(num_positive / (i+1))
        recall.append(num_positive / total_positives)
    prec = np.array(prec)
    np.savetxt('results/{}-{}-cos-cluster-prec.txt'.format(MODEL_NAME, str(TARGET_CLASS)), prec)
    np.savetxt('results/{}-{}-cos-cluster-recall.txt'.format(MODEL_NAME, str(TARGET_CLASS)), recall)


if __name__ == "__main__":
    main()
