import numpy as np
import data
import torch
from sklearn.metrics.pairwise import cosine_similarity as cos
from tqdm import tqdm

# Rarest class: zebra, giraffe, umbrella
TARGET_CLASS = 'zebra'
SEARCH_RADIUS = 20000 
BATCH_SIZE = 20
MODEL_NAME = 'pca200_vgg'


def smart_search(target_list, data, k, dataloader, target_idx, batch_size):
    """
    target_list: 5 x d
    data: 200,000 x d 
    """
    positive_list = target_list.copy()
    negative_list = []
    all_idx = [x for x in range(len(data))]
    score = compute_score(all_idx, data, positive_list, negative_list) # dim = 200,000
    all_tops = []
    print("Performing search...")
    for _ in tqdm(range(k // batch_size)):
        tops = []
        for _ in range(batch_size):
            top = max(score, key=lambda x: x[0])
            tops.append(top)
            all_tops.append(top[1])
            score.remove(top)

        for _, ind in tops:
            class_no = dataloader.get_id_class(ind)
            if class_no == target_idx:
                positive_list.append(ind)
            else:
                negative_list.append(ind)

        remaining_idx = [x[1] for x in score]
        score = compute_score(remaining_idx, data, positive_list, negative_list)

    return all_tops


def compute_score(indices, data, positive_list, negative_list):
    score = cos(data[positive_list], data[indices]) 
    score = np.sum(score, axis=0)
    if negative_list:
        neg_sim = cos(data[negative_list], data[indices])
        score -= np.sum(neg_sim, axis=0)
    res = []
    for i in range(len(score)):
        res.append((score[i], indices[i]))
    return res


def main():
    k = SEARCH_RADIUS
    batch_size = BATCH_SIZE
    dataloader = data.DataLoader('/lfs/1/zhyzhang/yt-videos/yt_bb_detection_validation/', '../cluster.db')
    print('loading weights')
    outputs = np.loadtxt('{}_final_weights.txt'.format(MODEL_NAME))
    print(outputs.shape)
    print('done loading weights')
    # targets = data.DataLoader.get_targets(TARGET_CLASS)

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

    # targets = [outputs[x] for x in targets]

    top = smart_search(targets, outputs, k, dataloader, target_idx, batch_size)
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
    np.savetxt('results/{}-{}-smartcos-cluster-prec.txt'.format(MODEL_NAME, str(TARGET_CLASS)), prec)
    np.savetxt('results/{}-{}-smartcos-cluster-recall.txt'.format(MODEL_NAME, str(TARGET_CLASS)), recall)


if __name__ == "__main__":
    main()
