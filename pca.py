import numpy as np
from sklearn.decomposition import PCA


def main():
    num_components = 1000 
    model_name = 'resnet'

    print('loading weights')
    outputs = np.loadtxt('{}_final_weights.txt'.format(model_name))
    pca = PCA(n_components=num_components)
    print('doing PCA')
    pca.fit(outputs)
    print('PCA done, transforming data')
    reduced_outputs = pca.transform(outputs)
    print('saving to txt file')
    np.savetxt('pca{}_{}_final_weights.txt'.format(num_components, model_name), np.array(reduced_outputs))


if __name__ == "__main__":
    main()
