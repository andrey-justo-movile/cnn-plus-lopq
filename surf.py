'''
Created on Mar 25, 2017

@author: andrey.justo
'''

import time
from os import path
from os import listdir
from os.path import isfile, join
import pickle as pkl

# sklearn libs
from sklearn.cross_validation import train_test_split
import sklearn.preprocessing as preprocessing
import numpy as np

#opencv libs
import cv2
from cv2 import xfeatures2d

# lopq
from lopq import LOPQModel, LOPQSearcher
from lopq.eval import compute_all_neighbors, get_recall
from lopq.model import eigenvalue_allocation

relpath = lambda x: path.abspath(path.join(path.dirname(__file__), x))

def load_feature_vectors(image_path, fvector_path, hessian_threshold, should_store):
    # Load the images
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # SURF extraction
    surf = xfeatures2d.SURF_create(hessian_threshold)
    surf.setExtended(True)
    kp, desc = surf.detectAndCompute(g_img, None)
    print('For:', image_path, 'KeyPoints:', len(kp), 'Descriptor Shape:', desc.shape if desc is not None else None)
    
    if (should_store):
        write_features(kp, desc, fvector_path)
 
    return kp, desc
    
def write_features(kp, desc, file_path):
    f = open(file_path, "w")
    f.write(pkl.dumps(np.array(desc)))
    f.close()

def load_feature_vectors_from_sample(images_trainning_dir, feature_vector_dir, hessian_threshold, should_store):
    
    all_features = None
    for f in listdir(images_trainning_dir):
        file_path = join(images_trainning_dir, f)
        file_name = path.basename(file_path).split('.')

        if isfile(file_path) and len(file_name) > 1:
            features = load_feature_vectors(file_path, feature_vector_dir + file_name[0], hessian_threshold, should_store)
            if features[1] is not None:
                norm = normalize_feature(features[1])
                if all_features is None:
                    all_features = norm
                else:
                    all_features = np.vstack((all_features, norm))
            
    return all_features

def normalize_feature(data):
    return preprocessing.normalize(data, norm='l2')

def pca(data):
    """
    A simple PCA implementation that demonstrates how eigenvalue allocation
    is used to permute dimensions in order to balance the variance across
    subvectors. There are plenty of PCA implementations elsewhere. What is
    important is that the eigenvalues can be used to compute a variance-balancing
    dimension permutation.
    """

    # Compute mean
    count, D = data.shape
    mu = data.sum(axis=0) / float(count)

    # Compute covariance
    summed_covar = reduce(lambda acc, x: acc + np.outer(x, x), data, np.zeros((D, D)))
    A = summed_covar / (count - 1) - np.outer(mu, mu)

    # Compute eigen decomposition
    eigenvalues, P = np.linalg.eigh(A)

    # Compute a permutation of dimensions to balance variance among 2 subvectors
    permuted_inds = eigenvalue_allocation(2, eigenvalues)

    # Build the permutation into the rotation matrix. One can alternately keep
    # these steps separate, rotating and then permuting, if desired.
    P = P[:, permuted_inds]

    return P, mu

def main():
    
    images_trainning_dir = './data/sample/'
    feature_vector_dir = './data/vectors/'
    hessian_threshold = 5000
    
    # Loading dataset
    a = time.time()
    data = load_feature_vectors_from_sample(images_trainning_dir, feature_vector_dir, hessian_threshold, False)
    print('Finished Loading SURF vectors:', (time.time() - a))

    # about this dataset.
    P, mu = pca(data)
 
    # Mean center and rotate the data; includes dimension permutation.
    # It is worthwhile see how this affects recall performance. On this
    # dataset, which is already PCA'd from higher dimensional features,
    # this additional step to variance balance the dimensions typically
    # improves recall@1 by 3-5%. The benefit can be much greater depending
    # on the dataset.
    data = data - mu
    data = np.dot(data, P)
    print('After PCA')

    # Create a train and test split. The test split will become
    # a set of queries for which we will compute the true nearest neighbors.
    random_state = 40
    train, test = train_test_split(data, test_size=0.2, random_state=random_state)
    print('After Split data')
    
    # Compute distance-sorted neighbors in training set for each point in test set.
    # These will be our groundtruth for recall evaluation.
    a = time.time()
    nns = compute_all_neighbors(test, train)
    print('Finished Calculating NNs:', (time.time() - a))
    
    # First test parameters
    a = time.time()
    m = create_model(train, 16, 8, 256, None)
    print('Finish Fiting Model:', (time.time() - a))

    a = time.time()
    search_and_recall(m, train, test, nns)
    print('Finish Searching in Model:', (time.time() - a))

    # Second test parameters
    a = time.time()
    m2 = create_model(train, 16, 16, 256, (m.Cs, None, None, None))
    print('Finish Fiting Model:', (time.time() - a))

    a = time.time()
    search_and_recall(m2, train, test, nns)
    print('Finish Searching Model:', (time.time() - a))

    # Third test parameters
    a = time.time()
    m3 = create_model(train, 16, 8, 512, (m.Cs, m.Rs, m.mus, None))
    print('Finish Fiting Model:', (time.time() - a))

    a = time.time()
    search_and_recall(m3, train, test, nns)
    print('Finish Searching Model:', (time.time() - a))


def create_model(train, v, m, clusters, parameters):
    model = LOPQModel(V=v, M=m, subquantizer_clusters=clusters, parameters=parameters)
    model.fit(train, n_init=1)
    return model

def search_and_recall(model, train, test, nns):
    searcher = LOPQSearcher(model)
    searcher.add_data(train)
    recall, _ = get_recall(searcher, test, nns)
    print 'Recall (V=%d, M=%d, subquants=%d): %s' % (model.V, model.M, model.subquantizer_clusters, str(recall))
    

if __name__ == '__main__':
    main()
