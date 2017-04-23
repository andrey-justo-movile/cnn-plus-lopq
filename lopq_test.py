'''
Created on Apr 3, 2017

@author: andrey.justo
'''

from surf import load_feature_vectors_from_sample
from sklearn.model_selection import train_test_split
import time
import numpy as np

# lopq
from lopq import LOPQModel, LOPQSearcher
from lopq.model import eigenvalue_allocation
from lopq.eval import compute_all_neighbors, get_cell_histogram, get_recall

def transform(array):
    P, mu = pca(array)
    all_features = array - mu
    all_features = np.dot(all_features, P)
    print('After PCA')
    
    return all_features

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
    hessian_threshold = 8000
    
    # Loading dataset
    a = time.time()
    all_features = load_feature_vectors_from_sample(images_trainning_dir, feature_vector_dir, hessian_threshold)
    all_features = transform(all_features)
    train, test = train_test_split(all_features, test_size=0.2)
    print('Finished Load SURF vectors:', (time.time() - a), 'with shape:', all_features.shape)
     
    # First test parameters
    a = time.time()
    m = create_model(train, 8, 16, 256, None)
    print('Finish Fit Model:', (time.time() - a))
 
    a = time.time()
    calc_recall(m, train, test)
    print('Finish Search in Model:', (time.time() - a))
  
    # Second test parameters
    a = time.time()
    m2 = create_model(train, 16, 16, 256, (m.Cs, None, None, None))
    print('Finish Fit Model:', (time.time() - a))
  
    a = time.time()
    calc_recall(m2, train, test)
    print('Finish Search in Model:', (time.time() - a))
  
    # Third test parameters
    a = time.time()
    m3 = create_model(train, 16, 16, 512, (m.Cs, m.Rs, m.mus, None))
    print('Finish Fit Model:', (time.time() - a))
  
    a = time.time()
    calc_recall(m3, train, test)
    print('Finish Search in Model:', (time.time() - a))


def create_model(train, v, m, clusters, parameters):
    model = LOPQModel(V=v, M=m, subquantizer_clusters=clusters, parameters=parameters)
    model.fit(train, n_init=1)
        
    return model
    
def calc_recall(model, train, test, quota=[1], threshold=0.1): 

    searcher = LOPQSearcher(model)
    searcher.add_data(train)
    recall = np.zeros(len(quota))
    
    for d in test:
        search_parts(d, searcher, quota, recall, threshold)

    N = test.shape[0]
    print('Recall:', recall / N)
    return recall / N

def search_parts(data, searcher, quota, recall, threshold):
    results, _ = searcher.search(data, quota[-1], with_dists=True)
    print('Results:', results)
    for j, res in enumerate(results):
        item = res
 
        if item.dist <= threshold:
            for k, t in enumerate(quota):
                if j < t:
                    recall[k] += 1

if __name__ == '__main__':
    main()
