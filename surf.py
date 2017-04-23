'''
Created on Mar 25, 2017

@author: andrey.justo
'''

from os import path
from os import listdir
from os.path import isfile, join
import pickle as pkl

# sklearn libs
import sklearn.preprocessing as preprocessing
import numpy as np

#opencv libs
import cv2
from cv2 import xfeatures2d

relpath = lambda x: path.abspath(path.join(path.dirname(__file__), x))

def load_feature_vectors(image_path, extractor):
    # Load the images
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kp, desc = extractor.detectAndCompute(g_img, None)
    print('For:', image_path, 'KeyPoints:', len(kp), 'Descriptor Shape:', desc.shape if desc is not None else None)
 
    return kp, desc
    
    
def extractor(hessian_threshold):
    # SURF extraction
    surf = xfeatures2d.SURF_create(hessian_threshold)
    surf.setExtended(True)
    
    return surf

def write_features(features, file_path):
    f = open(file_path, "w")
    f.write(pkl.dumps(features))
    f.close()

def load_feature_vectors_from_sample(images_trainning_dir, feature_vector_dir, hessian_threshold, should_store=True):
    
    all_features = []
    e = extractor(hessian_threshold)
    for f in listdir(images_trainning_dir):
        file_path = join(images_trainning_dir, f)
        file_name = path.basename(file_path).split('.')

        if isfile(file_path) and len(file_name) > 1:
            features = load_feature_vectors(file_path, e)
            if features[1] is not None:
                norm = normalize_feature(features[1])
                all_features.append(norm)
    
    if (should_store):
        write_features(all_features, feature_vector_dir + 'vectors-' + `hessian_threshold` + '.fvecs')
    
    return compact(all_features)

def normalize_feature(data):
    return preprocessing.normalize(data, norm='l2')

def compact(data):
    data_array = None
    for d in data:
        if data_array is None:
            data_array = d
        else:
            data_array = np.vstack((data_array, d))
            
    return data_array