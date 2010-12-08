import numpy
from numpy import array
import scipy.spatial.distance as scid
from PIL import Image
from os import listdir
from mdp import pca

ALLOWED_EXT = ['jpeg', 'gif', 'jpg']

# constants for various metrics
EUCLIDEAN = 0
MANHATTAN = 1
MAHALANOBIS = 2

class PCA_Classifier:
    def __init__(self):
        self.mean_vector = None
        self.face_classes = {}
        self.eigenfaces = {}
        
    def train(self):
        for i in face_classes:
            self.eigenfaces[i] = pca(self.face_classes[i])

    def batch_image_process(self, directory, label):
        """
            Args: a directory containing images
            Returns: A list containing a vector for every image in the
                     input directory
        """
        files = listdir(directory)
        image_vectors = []

        for file in files:
            if allow_file(file):
                vector = self.vectorize_image(directory + '/' + file)
                image_vectors.append(vector)
                
        self.face_classes[label] = image_vectors

    def vectorize_image(self, file_path):
        """
            Args: The path to an image file on disk
            Returns: A one-dimensional numpy array of the image
        """
        image = Image.open(file_path)
        matrix = numpy.array(image)
        return matrix.flatten()
        
    def allow_file(self, filename):
        """
            Args: A filename
            Returns: True if the file specifies an image we can use;
                     False otherwise.
        """
        for ext in ALLOWED_EXT:
            if filename.endswith(ext):
                return True
        return False

def distance(vec1, vec2, metric=EUCLIDEAN):
    '''
        Args: two vectors to be compared and metric to be used
            0 (default): euclidean
            1: manhattan
            2: mahalanobis (not yet implemented)
        Returns: distance according to given metric
    '''
    if metric is EUCLIDEAN:
        return scid.euclidean(vec1, vec2)
    elif metric is MANHATTAN:
        return scid.cityblock(vec1, vec2)
    elif metric is MAHALANOBIS:
        return "NOT YET IMPLEMENTED"
    else:
        return "Please choose as valid value for the metric"

def calc_weight_vector(normed_face, eigenfaces):
    '''
    Takes a normalized face (face minus the average face) and an array
    of eigenfaces
    Returns a weight vector for the face.
    '''
    weight_vector = list()
    for eigenface in eigenfaces:
        weight = numpy.dot(eigenface, normed_face)
        weight_vector.append(weight)
    return numpy.array(weight_vector)
        

def calc_group_weights(group_dict, eigenfaces, mean_face):
    '''
    Takes a dictionary in the form of {face_group_label -> group_image_vectors}
    a vector of eigenfaces and a vector containing the mean face
    Returns a dictionary in the form {face_group_label -> group_weight_vector}
    '''
    
    weight_dict = dict()
    for label, group in group_dict.iteritems():
        
        group_face = 0 * group[0]
        for face in group:
            group_face += face
        group_face /= len(group)
        normed_group_face = group_face - mean_face

        weight_vector = calc_weight_vector(normed_group_face, eigenfaces)

        weight_dict[label] = weight_vector

    return weight_dict

def project_to_face_space(new_face, eigenfaces, mean_face):
    '''
    Takes a face vector, eigenfaces, and the mean face
    Returns the weight vector for the new face
    '''
    normed_face = new_face - mean_face
    return calc_weight_vector(normed_face, eigenfaces)

def label_face(training_groups, new_face, distance_metric, eigenfaces, mean_face):
    '''
    Takes a dictionary of labels -> images, a new face, a distance metric,
    a vector of eigenfaces and the mean face
    Returns the most likely label for the face
    '''
    
    group_weights = calc_group_weights(training_groups, eigenfaes, mean_face)
    new_weight = project_to_face_space(new_face, eigenfaces, mean_face)
    face_threshold = 0.0
    new_face_threshold = 0.0
    min_group = ""
    min_distance = INFINITY
    for group, weight in group_weights.iteritems():
        d = distance(weight, new_weight, distance_metric)
        if d < min_distance:
            min_group = group
            min_distance = distance
    if min_distance < face_threshold:
        return "not a face"
    elif min_distance < new_face_threshold:
        return "new face"
    else:
        return min_group
    
    
