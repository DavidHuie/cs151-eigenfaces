import numpy
from numpy import array
import scipy.spatial.distance as scid
from PIL import Image
from os import listdir
from mdp import pca
import re
import pickle
import shutil

# constants for various metrics
EUCLIDEAN = 0
MANHATTAN = 1
MAHALANOBIS = 2

SIZE = (60, 90)

# sleeping eight
INFINITY = 1e40

class PCA_Classifier:
    def __init__(self):
        self.mean_vector = None
        self.face_classes = {}
        self.eigenfaces = []
        self.big = None

    def train(self):
        '''
        uses images as rows
        '''
        imList = []
        for cl in self.face_classes.itervalues():
            for im in cl:
                imList.append(im)
                
       # print imList
        self.big = numpy.vstack(array(imList).astype('float64'))
        print len(self.big)
        print len(self.big[0])
        #print self.big
        self.mean_vector = numpy.mean(self.big, 0)
        self.eigenfaces = pca(self.big, svd = True)
        #print self.eigenfaces

    def train2(self):
        '''
        uses images as columns
        '''
        imList = []
        for cl in self.face_classes.itervalues():
            for im in cl:
                imList.append(im)

        #self.big = array(numpy.hstack(numpy.matrix(imList).transpose().astype('f')))
        self.big = numpy.vstack(array(imList).astype('float64'))
        self.mean_vector = numpy.mean(self.big, 0)
        mean_matrix = numpy.array([self.mean_vector]*len(self.big))
        big = self.big - mean_matrix
        print len(big)
        print len(big[0])
        self.eigenfaces = numpy.transpose(pca(numpy.transpose(big), 
                                              svd = True, output_dim = 10))
        print self.mean_vector


    def save_eigenfaces(self, filename):
        with open(filename,'w') as f:
            p = pickle.Pickler(f)
            p.dump(self.eigenfaces)
            p.dump(self.big)
            p.dump(self.mean_vector)
            p.dump(self.face_classes)

    def load_eigenfaces(self, filename):
        with open(filename, 'r') as f:
            u = pickle.Unpickler(f)
            self.eigenfaces = u.load()
            self.big = u.load()
            self.mean_vector = u.load()
            self.face_classes = u.load()

    def print_face_classes(self):
        print self.face_classes

    def batch_label_process(self, directory):
        """
        Args: a directory containing folders with images
        Returns: Nothing
        """
        files = listdir(directory)
        for file in files:
            if re.match("\w+", file):
                self.batch_image_process(directory + '/' + file, file)

    def batch_label_process2(self, directory):
        """
        Args: a directory containing folders with images
        Returns: Nothing
        """
        files = listdir(directory)
        for file in files:
            if re.match("\w+", file):
                match = re.search("[1-9][0-9]*", file)
                label = str(file[match.start():match.end()])
                vector = self.vectorize_image(directory + '/' + file)
                if label in self.face_classes:
                    self.face_classes[label].append(vector)
                else:
                    self.face_classes[label] = [vector]

    def batch_image_process(self, directory, label):
        """
            Args: a directory containing images
            Returns: A list containing a vector for every image in the
                     input directory
        """
        files = listdir(directory)
        image_vectors = []

        for file in files:
            if self.allow_file(file):
                #print file
                vector = self.vectorize_image(directory + '/' + file)
                image_vectors.append(vector)

        #print image_vectors

        self.face_classes[label] = image_vectors
    
    def vectorize_image(self, file_path):
        """
            Args: The path to an image file on disk
            Returns: A one-dimensional numpy array of the image
        """
        image = Image.open(file_path).resize(SIZE)
        matrix = numpy.array(image)
        return matrix.flatten()

    def allow_file(self, filename):
        """
            Args: A filename
            Returns: True if the file specifies an image we can use;
                     False otherwise.
        """
        if re.match("\w+", filename):
                return True
        return False


    def distance(self, vec1, vec2, metric=EUCLIDEAN):
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

    def calc_weight_vector(self, normed_face):
        '''
        Takes a normalized face (face minus the average face) and an array
        of eigenfaces
        Returns a weight vector for the face.
        '''
        weight_vector = list()
        for eigenface in self.eigenfaces:
            weight = numpy.dot(eigenface, normed_face)
            weight_vector.append(weight)
        return numpy.array(weight_vector)


    def calc_group_weights(self):
        '''
        Takes a dictionary in the form of {face_group_label -> group_image_vectors}
        a vector of eigenfaces and a vector containing the mean face
        Returns a dictionary in the form {face_group_label -> group_weight_vector}
        '''

        weight_dict = dict()
        for label, group in self.face_classes.iteritems():

            group_face = 0 * group[0]
            for face in group:
                group_face += face
            group_face /= len(group)
            normed_group_face = group_face - self.mean_vector

            weight_vector = self.calc_weight_vector(normed_group_face)

            weight_dict[label] = weight_vector

        return weight_dict

    def project_to_face_space(self, new_face):
        '''
        Takes a face vector, eigenfaces, and the mean face
        Returns the weight vector for the new face
        '''

        print len(new_face)
        print len(self.mean_vector)
        normed_face = new_face - self.mean_vector
        return self.calc_weight_vector(normed_face)

    def label_face(self, new_face, distance_metric):
        '''
        Takes a dictionary of labels -> images, a new face, a distance metric,
        a vector of eigenfaces and the mean face
        Returns the most likely label for the face
        '''

        group_weights = self.calc_group_weights()
        new_weight = self.project_to_face_space(new_face)
        face_threshold = 0.0
        new_face_threshold = 0.0
        min_group = ""
        min_distance = INFINITY
        for group, weight in group_weights.iteritems():
            d = self.distance(weight, new_weight, distance_metric)
            if d < min_distance:
                min_group = group
                min_distance = d
        if min_distance < face_threshold:
            return "not a face"
        elif min_distance < new_face_threshold:
            return "new face"
        else:
            return min_group

    def test_classifier(self, directory, metric):
        files = listdir(directory)
        correct = 0
        total = 0
        for file in files:
            if re.match("\w+", file):
                face = self.vectorize_image(directory + '/' + file)
                label = self.label_face(face, metric)
                match = re.search("[1-9][0-9]*", file)
                actual_label = str(file[match.start():match.end()])
                if label == actual_label:
                    correct +=1
                total += 1
        print correct
        print total

def partition_test_train(directory):
    # calc number of images in directory
    length = 0
    files = listdir(directory)
    for file in files:
        if re.match("\w+", file) and file != 'test' and file != 'train':
            length += 1
    test_indexes = []
    while len(test_indexes) <(length*.2):
        x = random.randrange(0, length)
        if not(x in test_indexes):
            test_indexes.append(x)
    i = 0
    for file in files:
        if re.match("\w+", file) and file != 'test' and file != 'train':
            if i in test_indexes:
                shutil.move(directory + '/' + file,  directory + '/' + 'test')
            else:
                shutil.move(directory + '/' + file,  directory + '/' + 'train')
        i += 1
                
    