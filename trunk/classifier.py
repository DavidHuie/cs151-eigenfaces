import numpy
from numpy import array
import scipy.spatial.distance as scid
from PIL import Image
from os import listdir, makedirs, path
from mdp import pca
import re
import pickle
import shutil
import random

# constants for various metrics
EUCLIDEAN = 0
MANHATTAN = 1
MAHALANOBIS = 2

# sleeping eight
INFINITY = float('infinity')

SAVE_EXT = '.gif'

RESIZE = True
RESIZE_SIZE = (60, 90)

class PCA_Classifier:
    def __init__(self):
        self.mean_vector = None
        self.face_classes = {}
        self.eigenfaces = []
        self.big = None
        self.input_image_dimensions = None

    def train(self):
        '''
            Trains the classifier using saved face_classes
        '''
        print "Training classifier using face classes..."
        imList = []
        for cl in self.face_classes.itervalues():
            for im in cl:
                imList.append(im)

        self.big = numpy.vstack(array(imList).astype('float64'))
        self.mean_vector = numpy.mean(self.big, 0)
        mean_matrix = numpy.array([self.mean_vector]*len(self.big))
        big = self.big - mean_matrix
        self.unnormed_eigenfaces = numpy.transpose(pca(numpy.transpose(big), output_dim = .95, svd = True))
        self.eigenfaces = [i/numpy.linalg.norm(i) for i in self.unnormed_eigenfaces]

    def save_eigenface_images(self, dir):
        """
            Args: directory where eigenfaces should be saved
            Returns: nothing
        """
        print "Saving eigenfaces..."
        if dir[-1] != '/':
            dir = dir + '/'
            
        if not path.exists(dir):
            makedirs(dir)
            
        if RESIZE:
            images = [Image.fromarray(numpy.reshape(i + self.mean_vector, RESIZE_SIZE)) \
                  for i in self.unnormed_eigenfaces]
        else:
            images = [Image.fromarray(numpy.reshape(i + self.mean_vector, self.input_image_dimensions)) \
                  for i in self.unnormed_eigenfaces]
                  
        num = len(images)
        for i in range(1, num + 1):
            images[i-1].save(dir + 'eigenface' + str(i) + SAVE_EXT)


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
        print "Labeling and vectorizing training images..."
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
                vector = self.vectorize_image(directory + '/' + file)
                image_vectors.append(vector)

        self.face_classes[label] = image_vectors
    
    def vectorize_image(self, file_path):
        """
            Args: The path to an image file on disk
            Returns: A one-dimensional numpy array of the image
        """
        image = Image.open(file_path)
        
        if RESIZE:
            image = image.resize(RESIZE_SIZE)
            self.input_image_dimensions = RESIZE_SIZE
        else:
            self.input_image_dimensions = (image.size[1], image.size[0])

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
            print "NOT YET IMPLEMENTED"
            return 0
        else:
            print "Please choose as valid value for the metric"
            return 0

    def project_to_face_space(self, new_face):
        '''
            Args:
                new_face: A face matrix
            Returns:
                An array containing the weight vectors for the input image
        '''
        normed_face = new_face - self.mean_vector
        weight_vector = list()
        
        for eigenface in self.eigenfaces:
            weight = numpy.dot(eigenface, normed_face)
            weight_vector.append(weight)
            
        return numpy.array(weight_vector)
        
    def label_face(self, face, distance_metric):
        """
            Args:
                face: A face matrix
                distance_metric: A distance metric
            Returns:
                A label that is guessed for the input face
                
            Determines label by comparing face to every image in every face class
        """
        omega_new_face = self.project_to_face_space(face)
        omega_face_classes = []
        
        # gather omegas for each image
        for label in self.face_classes:
            for face in self.face_classes[label]:
                omega_face_classes.append((self.project_to_face_space(face), label))
                
        # find min distance
        distances = []
        for face, label in omega_face_classes:
            distances.append((self.distance(face, omega_new_face, metric = distance_metric), label))
        
        min_face = min(distances)

        return min_face[1] # return the label

    def classify(self, directory, metric):
        """
            Args:
                directory: where testing images are found
                metric: distance metric to use
            Returns:
                Nothing
            
            Prints out statistics about the testing set (precision, recall, f)
        """
        print "Classifying images..."
        stats = {}
        labels = self.face_classes.keys()
        files = listdir(directory)
                
        # inititialize stats
        for label in labels:
            stats[label] = {}
            stats[label]['total'] = 0
            stats[label]['tp'] = 0
            stats[label]['fp'] = 0
            stats[label]['fn'] = 0
        
        correct = 0
        total = 0
        
        # gather stats
        for file in files:
            if re.match("\w+", file):
                face = self.vectorize_image(directory + '/' + file)
                guessed_label = self.label_face(face, metric)
                match = re.search("[1-9][0-9]*", file)
                actual_label = str(file[match.start():match.end()])
                
                stats[actual_label]['total'] += 1
                total += 1
                if guessed_label == actual_label:
                    stats[actual_label]['tp'] +=1
                    correct += 1
                else:
                    stats[actual_label]['fn'] += 1
                    stats[guessed_label]['fp'] += 1
        
        for label in labels:
            tp = stats[label]['tp']
            fp = stats[label]['fp']
            fn = stats[label]['fn']
            if tp+fp>0 and tp+fn>0:
                print label, "statistics:"
                # Print out stats
                p = 1.0*tp/(tp+fp)
                r = 1.0*tp/(tp+fn)
                f = 2 * (p*r)/(p+r)
                print "Precision:", p
                print "Recall:", r
                print "F-measure:", f
                print
                
        print "Accuracy Statistics:"
        print "Correct:", correct
        print "Total:", total
        print "Accuracy:", 1.0*correct/total
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

def main():
    db = raw_input("Enter a face database: ")
    if not path.exists(db):
        print "Incorrect face database."
        return    
    classifier = PCA_Classifier()
    classifier.batch_label_process(db + '/train')
    classifier.train()
    classifier.save_eigenface_images('efaces_' + db)
    classifier.classify(db + '/test', EUCLIDEAN)
    print "Done."
    
if __name__ == '__main__':
    main()
    
