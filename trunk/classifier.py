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
import sys

# constants for various metrics
EUCLIDEAN = 0
MANHATTAN = 1

EIGENFACE_SAVE_EXT = '.gif'

RESIZE_SIZE = (120, 90)

TRAIN_PROPORTION = .8

class PCA_Classifier:
    def __init__(self, resize = False):
        self.mean_vector = None
        self.face_classes = {}
        self.eigenfaces = []
        self.omega_face_classes = []
        self.big = None
        self.input_image_dimensions = None
        self.resize = resize
        self.train_data = []
        self.test_data = []

    def train(self):
        '''
            Args: None
            Returns: Nothing
            
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
        
        self.calculate_omegas()

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
            
        images = [Image.fromarray(numpy.reshape(i + self.mean_vector, self.input_image_dimensions)) \
              for i in self.unnormed_eigenfaces]
                  
        num = len(images)
        for i in range(1, num + 1):
            images[i-1].save(dir + 'eigenface' + str(i) + EIGENFACE_SAVE_EXT)
    
    def vectorize_image(self, file_path):
        """
            Args: The path to an image file on disk
            Returns: A one-dimensional numpy array of the image
        """
        image = Image.open(file_path)
        
        if self.resize:
            image = image.resize(RESIZE_SIZE)
            
        if self.input_image_dimensions is None:
            self.input_image_dimensions = (image.size[1], image.size[0])

        matrix = numpy.array(image)
        return matrix.flatten()

    def distance(self, vec1, vec2):
        '''
            Args: two vectors to be compared
            Returns: Euclidean distance between the input vectors
        '''
        return scid.euclidean(vec1, vec2)

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
        
    def calculate_omegas(self):
        """
            Args: None
            Returns: Nothing
                
            Calculates an Omega value for each image in the face class.
        """
        for label in self.face_classes:
            for face in self.face_classes[label]:
                self.omega_face_classes.append((self.project_to_face_space(face), label))
        
    def label_face(self, face):
        """
            Args:
                face: A face matrix
                distance_metric: A distance metric
            Returns:
                A label that is guessed for the input face
                
            Determines label by comparing face to every image in every face class
        """
        omega_new_face = self.project_to_face_space(face)
                
        # find min distance
        distances = []
        for face, label in self.omega_face_classes:
            distances.append((self.distance(face, omega_new_face), label))
        
        min_face = min(distances)

        return min_face[1] # return the label

    def classify(self, print_stats = False):
        """
            Args:
                directory: where testing images are found
                metric: distance metric to use
            Returns:
                Nothing
            
            Prints out statistics about the testing set (precision, recall, f)
        """
        print "Classifying testing set..."
        print
        stats = {}
        labels = self.face_classes.keys()
                
        # inititialize stats
        for label in labels:
            stats[label] = {}
            stats[label]['total'] = 0
            stats[label]['tp'] = 0
            stats[label]['fp'] = 0
            stats[label]['fn'] = 0
        
        correct = 0
        total = 0
        
        for path, actual_label in self.test_data:
            face = self.vectorize_image(path)
            guessed_label = self.label_face(face)
            stats[actual_label]['total'] += 1
            total += 1
            if guessed_label == actual_label:
                stats[actual_label]['tp'] +=1
                correct += 1
            else:
                stats[actual_label]['fn'] += 1
                stats[guessed_label]['fp'] += 1
        
        if print_stats:
            labels.sort()
            for label in labels:
                tp = stats[label]['tp']
                fp = stats[label]['fp']
                fn = stats[label]['fn']
                if tp + fp > 0 and tp + fn > 0:
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
        print
        
    def partition_data(self, directory):
        """
            Args:
                directory: a directory containing folders
                           corresponding to each image class.
            Returns:
                Nothing.
                
            Creates training and testing datasets using the
            labeled directories inside the input directory.
        """
        
        print "Creating testing and training data sets..."
        
        if directory[-1] != '/':
            directory = directory + '/'
            
        data = {}
        labels = listdir(directory)
    
        # Find images and initialize face classes
        for label in labels:
            if re.match("[0-9]", label):
                self.face_classes[label] = []
                data[label] = []
                for file in listdir(directory + label):
                    if re.match("\w+", file):
                        data[label].append((directory + label + '/' + file, label))
        
        # Partition data
        for label in data:
            t1, t2 = self.partition(data[label])
            self.train_data += t1
            self.test_data += t2
            
        # Add image vectors to appropriate face class
        for path, label in self.train_data:
            vector = self.vectorize_image(path)
            self.face_classes[label].append(vector)
        
        
    def partition(self, L, prop = TRAIN_PROPORTION):
        """
            Args:
                L: the list to partition
                prop: the proportion of training data required
            Returns:
                train: a random sublist of L of size prop*len(L)
                test: a random sublist of L of size (1-prop)*len(L)
        """
        train = []
        test = []
        
        for i in range(int(len(L)*prop)):
            c = random.choice(L)
            L.remove(c)
            train.append(c)
            
        for i in L:
            test.append(i)
            
        return train, test

def main():
    """
    Command line interface
    """
    if len(sys.argv) <= 1:
        print """classifier.py face_db [-r] [-s] 
        -r: resize images (for speed)
        -s: print detailed statistics    
        """
        return
        
    stats = False
    resize = False
    db = ''
    
    for arg in sys.argv:
        if arg == '-s':
            stats = True
        elif arg == '-r':
            resize = True
        else:
            db = arg

    if db.endswith('/'):
        db = db[:-1]
    
    if not path.exists(db):
        print "Incorrect face database."
        return    
        
    classifier = PCA_Classifier(resize)
    classifier.partition_data(db)
    classifier.train()
    classifier.save_eigenface_images('efaces_' + db)
    classifier.classify(print_stats=stats)
    
if __name__ == '__main__':
    main()
    
