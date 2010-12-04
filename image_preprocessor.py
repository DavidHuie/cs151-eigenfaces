import numpy
from PIL import Image
from os import listdir

ALLOWED_EXT = ['jpeg', 'gif', 'jpg']

def allow_file(filename):
    """
        Args: A filename
        Returns: True if the file specifies an image we can use;
                 False otherwise.
    """
    for ext in ALLOWED_EXT:
        if filename.endswith(ext):
            return True
    return False

def batch_image_process(directory):
    """
        Args: a directory containing images
        Returns: A list containing a vector for every image in the
                 input directory
    """
    files = listdir(directory)
    image_vectors = []

    for file in files:
        if allow_file(file):
            vector = vectorize_image(directory + '/' + file)
            image_vectors.append(vector)

    return image_vectors

def vectorize_image(file_path):
    """
        Args: The path to an image file on disk
        Returns: A one-dimensional numpy array of the image
    """
    image = Image.open(file_path)
    matrix = numpy.array(image)
    return matrix.flatten()

