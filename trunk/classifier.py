import numpy
import scipy.spatial.distance as scid

# constants for various metrics
EUCLIDEAN = 0
MANHATTAN = 1
MAHALANOBIS = 2

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

