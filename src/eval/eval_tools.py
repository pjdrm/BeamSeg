'''
Created on Feb 9, 2017

@author: root
'''
from segeval import window_diff
import numpy as np
from scipy.sparse import coo_matrix
from scipy.misc import comb

def segeval_converter(segmentation):
    segeval_format = []
    sent_count = 0
    for sent in segmentation:
        if sent == 1:
            segeval_format.append(sent_count+1)
            sent_count = 0
        else:
            sent_count += 1
    segeval_format.append(sent_count)
    return segeval_format

def wd(hyp_seg, ref_seg):
    hyp_seg = segeval_converter(hyp_seg)
    ref_seg = segeval_converter(ref_seg)
    wd = window_diff(hyp_seg, ref_seg)
    return float(wd)

def wd_evaluator(estimated_rho, doc):
    if doc.isMD:
        doc_begin = 0
        wd_results = []
        for doc_end in doc.docs_index:
            doc_rho = doc.rho[doc_begin:doc_end]
            doc_estimated_rho = estimated_rho[doc_begin:doc_end]
            doc_estimated_rho[-1] = 0
            wd_results.append(wd(doc_estimated_rho, doc_rho))
            doc_begin = doc_end
        return wd_results
    else:
        return [wd(estimated_rho, doc.rho)]
    
def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays"""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred

def contingency_matrix(labels_true, labels_pred, eps=None):
    """Build a contengency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate
    eps: None or float
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propogation.
        If ``None``, nothing is adjusted.
    Returns
    -------
    contingency: array, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
    """
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = coo_matrix((np.ones(class_idx.shape[0]),
                              (class_idx, cluster_idx)),
                             shape=(n_classes, n_clusters),
                             dtype=np.int).toarray()
    if eps is not None:
        # don't use += as contingency is integer
        contingency = contingency + eps
    return contingency

def comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)

def f_measure(labels_true, labels_pred): #Return the F1 score
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred)
    #print contingency
    # Compute the ARI using the contingency data
    TP_plus_FP = sum(comb2(n_c) for n_c in contingency.sum(axis=1)) # TP+FP
    
    TP_plus_FN = sum(comb2(n_k) for n_k in contingency.sum(axis=0)) #TP+FN
    
    TP = sum(comb2(n_ij) for n_ij in contingency.flatten()) #TP
    
    #print "TP = %d, TP_plus_FP = %d, TP_plus_FN = %d" %(TP,TP_plus_FP,TP_plus_FN)
    P = float(TP) / TP_plus_FP
    R = float(TP) / TP_plus_FN
    
    return 2*P*R/(P+R) 
