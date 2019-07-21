import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def confusion_matrix_analysis(y_true, y_predicted, filename, labels, y_map=None, figure_size=(25, 25)):
    """
    Thanks to https://github.com/hitvoice
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.

    Arguments:
        y_true {array} -- true label of the data, with shape (nsamples,)
        y_predicted {array} -- prediction of the data, with shape (nsamples,)
        filename {string} -- filename of figure file to save
        labels {array} -- array, name the order of class labels in the 
                            confusion matrix. use `clf.classes_` if using
                            scikit-learn models. with shape (nclass,).

    Keyword Arguments:
        y_map {array} -- length == nclass. if not None, map the labels
                             & ys to more understandable strings.
                             (default: {None})
        figure_size {tuple} --  the size of the figure plotted. (default: {(25, 25)})
    """

    if y_map is not None:
        y_predicted = [y_map[yi] for yi in y_predicted]
        y_true = [y_map[yi] for yi in y_true]
        labels = [y_map[yi] for yi in labels]

    c_matrix = confusion_matrix(y_true, y_predicted, labels=labels)
    np.save(f'{filename}.npy', c_matrix)
    c_matrix_sum = np.sum(c_matrix, axis=1, keepdims=True)
    c_matrix_percentage = c_matrix / c_matrix_sum.astype(float) * 100
    annotations = np.empty_like(c_matrix).astype(str)
    rows, cols = c_matrix.shape
    for i in range(rows):
        for j in range(cols):
            c = c_matrix[i, j]
            p = c_matrix_percentage[i, j]
            if i == j:
                s = c_matrix_sum[i]
                annotations[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annotations[i, j] = ''
            else:
                annotations[i, j] = '%.1f%%\n%d' % (p, c)
    c_matrix = pd.DataFrame(c_matrix, index=labels, columns=labels)
    c_matrix.index.name = 'Actual'
    c_matrix.columns.name = 'Predicted'
    _, axis = plt.subplots(figsize=figure_size)
    sns.heatmap(c_matrix, annot=annotations, fmt='', ax=axis)
    plt.savefig(f'{filename}.jpg')
