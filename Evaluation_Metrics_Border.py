import torch
from optparse import OptionParser
from skimage.measure import label
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import jaccard_similarity_score, f1_score
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from skimage.morphology import erosion, disk
from os.path import join
import os
from skimage.io import imsave, imread
import numpy as np
import pdb
import time
from postprocessing import PostProcess
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, watershed, remove_small_holes
from skimage import measure
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max


def my_watershed(what, mask1, mask2):
    # markers = ndi.label(mask2, output=np.uint32)[0]
    # big_seeds = watershed(what, markers, mask=mask1, watershed_line=False)
    # m2 = mask1 - (big_seeds > 0)
    # mask2 = mask2 | m2

    markers = ndi.label(mask2, output=np.uint32)[0]
    labels = watershed(what, markers, mask=mask1, watershed_line=True)
    # labels = watershed(what, markers, mask=mask1, watershed_line=False)
    return labels

def wsh(mask_img, threshold, border_img):
    img_copy = np.copy(mask_img)
    m = mask_img * (1-border_img)
    # m = mask_img-border_img
    # m[m < 0] = 0
    # m[m > 0] = 1

    img_copy[m <= threshold+0.0] = 0
    img_copy[m > threshold+0.0] = 1
    img_copy = img_copy.astype(np.bool)
    # img_copy = remove_small_objects(img_copy, 10).astype(np.uint8)

    mask_img[mask_img <= threshold] = 0
    mask_img[mask_img > threshold] = 1
    mask_img = mask_img.astype(np.bool)
    # mask_img = remove_small_holes(mask_img, 500)
    # mask_img = remove_small_objects(mask_img, 10).astype(np.uint8)
    # cv2.imwrite('t.png', (mask_img * 255).astype(np.uint8))
    # cv2.imwrite('t2.png', (img_copy * 255).astype(np.uint8))
    labeled_array = my_watershed(mask_img, mask_img, img_copy)
    return labeled_array

def postprocess_victor(pred):
    av_pred = pred
    av_pred = av_pred[0,:,:] * (1 - av_pred[1,:,:])
    av_pred = 1 * (av_pred > 0.5)
    av_pred = av_pred.astype(np.uint8)

    y_pred = measure.label(av_pred, neighbors=8, background=0)
    props = measure.regionprops(y_pred)
    for i in range(len(props)):
        if props[i].area < 12:
            y_pred[y_pred == i + 1] = 0
    y_pred = measure.label(y_pred, neighbors=8, background=0)

    nucl_msk = (1 - pred[0,:,:])
    nucl_msk = nucl_msk.astype('uint8')
    y_pred = watershed(nucl_msk, y_pred, mask=((pred[0,:,:] > 0.5)), watershed_line=True)
    return y_pred

def watershed_hyy(prob, threshold=0.5):

    prob[prob <= threshold]=0
    prob[prob > threshold]=1
    distance = ndi.distance_transform_edt(prob)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((15, 15)),
                                labels=prob)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=prob)

    return labels


def ComputeMetrics_Batch(prob, batch_labels, inputs_ori=None, save_dir=None, img_name=None):
    batch_size = prob.shape[0]
    sum_acc=0
    sum_roc=0
    sum_jac=0
    sum_recall=0
    sum_precision=0
    sum_f1=0
    sum_aji=0
    for i in range(0, batch_size):
        tmp_prob = prob[i, :, :, :].squeeze()
        tmp_batch_labels = batch_labels[i, :, :, :].squeeze()
        if inputs_ori is not None:
            tmp_inputs_ori = inputs_ori[i, :, :, :]
            tmp_img_name = img_name[i]
            tmp_save_path = join(save_dir, 'Pred_'+tmp_img_name)
            acc, roc, jac, recall, precision, f1, aji = ComputeMetrics(tmp_prob, tmp_batch_labels, 35, 0.5, inputs_ori=tmp_inputs_ori, save_path=tmp_save_path)
        else:
            acc, roc, jac, recall, precision, f1, aji = ComputeMetrics(tmp_prob, tmp_batch_labels, 35, 0.5)

        sum_acc += acc
        sum_roc += roc
        sum_jac += jac
        sum_recall += recall
        sum_precision += precision
        sum_f1 += f1
        sum_aji += aji

    # return sum_acc/batch_size, sum_roc/batch_size, sum_jac/batch_size, sum_recall/batch_size, sum_precision/batch_size, sum_f1/batch_size, sum_aji/batch_size
    return sum_acc, sum_roc, sum_jac, sum_recall, sum_precision, sum_f1, sum_aji

# SR : Segmentation Result
# GT : Ground Truth
def ComputeMetrics(prob, batch_labels, p1, p2, inputs_ori=None, save_path=None):
    """
    Computes all metrics between probability map and corresponding label.
    If you give also an rgb image it will save many extra meta data image.
    """


    GT = label(batch_labels.copy())
    ### border
    # PRED = wsh(prob[0,:,:], 0.5, prob[1,:,:])
    # PRED = postprocess_victor(prob)


    ### no border
    # prob = prob[0, :, :] * (1 - prob[1, :, :])
    prob = prob[0, :, :]  - prob[1, :, :]
    prob[prob < 0] = 0
    prob[prob > 1] = 1
    #
    # PRED = watershed_hyy(prob, threshold=0.3)
    PRED = PostProcess(prob, p1, p2)

    # tmp_prob = np.copy(prob)
    # prob[tmp_prob <= 0.6] = 0
    # prob[tmp_prob > 0.6] = 1
    # prob = prob.astype('int64')
    # PRED = label(prob)


    lbl = GT.copy()
    pred = PRED.copy()
    aji = AJI_fast(lbl, pred)
    # aji = AJI(batch_labels, prob)

    lbl[lbl > 0] = 1
    pred[pred > 0] = 1

    l, p = lbl.flatten(), pred.flatten()
    acc = accuracy_score(l, p)
    roc = roc_auc_score(l, p)
    jac = jaccard_similarity_score(l, p)
    f1 = f1_score(l, p)
    recall = recall_score(l, p)
    precision = precision_score(l, p)
    # if rgb is not None:
    #     xval_n = join(save_path, "xval_{}.png").format(ind)
    #     yval_n = join(save_path, "yval_{}.png").format(ind)
    #     prob_n = join(save_path, "prob_{}.png").format(ind)
    #     pred_n = join(save_path, "pred_{}.png").format(ind)
    #     c_gt_n = join(save_path, "C_gt_{}.png").format(ind)
    #     c_pr_n = join(save_path, "C_pr_{}.png").format(ind)
    #
    #     imsave(xval_n, rgb)
    #     imsave(yval_n, color_bin(GT))
    #     imsave(prob_n, prob)
    #     imsave(pred_n, color_bin(PRED))
    #     imsave(c_gt_n, add_contours(rgb, GT))
    #     imsave(c_pr_n, add_contours(rgb, PRED))
    if inputs_ori is not None:
        imsave(save_path, add_contours(inputs_ori, PRED))

    return acc, roc, jac, recall, precision, f1, aji


def color_bin(bin_labl):
    """
    Colors bin image so that nuclei come out nicer.
    """
    dim = bin_labl.shape
    x, y = dim[0], dim[1]
    res = np.zeros(shape=(x, y, 3))
    for i in range(1, bin_labl.max() + 1):
        rgb = np.random.normal(loc=125, scale=100, size=3)
        rgb[rgb < 0] = 0
        rgb[rgb > 255] = 255
        rgb = rgb.astype(np.uint8)
        res[bin_labl == i] = rgb
    return res.astype(np.uint8)


def add_contours(rgb_image, contour, color=[0,128,0]):
    """
    Adds contours to images.
    The image has to be a binary image
    """
    rgb = rgb_image.copy()
    contour[contour > 0] = 1
    boundary = contour - erosion(contour, disk(2))
    rgb[boundary > 0] = color
    return rgb


def CheckOrCreate(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def Intersection(A, B):
    """
    Returns the pixel count corresponding to the intersection
    between A and B.
    """
    C = A + B
    C[C != 2] = 0
    C[C == 2] = 1
    return C


def Union(A, B):
    """
    Returns the pixel count corresponding to the union
    between A and B.
    """
    C = A + B
    C[C > 0] = 1
    return C


def AssociatedCell(G_i, S):
    """
    Returns the indice of the associated prediction cell for a certain
    ground truth element. Maybe do something if no associated cell in the
    prediction mask touches the GT.
    """

    def g(indice):
        S_indice = np.zeros_like(S)
        S_indice[S == indice] = 1
        NUM = float(Intersection(G_i, S_indice).sum())
        DEN = float(Union(G_i, S_indice).sum())
        return NUM / DEN

    res = list(map(g, range(1, S.max() + 1)))
    indice = np.array(res).argmax() + 1
    return indice


def AJI(G, S):
    """
    AJI as described in the paper, AJI is more abstract implementation but 100times faster.
    """
    G = label(G, background=0)
    S = label(S, background=0)

    C = 0
    U = 0
    USED = np.zeros(S.max())

    for i in range(1, G.max() + 1):
        only_ground_truth = np.zeros_like(G)
        only_ground_truth[G == i] = 1
        j = AssociatedCell(only_ground_truth, S)
        only_prediction = np.zeros_like(S)
        only_prediction[S == j] = 1
        C += Intersection(only_prediction, only_ground_truth).sum()
        U += Union(only_prediction, only_ground_truth).sum()
        USED[j - 1] = 1

    def h(indice):
        if USED[indice - 1] == 1:
            return 0
        else:
            only_prediction = np.zeros_like(S)
            only_prediction[S == indice] = 1
            return only_prediction.sum()

    # U_sum = map(h, range(1, S.max() + 1))
    U_sum = list(map(h, range(1, S.max() + 1)))

    U += np.sum(U_sum)
    return float(C) / float(U)


def AJI_fast(G, S):
    """
    AJI as described in the paper, but a much faster implementation.
    """
    G = label(G, background=0)
    S = label(S, background=0)
    if S.sum() == 0:
        return 0.
    C = 0
    U = 0
    USED = np.zeros(S.max())

    G_flat = G.flatten()
    S_flat = S.flatten()
    G_max = np.max(G_flat)
    S_max = np.max(S_flat)
    m_labels = max(G_max, S_max) + 1
    cm = confusion_matrix(G_flat, S_flat, labels=range(m_labels)).astype(np.float)
    LIGNE_J = np.zeros(S_max)
    for j in range(1, S_max + 1):
        LIGNE_J[j - 1] = cm[:, j].sum()

    for i in range(1, G_max + 1):
        LIGNE_I_sum = cm[i, :].sum()

        def h(indice):
            LIGNE_J_sum = LIGNE_J[indice - 1]
            inter = cm[i, indice]

            union = LIGNE_I_sum + LIGNE_J_sum - inter
            return inter / union

        JI_ligne = list(map(h, range(1, S_max + 1)))
        best_indice = np.argmax(JI_ligne) + 1
        C += cm[i, best_indice]
        U += LIGNE_J[best_indice - 1] + LIGNE_I_sum - cm[i, best_indice]
        USED[best_indice - 1] = 1

    U_sum = ((1 - USED) * LIGNE_J).sum()
    U += U_sum
    return float(C) / float(U)


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC