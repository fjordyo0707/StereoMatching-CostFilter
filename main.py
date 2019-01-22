import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from numpy.linalg import inv
import numpy.matlib
import math
from scipy.signal import medfilt2d
from scipy.signal import medfilt
from scipy.sparse import coo_matrix
from scipy.sparse import bsr_matrix

r = 9
eps = 0.0001
thresColor = 7/255
thresGrad = 2/255
gamma = 0.11
threshBorder = 3/255
gamma_c = 0.1
gamma_d = 9
r_median = 19


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)




    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    # Convert to grayscale
    Il_g = matlabBGR2gray(Il)/255.0
    Ir_g = matlabBGR2gray(Ir)/255.0
    Il = Il/255.0
    Ir = Ir/255.0
    
    # Mirror image
    Il_1 = (np.fliplr(Ir)).copy()
    Ir_1 = (np.fliplr(Il)).copy()

    # Compute gradient in X-direction form grayscale images
    fx_l = np.gradient(Il_g, axis=1)
    fx_r = np.gradient(Ir_g, axis=1)
    fx_l = fx_l+0.5
    fx_r = fx_r+0.5
    fx_l_1 = (np.fliplr(fx_r)).copy()
    fx_r_1 = (np.fliplr(fx_l)).copy()

    dispVol = np.ones((h,w,max_disp))*threshBorder
    dispVol1 = np.ones((h,w,max_disp))*threshBorder

    for d in range(1,max_disp+1):
        # Right to left
        tmp = np.ones((h,w,ch))*threshBorder
        tmp[:,d:w,:] = Ir[:,:w-d,:]
        p_color = abs(tmp - Il)
        p_color = np.mean(p_color, axis = 2)
        p_color = np.minimum(p_color, thresColor)

        tmp = np.ones((h,w))*threshBorder
        tmp[:,d:w] = fx_r[:,:w-d]
        p_grad = abs(tmp - fx_l)
        p_grad = np.minimum(p_grad, thresGrad)

        p = gamma*p_color+(1-gamma)*p_grad

        # Left to Right
        tmp1 = np.ones((h,w,ch))*threshBorder
        tmp1[:,d:w,:] = Ir_1[:,:w-d,:]
        p1_color = abs(tmp1 - Il_1)
        p1_color = np.mean(p1_color, axis = 2)
        p1_color = np.minimum(p1_color, thresColor)

        tmp1 = np.ones((h,w))*threshBorder
        tmp1[:,d:w] = fx_r_1[:,:w-d]
        p1_grad = abs(tmp1 - fx_l_1)
        p1_grad = np.minimum(p1_grad, thresGrad)

        p1 = gamma*p1_color+(1-gamma)*p1_grad

        dispVol[:,:,d-1] = p
        dispVol1[:,:,d-1] = (np.fliplr(p1)).copy()
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))



    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    for d in range(max_disp):
        p = dispVol[:,:,d]
        p1 = dispVol1[:,:,d]

        q = guidedfilter_color(Il, np.double(p), r, eps)
        p1 = (np.fliplr(p1)).copy()
        q1 = guidedfilter_color(Il_1, np.double(p1), r, eps)
        dispVol[:,:,d] = q
        dispVol1[:,:,d] = (np.fliplr(q1)).copy()
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))





    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    labels_left = (np.argmin(dispVol, axis = 2) + np.ones((h,w)) ).astype(int)
    labels_right = (np.argmin(dispVol1, axis = 2) + np.ones((h,w)) ).astype(int)

    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))







    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map

    Y = np.matlib.repmat(np.array(range(h)).reshape(-1,1), 1, w)
    X = np.matlib.repmat(np.array(range(w)), h,1 ) - labels_left
    X[X<0] = 0
    labelstmp = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            labelstmp[i,j] = labels_right[i,X[i,j]]
    final_labels = labels_left
    final_labels[abs(labels_left - labelstmp)>=1] = -1
    
    inputLabels = final_labels.copy()
    final_labels = fillPixelaReference(Il, inputLabels, max_disp)    
    
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return final_labels

def guidedfilter_color(I, p, r, eps):
    hei, wid = p.shape
    N = boxFilter(np.ones((hei, wid)), r)

    mean_I_r = boxFilter(I[:,:,0], r)/N
    mean_I_g = boxFilter(I[:,:,1], r)/N
    mean_I_b = boxFilter(I[:,:,2], r)/N

    mean_p = boxFilter(p, r)/N

    mean_Ip_r = boxFilter(np.multiply(I[:,:,0], p),r)/N
    mean_Ip_g = boxFilter(np.multiply(I[:,:,1], p),r)/N
    mean_Ip_b = boxFilter(np.multiply(I[:,:,2], p),r)/N

    cov_Ip_r = mean_Ip_r - np.multiply(mean_I_r, mean_p);
    cov_Ip_g = mean_Ip_g - np.multiply(mean_I_g, mean_p);
    cov_Ip_b = mean_Ip_b - np.multiply(mean_I_b, mean_p);

    var_I_rr = boxFilter(np.multiply(I[:,:,0],I[:,:,0]), r)/N - np.multiply(mean_I_r,mean_I_r)
    var_I_rg = boxFilter(np.multiply(I[:,:,0],I[:,:,1]), r)/N - np.multiply(mean_I_r,mean_I_g)
    var_I_rb = boxFilter(np.multiply(I[:,:,0],I[:,:,2]), r)/N - np.multiply(mean_I_r,mean_I_b)
    var_I_gg = boxFilter(np.multiply(I[:,:,1],I[:,:,1]), r)/N - np.multiply(mean_I_g,mean_I_g)
    var_I_gb = boxFilter(np.multiply(I[:,:,1],I[:,:,2]), r)/N - np.multiply(mean_I_g,mean_I_b)
    var_I_bb = boxFilter(np.multiply(I[:,:,2],I[:,:,2]), r)/N - np.multiply(mean_I_b,mean_I_b)

    a = np.zeros((hei, wid, 3))

    for i in range(hei):
        for j in range(wid):
            Sigma = np.array([ [var_I_rr[i,j], var_I_rg[i,j], var_I_rb[i,j]] ,
                [ var_I_rg[i,j], var_I_gg[i,j], var_I_gb[i,j] ],
                [ var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j]] ])

            cov_Ip = [cov_Ip_r[i,j], cov_Ip_g[i,j], cov_Ip_b[i,j]]

            a[i,j,:] = np.dot(cov_Ip,inv(Sigma + eps*np.eye(3)))
    b = mean_p - np.multiply(a[:,:,0], mean_I_r) - np.multiply(a[:,:,1], mean_I_g) - np.multiply(a[:,:,2], mean_I_b)

    q = (np.multiply(boxFilter(a[:,:,0], r), I[:,:,0]) + 
    np.multiply(boxFilter(a[:,:,1], r), I[:,:,1]) +
    np.multiply(boxFilter(a[:,:,2], r), I[:,:,2])+
    boxFilter(b,r))/N
    return q
def boxFilter(imSrc, r):
    h, w = imSrc.shape
    imDst = np.zeros((h, w))
    imCum = np.cumsum(imSrc, axis = 0)
    imDst[: r+1, :] = imCum[r: 2*r+1, :]
    imDst[r+1 : h - r, :] = imCum[2*r + 1: h, :] - imCum[: h - 2*r - 1, :]
    imDst[h - r : h, :] = np.matlib.repmat(imCum[h-1, :], r, 1) - imCum[h - 2*r -1 : h - r -1 , :]
    imCum = np.cumsum(imDst, axis = 1)
    imDst[:, : r+1] = imCum[:, r: 2*r+1]
    imDst[:, r+1 : w - r] = imCum[:, 2*r + 1: w] - imCum[:, : w - 2*r - 1]
    imDst[:, w - r : w] = np.matlib.repmat(imCum[:, w-1].reshape(-1,1), 1, r) - imCum[:, w - 2*r -1 : w - r -1]
    return imDst

def fillPixelaReference(Il, final_labels, max_disp):
    h,w = final_labels.shape
    occPix = np.zeros((h,w))
    occPix[final_labels<0] = 1
    
    fillVals = np.ones((h)) * max_disp
    final_labels_filled = final_labels.copy()

    for col in range(w):
        curCol = final_labels[:,col].copy()
        curCol[curCol==-1] = fillVals[curCol==-1]
        fillVals[curCol!=-1] = curCol[curCol!=-1]
        final_labels_filled[:,col] = curCol
    
    fillVals = np.ones((h)) * max_disp
    final_labels_filled1 = final_labels.copy()
    for col in reversed(range(w)):
        curCol = final_labels[:,col].copy()
        curCol[curCol==-1] = fillVals[curCol==-1]
        fillVals[curCol!=-1] = curCol[curCol!=-1]
        final_labels_filled1[:,col] = curCol

    final_labels = np.fmin(final_labels_filled, final_labels_filled1)

    final_labels_smoothed = weightedMedianMatlab(Il, final_labels.copy(), r_median)
    final_labels[occPix==1] = final_labels_smoothed[occPix==1]


    return final_labels

def weightedMedianMatlab(left_img, disp_img, winsize):
    h, w, c = left_img.shape

    smoothed_left_img = np.zeros((h,w,c))
    smoothed_left_img[:,:,0] = medfilt(left_img[:,:,0],3)
    smoothed_left_img[:,:,1] = medfilt(left_img[:,:,1],3)
    smoothed_left_img[:,:,2] = medfilt(left_img[:,:,2],3)

    radius = math.floor(winsize/2.0)

    medianFiltered = np.zeros((h,w))

    for y in range(h):
        for x in range(w):
            maskVals = np.double(filtermask(smoothed_left_img, x, y, winsize))
            dispVals = disp_img[max(0,y-radius):min(h,y+radius),max(0,x-radius):min(w,x+radius)].copy()
            maxDispVal = int(np.amax(dispVals))

            dispVals_f = (dispVals.copy() - 1).astype(np.int).flatten()
            maskVals_f = (maskVals.flatten()).astype(np.double)
            zeros_f = np.zeros(dispVals.shape).astype(np.int).flatten()

            hist = coo_matrix((maskVals_f,(zeros_f,dispVals_f)), shape = (1,maxDispVal)).toarray()
            hist_sum = np.sum(hist)
            hist_cumsum = np.cumsum(hist)

            possbileDispVals = np.arange(1,maxDispVal+1)
            medianval = possbileDispVals[hist_cumsum>(hist_sum/2.0)]
            medianFiltered[y,x] = medianval[0]

    return medianFiltered       

def filtermask(colimg, x, y, winsize):
    radius = math.floor(winsize/2.0)
    h, w, c = colimg.shape

    patch_h = len( np.arange(max(0,y-radius),min(h,y+radius)) ) 
    patch_w = len( np.arange(max(0,x-radius),min(w,x+radius)) )

    centercol = colimg[y,x,:]
    centerVol = np.zeros((patch_h,patch_w,3))
    centerVol[:,:,0] = centercol[0]
    centerVol[:,:,1] = centercol[1]
    centerVol[:,:,2] = centercol[2]

    Yinds = np.arange(max(0,y-radius),min(h,y+radius), dtype=int)
    patchYinds = np.matlib.repmat( Yinds.reshape(-1,1), 1, patch_w)
    Xinds = np.arange(max(0,x-radius),min(w,x+radius), dtype=int)
    patchXinds = np.matlib.repmat( Xinds, patch_h, 1)

    curPatch = colimg[Yinds[0]:Yinds[-1]+1,Xinds[0]:Xinds[-1]+1,:]
    coldiff = np.sqrt(np.sum( np.square(centerVol - curPatch), axis = 2))

    x_patch = np.ones((patch_h,patch_w))*x
    y_patch = np.ones((patch_h,patch_w))*y
    sdiff = np.sqrt( np.square(x_patch - patchXinds) + np.square(y_patch - patchYinds))

    weights = np.exp(-1* coldiff/(gamma_c*gamma_c)) * np.exp(-1* sdiff/(gamma_d*gamma_d))

    return weights

def matlabBGR2gray(img):
    h, w, c = img.shape
    ans = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.2989
    return ans


def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))
    
    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))

    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))
    

if __name__ == '__main__':
    main()

