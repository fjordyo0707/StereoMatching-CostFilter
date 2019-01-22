import numpy as np
import cv2


def evaluate(input_path, gt_path, scale_factor, threshold=1.0):
    disp_gt = cv2.imread(gt_path, -1)
    disp_gt = np.int32(disp_gt/scale_factor)
    disp_input = cv2.imread(input_path, -1)
    disp_input = np.int32(disp_input/scale_factor)

    nr_pixel = 0
    nr_error = 0
    h, w = disp_gt.shape
    for y in range(0, h):
        for x in range(0, w):
            if disp_gt[y, x] > 0:
                nr_pixel += 1
                if np.abs(disp_gt[y, x] - disp_input[y, x]) > threshold:
                    nr_error += 1

    return float(nr_error)/nr_pixel


def main():
    print('[Bad Pixel Ratio]')
    avg = 0

    res = evaluate('./tsukuba.png', './testdata/tsukuba/disp3.pgm', scale_factor=16)
    avg += res
    print('Tsukuba: %.2f%%' % (res*100))

    res = evaluate('./venus.png', './testdata/venus/disp2.pgm', scale_factor=8)
    avg += res
    print('Venus: %.2f%%' % (res*100))

    res = evaluate('./teddy.png', './testdata/teddy/disp2.png', scale_factor=4)
    avg += res
    print('Teddy: %.2f%%' % (res*100))

    res = evaluate('./cones.png', './testdata/cones/disp2.png', scale_factor=4)
    avg += res
    print('Cones: %.2f%%' % (res*100))

    print('Average: %.2f%%' % (avg*100/4))


if __name__ == '__main__':
    main()
