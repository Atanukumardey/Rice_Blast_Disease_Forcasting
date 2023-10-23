import os
import glob
import sys
import configparser
import random
import cv2 as cv
import numpy as np
from keras.preprocessing.image import img_to_array, ImageDataGenerator


# cv.namedWindow("image", cv.WINDOW_NORMAL)
# cv.resizeWindow("image", 800, 500)

# np_oldopt = np.get_printoptions()
# np.set_printoptions(threshold=np.inf)

#
# file and folder names
#
config = configparser.ConfigParser()
config.read('codeconfig.ini')

infolder = config['path']['infolder']
outfolder = config['path']['outfolder']
img_out = config['path']['img_out']

extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

imgoutfolder = os.path.join(outfolder + img_out.split("/")[0])
if (not os.path.exists(imgoutfolder)):
    os.makedirs(imgoutfolder)

inpath = glob.glob(infolder + "*.*")

n_classes = int(config['classinfo']['n_classes'])
clsnames = str(config['classinfo']['target_names']).encode('ascii', 'ignore')

target_names = []
labels = []

for i in range(n_classes):
    target_names.append(clsnames.split(', ')[i])
    labels.append(float(i + 1))

filecount = 0

# flags
preproc_f = 0
rembg_f = 0
seg_f = 0
crop_f = 0
aug_f = 1
only_aug = 1


#
# MAIN CODE BEGINS
#

if(not only_aug):
    for f in inpath:
        name, ext = os.path.splitext(os.path.basename(f))

        if (str(ext).lower() not in extensions):
            continue

        file_number = (name.split('(', 1)[1]).split(')', 1)[0]
        filecount += 1

        print ("\n\t {}. processing {}").format(filecount, name)

        found = 0
        for i in range(n_classes):
            if (name.startswith(str(target_names[i]).lower())):
                class_number = labels[i]
                found = 1

        if (not found):
            print ("Wrong class - {}").format(class_number)
            sys.exit(0)

        #
        # read image
        #
        img = im1 = cv.imread(f)

        # image processing
        if(preproc_f):
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            img_rgb_chanel = cv.split(img)
            img_hsv_chanel = cv.split(img_hsv)

        # remove background
        if(rembg_f):
            thresh_hsv_s = 0.28 * 255

            ret_s, hsv_s_thresh = cv.threshold(
                img_hsv_chanel[1], thresh_hsv_s, 255, cv.THRESH_BINARY)
            hsv_s_thresh = cv.GaussianBlur(hsv_s_thresh, (5, 5), 0)
            rembg = cv.bitwise_and(img, img, mask=hsv_s_thresh)

            # cv.imwrite(outfolder + img_out + name + "-rembg" + ".png", rembg)

        # TODO write comments

        # disease segmentation
        if (seg_f):
            hue = img_hsv_chanel[0]
            thresh_hsv_h = 22

            _, k = cv.threshold(hue, thresh_hsv_h, 255, cv.THRESH_BINARY)

            k = cv.medianBlur(k, 15)
            k_inv = cv.bitwise_not(k)

            seg = cv.bitwise_and(rembg, rembg, mask=k_inv)
            seg = cv.medianBlur(seg, 7)
            # cv.imwrite(outfolder + img_out + name + "-seg" + ".png", seg)

            seg_gray = cv.cvtColor(seg, cv.COLOR_RGB2GRAY)
            _, seg_bin = cv.threshold(seg_gray, 40, 255, cv.THRESH_BINARY)

            seg_bin = cv.copyMakeBorder(seg_bin, 10, 10, 10, 10,
                                        cv.BORDER_CONSTANT, (0, 0, 0))

            # cv.imwrite(outfolder + img_out + name + "-segbin" + ".png", seg_bin)

            new_segmask = np.zeros(seg_bin.shape, np.uint8)

            seg_canny = cv.Canny(seg_bin, 100, 200)

            contours, hierarchy = cv.findContours(
                seg_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if(len(contours)):
                sorted_cnts = sorted(
                    contours, key=cv.contourArea, reverse=True)
                cnt_max = cv.contourArea(sorted_cnts[0])

                for c, e in enumerate(contours):
                    area = cv.contourArea(contours[c])
                    if (area >= cnt_max*0.1):
                        cv.drawContours(new_segmask, [contours[c]], 0, 255, -1)

            rb = rembg.copy()
            rb = cv.copyMakeBorder(rb, 10, 10, 10, 10,
                                   cv.BORDER_CONSTANT, (0, 0, 0))
            seg = cv.bitwise_and(rb, rb, mask=new_segmask)
            # cv.imwrite(outfolder + img_out + name + "-newseg" + ".png", seg)

        # cropping images
        if (crop_f):
            # points are col x row
            # images are row x col

            seg_bin = new_segmask
            segline = seg.copy()

            row_y = seg_bin.shape[0]
            col_x = seg_bin.shape[1]

            left = seg_bin.shape[1]
            p_left1 = (0, 0)
            p_left2 = (0, 0)

            for y in range(0, row_y):
                for x in range(0, col_x):
                    if (seg_bin[y][x]):
                        if (x <= left):
                            left = x

            p_left1 = (left-2, 0)
            p_left2 = (left-2, col_x-2)
            # left_line = cv.line(segline, p_left1, p_left2, (255, 0, 0), 1)

            # print("Left = {} to {}").format(p_left1, p_left2)
            # cv.imwrite(outfolder + img_out + name + "-cropbox" + ".png", left_line)

            top = seg_bin.shape[0]
            p_top1 = (0, 0)
            p_top2 = (0, 0)

            for x in range(col_x-1, 0, -1):
                for y in range(row_y-1, 0, -1):
                    if (seg_bin[y][x]):
                        if (y <= top):
                            top = y

            p_top1 = (0, top-2)
            p_top2 = (col_x-2, top-2)
            # top_line = cv.line(segline, p_top1, p_top2, (255, 0, 0), 1)

            # print("Top = {} to {}").format(p_top1, p_top2)
            # cv.imwrite(outfolder + img_out + name + "-cropbox" + ".png", top_line)

            right = 0
            p_right1 = (0, 0)
            p_right2 = (0, 0)

            for y in range(row_y-1, 0, -1):
                for x in range(col_x-1, 0, -1):
                    if (seg_bin[y][x]):
                        if (x >= right):
                            right = x

            p_right1 = (right+2, 0)
            p_right2 = (right+2, col_x+2)
            # right_line = cv.line(segline, p_right1, p_right2, (255, 0, 0), 1)

            # print("Right = {} to {}").format(p_right1, p_right2)
            # cv.imwrite(outfolder + img_out + name + "-cropbox" + ".png", right_line)

            bottom = 0
            p_bottom1 = (0, 0)
            p_bottom2 = (0, 0)

            for x in range(0, col_x):
                for y in range(0, row_y):
                    if (seg_bin[y][x]):
                        if (y >= bottom):
                            bottom = y

            p_bottom1 = (0, bottom+2)
            p_bottom2 = (col_x+2, bottom+2)
            # bottom_line = cv.line(segline, p_bottom1, p_bottom2, (255, 0, 0), 1)

            # print("Bottom = {} to {}").format(p_bottom1, p_bottom2)
            # cv.imwrite(outfolder + img_out + name + "-croplines" + ".png", bottom_line)

            # intersection of lines
            x_tl = (p_left1[0], p_top1[1])
            x_tr = (p_right1[0], p_top2[1])
            x_bl = (p_left2[0], p_bottom1[1])
            x_br = (p_right2[0], p_bottom2[1])

            # test the intersections
            # bottom_line[x_bl[1]][x_bl[0]] = (255, 255, 255)
            # cv.imwrite(outfolder + img_out + name + "-cross" + ".png", bottom_line)

            # print("Top-left point = {} \nBottom-right point = {}").format(x_tl, x_br)
            # cropbox = cv.rectangle(seg, x_tl, x_br, (0,0,0), 1)
            # cv.imwrite(outfolder + img_out + name + "-cropbox" + ".png", cropbox)

            cropped = seg.copy()
            cropped = cropped[x_tl[1]:x_br[1], x_tl[0]:x_br[0]]
            cropped = cv.copyMakeBorder(cropped, 10, 10, 10, 10,
                                        cv.BORDER_CONSTANT, (0, 0, 0))
            # cv.imwrite(outfolder + img_out + name + "-cropped" + ".png", cropped)
            cv.imwrite(outfolder + img_out + name + ".png", cropped)


# image augmentation
if(aug_f):
    infolder = imgoutfolder
    inpath = glob.glob(infolder + "/" + "*.*")

    data_aug = os.path.join(outfolder + "aug/")
    if (not os.path.exists(data_aug)):
        os.makedirs(data_aug)

    def motion_blur(img, angle, deg):
        rotation = cv.getRotationMatrix2D((deg / 2, deg / 2), angle, 1)

        kernel = np.diag(np.ones(deg))
        kernel = (cv.warpAffine(kernel, rotation, (deg, deg))) / deg

        res = cv.filter2D(img, -1, kernel)
        cv.normalize(res, res, 0, 255, cv.NORM_MINMAX)
        res = np.array(res, dtype=np.uint8)
        return res

    for f in inpath:
        name, ext = os.path.splitext(os.path.basename(f))

        print("\n\tAugmenting {}").format(name)

        img_orig = cv.imread(f)
        img = cv.resize(img_orig, (512, 512), interpolation=cv.INTER_AREA)
        data = img_to_array(img)
        samples = np.expand_dims(data, axis=0)

        aug = ImageDataGenerator(
            horizontal_flip=True,
            rotation_range=90, shear_range=0.15,
            width_shift_range=0.2, height_shift_range=0.2,
            zoom_range=[0.5, 2], brightness_range=[0.25, 1.5],
            fill_mode="nearest")

        gen = aug.flow(samples)

        for i in range(25):
            im = gen.next()
            im = im.reshape((512, 512, 3))

            # motion blur randomly, about 30%  data
            rand = random.randint(0, 2)
            if (not rand):
                im = motion_blur(im, 50, 16)

            cv.imwrite(data_aug + name + "#" + str(i) + ".png", im)


print "COMPLETED"
