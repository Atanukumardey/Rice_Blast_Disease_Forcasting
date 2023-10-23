import os
import glob
import sys
import shutil
import cPickle
import configparser
import time
import datetime as dt
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern


# timing and timestamping functions
genesis = time.time()  # timing
timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# cv.namedWindow("image", cv.WINDOW_NORMAL)
# cv.resizeWindow("image", 800, 500)

#np_oldopt = np.get_printoptions()
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

fextout = os.path.join(outfolder + 'FEXT-' + timestamp)
os.makedirs(fextout)

feat_list = open(fextout + '/' + 'Feature_List-' + timestamp + '.txt', 'w')
colout = open(fextout + '/' + 'FL-' + timestamp + '-feature columns.txt', 'w')

FL = []
flname = []
flname_on = 1
csv_idx = []
csv_idx_on = 1
filecount = 0

n_classes = int(config['classinfo']['n_classes'])
clsnames = str(config['classinfo']['target_names']).encode('ascii', 'ignore')

target_names = []
labels = []

for i in range(n_classes):
    target_names.append(clsnames.split(', ')[i])
    labels.append(float(i + 1))

#
# flags
#
resize_f = 0

color_f = 1
shape_f = 1
texture_f = 1

mb_f = 1
mg_f = 1
mr_f = 1
mh_f = 1
ms_f = 1
mv_f = 0
sdb_f = 1
sdg_f = 1
sdr_f = 1
bghist_f = (16 * 16)
hshist_f = (16 * 16)  # bin * bin
skw_f = 1
kurt_f = 1
sdseg_f = 1

cntperi_f = 1
cntarea_f = 1
hum_f = 7  # 7 moments

glcm_f = 4 * 1  # mean of 4 glcm, 4 angle
lbphist_f = 32  # 32  # bins

# flags finished

# calculate feature number for feature list shaping
prid_f_num = 1
class_f_num = 1

if (color_f):
    color_f_num = mb_f + mg_f + mr_f + mh_f + ms_f + mv_f + \
        sdb_f + sdg_f + sdr_f + bghist_f + hshist_f + skw_f + kurt_f + sdseg_f
else:
    color_f_num = 0

if (shape_f):
    shape_f_num = cntperi_f + cntarea_f + hum_f
else:
    shape_f_num = 0

if (texture_f):
    texture_f_num = glcm_f + lbphist_f
else:
    texture_f_num = 0

feat_num = prid_f_num + class_f_num + color_f_num + \
    shape_f_num + texture_f_num

#
# MAIN CODE BEGINS
#

print "\n Beginning feature extraction... \n"


fext_gen = time.time()

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

    # processing_id = classnum(1) + file_num
    processing_id = int(str(int(class_number)) + str(file_number))
    FL.append(processing_id)

    if flname_on:
        flname.append("Processing ID")
    if csv_idx_on:
        csv_idx.append("Processing ID")

    FL.append(class_number)

    if flname_on:
        flname.append("Class Label")
    if csv_idx_on:
        csv_idx.append("Class Label")

    #
    # read images
    #
    img = im1 = cv.imread(f)

    if (resize_f):
        res_row = min(512, img.shape[1])
        res_col = min(256, img.shape[0])
        # res_row = res_col = 512
        img = cv.resize(im1, (res_row, res_col),
                        interpolation=cv.INTER_AREA)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    img_rgb_chanel = cv.split(img)
    img_hsv_chanel = cv.split(img_hsv)

    # feature extraction
    seg = img
    segf = seg.flatten()
    seg_gray = cv.cvtColor(seg, cv.COLOR_RGB2GRAY)

    if (color_f):
        print "Extracting color features..."

        b_seg, g_seg, r_seg = cv.split(seg)

        b_segf = b_seg.flatten()
        g_segf = g_seg.flatten()
        r_segf = r_seg.flatten()
        mean_r_seg = mean_g_seg = mean_b_seg = 0

        if (mb_f):
            if flname_on:
                flname.append("Mean of RGB-Blue of segmented image")
            if csv_idx_on:
                csv_idx.append("Mean of RGB-Blue")

            if np.count_nonzero(b_segf) != 0:
                mean_b_seg = sum(b_segf[np.nonzero(b_segf)]) / \
                    np.count_nonzero(b_segf)  # FEX mean_b_seg
                FL.append(mean_b_seg)
            else:
                FL.append(0.0)

        if (mg_f):
            if flname_on:
                flname.append("Mean of RGB-Green of segmented image")
            if csv_idx_on:
                csv_idx.append("Mean of RGB-Green")

            if np.count_nonzero(g_segf) != 0:
                mean_g_seg = sum(g_segf[np.nonzero(g_segf)]) / \
                    np.count_nonzero(g_segf)  # FEX mean_g_seg
                FL.append(mean_g_seg)
            else:
                FL.append(0.0)

        if (mr_f):
            if flname_on:
                flname.append("Mean of RGB-Red of segmented image")
            if csv_idx_on:
                csv_idx.append("Mean of RGB-Red")

            if np.count_nonzero(r_segf) != 0:
                mean_r_seg = sum(r_segf[np.nonzero(r_segf)]) / \
                    np.count_nonzero(r_segf)  # FEX mean_r_seg
                FL.append(mean_r_seg)
            else:
                FL.append(0.0)

        seg_hsv = cv.cvtColor(seg, cv.COLOR_RGB2HSV)
        h_seg, s_seg, v_seg = cv.split(seg_hsv)

        h_segf = h_seg.flatten()
        s_segf = s_seg.flatten()
        v_segf = v_seg.flatten()
        mean_h_seg = mean_s_seg = mean_v_seg = 0

        if (mh_f):
            if flname_on:
                flname.append("Mean of HSV-Hue of segmented image")
            if csv_idx_on:
                csv_idx.append("Mean of HSV-Hue")

            if np.count_nonzero(h_segf) != 0:
                mean_h_seg = sum(h_segf[np.nonzero(h_segf)]) / \
                    np.count_nonzero(h_segf)  # FEX mean_h_seg
                FL.append(mean_h_seg)
            else:
                FL.append(0.0)

        if (ms_f):
            if flname_on:
                flname.append("Mean of HSV-Saturation of segmented image")
            if csv_idx_on:
                csv_idx.append("Mean of HSV-Saturation")

            if np.count_nonzero(s_segf) != 0:
                mean_s_seg = sum(s_segf[np.nonzero(s_segf)]) / \
                    np.count_nonzero(s_segf)  # FEX mean_s_seg
                FL.append(mean_s_seg)
            else:
                FL.append(0.0)

        if (mv_f):
            if flname_on:
                flname.append("Mean of HSV-Value of segmented image")
            if csv_idx_on:
                csv_idx.append("Mean of HSV-Value")

            if np.count_nonzero(v_segf) != 0:
                mean_v_seg = sum(v_segf[np.nonzero(v_segf)]) / \
                    np.count_nonzero(v_segf)  # FEX mean_v_seg
                FL.append(mean_v_seg)
            else:
                FL.append(0.0)

        # standard deviation
        if (sdb_f):
            b_seg_std = np.std(b_segf)  # FEX b_seg_std
            FL.append(b_seg_std)
            if flname_on:
                flname.append(
                    "Standard Deviation of RGB-Blue of segmented image")
            if csv_idx_on:
                csv_idx.append("SD of RGB-Blue")

        if (sdg_f):
            g_seg_std = np.std(g_segf)  # FEX g_seg_std
            FL.append(g_seg_std)
            if flname_on:
                flname.append(
                    "Standard Deviation of RGB-Green of segmented image")
            if csv_idx_on:
                csv_idx.append("SD of RGB-Green")

        if (sdr_f):
            r_seg_std = np.std(r_segf)  # FEX r_seg_std
            FL.append(r_seg_std)
            if flname_on:
                flname.append(
                    "Standard Deviation of RGB-Red of segmented image")
            if csv_idx_on:
                csv_idx.append("SD of RGB-Red")

        if (bghist_f):
            bghistbins = 16
            total_bins = bghistbins * bghistbins
            img_rgb_hist_bg, img_rgb_hist_bg_edge1, img_rgb_hist_bg_edge2 = np.histogram2d(
                b_segf, g_segf, bins=bghistbins, range=((0, 255), (0, 255)))

            img_rgb_hist_bgf = np.ravel(img_rgb_hist_bg)

            # min-max normalization = (x - min)/(max - min)
            img_rgb_hist_bg_norm = (img_rgb_hist_bgf - min(img_rgb_hist_bgf)) / (
                max(img_rgb_hist_bgf) - min(img_rgb_hist_bgf))  # FEX img_rgb_hist_bg_norm

            FL.extend(np.ravel(img_rgb_hist_bg_norm))
            if flname_on:
                flname.append(
                    "min-max normalized Blue-Green Histogram (16*16 bins)")

            # writing feature column names
            if csv_idx_on:
                for i in range(total_bins):
                    csv_idx.append("BG-Hist-" + str(i))

        if (hshist_f):
            hshistbins = 16
            total_bins = hshistbins * hshistbins
            img_hsv_hist_hs, img_hsv_hist_hs_edge1, img_hsv_hist_hs_edge2 = np.histogram2d(
                img_hsv_chanel[0].flatten(), img_hsv_chanel[1].flatten(), bins=hshistbins, range=((0, 180), (0, 256)))

            img_hsv_hist_hsf = np.ravel(img_hsv_hist_hs)

            # min-max normalization = (x - min)/(max - min)
            img_hsv_hist_hs_norm = (img_hsv_hist_hsf - min(img_hsv_hist_hsf)) / (
                max(img_hsv_hist_hsf) - min(img_hsv_hist_hsf))  # FEX img_hsv_hist_hs_norm

            FL.extend(np.ravel(img_hsv_hist_hs_norm))
            if flname_on:
                flname.append(
                    "min-max normalized Hue-Saturation Histogram (16*16 bins)")

            # writing feature column names
            if csv_idx_on:
                for i in range(total_bins):
                    csv_idx.append("HS-Hist-" + str(i))

        # processing for contour detection
        ret1, seg_bin = cv.threshold(seg_gray, 40, 255, cv.THRESH_BINARY)
        seg_bin = cv.medianBlur(seg_bin, 5)
        seg_canny = cv.Canny(seg_bin, 100, 200)
        # cv.imwrite(outfolder+ img_out + name + "-seg_canny" + ".png", seg_canny)

        contours, hierarchy = cv.findContours(
            seg_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # seg_cnt = cv.drawContours(seg, contours, -1, (0, 0, 255), 2)
        # cv.imwrite(outfolder + img_out + name + "-seg_cntdraw" + ".png", seg_cnt)

        #
        # find contour pixel coordinates and intensity
        #
        cnt_intensity = []
        cnt_coord = []

        for c, e in enumerate(contours):
            mask = np.zeros(seg_canny.shape, np.uint8)
            cv.drawContours(mask, [contours[c]], 0, 255, -1)
            pixelpoints = np.transpose(np.nonzero(mask))  # contour coordinates
            cnt_coord.extend(pixelpoints)

            for p, e in enumerate(pixelpoints):
                # intensity at the contour coordinates
                intensity = seg_gray[pixelpoints[p][0]][pixelpoints[p][1]]
                if (intensity):
                    cnt_intensity.append(intensity)

        if (skw_f):
            seg_skew = skew(cnt_intensity, None)  # FEX seg_skew
            FL.append(seg_skew)
            if flname_on:
                flname.append("Skewness of diseased segment")
            if csv_idx_on:
                csv_idx.append("Skewness")

        if (kurt_f):
            # kurtosis (pearson)     #FEX seg_kurt
            seg_kurt = kurtosis(cnt_intensity, None, fisher=False)
            FL.append(seg_kurt)
            if flname_on:
                flname.append("Kurtosis of diseased segment")
            if csv_idx_on:
                csv_idx.append("Kurtosis")

        if (sdseg_f):
            seg_cnt_std = np.std(cnt_intensity)
            FL.append(seg_cnt_std)
            if flname_on:
                flname.append("Standard Deviation of diseased segment")
            if csv_idx_on:
                csv_idx.append("SD of diseased segment")

    if (shape_f):
        print "Extracting shape features..."

        if (cntperi_f):
            cnt_peri = []

            for c, e in enumerate(contours):
                perimeter = cv.arcLength(contours[c], True)
                cnt_peri.append(perimeter)

            # removing contours having less than 30% perimeter of the biggest contour
            if (np.count_nonzero(cnt_peri) != 0):
                max_cntp = max(cnt_peri)
            else:
                max_cntp = 0.0

            cnt_perimeter = []

            for i, e in enumerate(cnt_peri):
                if (cnt_peri[i] > (max_cntp * 0.3)):
                    cnt_perimeter.append(cnt_peri[i])

            if cnt_perimeter:
                cnt_perimeter_avg = sum(cnt_perimeter) / len(cnt_perimeter)
            else:
                cnt_perimeter_avg = 0.0

            FL.append(cnt_perimeter_avg)  # FEX cnt_perimeter_avg
            if flname_on:
                flname.append("Average contour perimeter")
            if csv_idx_on:
                csv_idx.append("Average contour perimeter")

        if (cntarea_f):
            cnt_ar = []

            for c, e in enumerate(contours):
                area = cv.contourArea(contours[c])
                cnt_ar.append(area)

            # removing contours having less than 30% area of the biggest contour
            if (np.count_nonzero(cnt_ar) != 0):
                max_cnt = max(cnt_ar)
            else:
                max_cnt = 0.0

            cnt_area = []

            for i, e in enumerate(cnt_ar):
                if (cnt_ar[i] > (max_cnt * 0.3)):
                    cnt_area.append(cnt_ar[i])

            if cnt_area:
                cnt_area_avg = sum(cnt_area) / len(cnt_area)
            else:
                cnt_area_avg = 0.0

            FL.append(cnt_area_avg)  # FEX cnt_area_avg
            if flname_on:
                flname.append("Average contour area")
            if csv_idx_on:
                csv_idx.append("Average contour area")

        if (hum_f):
            mmnt = cv.moments(seg_bin)
            hu_m = cv.HuMoments(mmnt)

            FL.extend(np.ravel(hu_m))  # FEX hu_mmnt
            if flname_on:
                flname.append("Hu Moments (7)")
            if csv_idx_on:
                for i in range(7):
                    csv_idx.append("Hu moment-" + str(i))

    if (texture_f):
        print "Extracting texture features..."

        if (glcm_f):
            # mean of 4 GLCM features in 0, 45 (np.pi / 4), 90 (np.pi / 2), 135 (3 * np.pi / 4) degrees
            glcm_seg = greycomatrix(
                seg_gray, [3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], normed=True)

            contrast = greycoprops(glcm_seg, 'contrast')  # FEX contrast

            contrast = contrast.mean()
            FL.append(contrast)

            # use the commented version for individual values (not mean)
            # contrast = [contrast[0][0], contrast[0][1], contrast[0][2], contrast[0][3]]
            # FL.extend(contrast)
            # if flname_on: flname.append("GLCM [at distance=3, angles=0,45,90,135]  Contrast")

            if flname_on:
                flname.append(
                    "GLCM [at distance=3, mean, angles=0,45,90,135]  Contrast")
            if csv_idx_on:
                csv_idx.append("GLCM Contrast")

            correlation = greycoprops(
                glcm_seg, 'correlation')  # FEX correlation

            correlation = correlation.mean()
            FL.append(correlation)

            # correlation = [correlation[0][0], correlation[0][1], correlation[0][2], correlation[0][3]]
            # FL.extend(correlation)

            if flname_on:
                flname.append("GLCM Correlation")
            if csv_idx_on:
                csv_idx.append("GLCM Correlation")

            energy = greycoprops(glcm_seg, 'energy')  # FEX energy

            energy = energy.mean()
            FL.append(energy)

            # energy = [energy[0][0], energy[0][1], energy[0][2], energy[0][3]]
            # FL.extend(energy)

            if flname_on:
                flname.append("GLCM Energy")
            if csv_idx_on:
                csv_idx.append("GLCM Energy")

            homogeneity = greycoprops(
                glcm_seg, 'homogeneity')  # FEX homogeneity

            homogeneity = homogeneity.mean()
            FL.append(homogeneity)

            # homogeneity = [homogeneity[0][0], homogeneity[0][1], homogeneity[0][2], homogeneity[0][3]]
            # FL.extend(homogeneity)

            if flname_on:
                flname.append("GLCM Homogeneity")
            if csv_idx_on:
                csv_idx.append("GLCM Homogeneity")

        if (lbphist_f):
            # normalized LBP (P, R = 24, 3.0) histogram
            # print "Calculating normalized LBP (P,R = 24, 3.0) histogram..."

            R = 3.0
            P = int(8 * R)
            lbphbins = 32

            lbp_seg = local_binary_pattern(seg_gray, P, R, method='uniform')

            lbp_seg_hist, lbp_seg_hist_edges = np.histogram(
                lbp_seg, bins=lbphbins, range=(0, 256))

            lbp_seg_hist = lbp_seg_hist.astype("float")
            lbp_seg_hist_norm = (lbp_seg_hist - min(lbp_seg_hist)) / \
                (max(lbp_seg_hist) - min(lbp_seg_hist))

            FL.extend(np.ravel(lbp_seg_hist_norm))
            if flname_on:
                flname.append(
                    "min-max normalized LBP (P, R = 24, 3.0) histogram (32 bins)")
            if csv_idx_on:
                for i in range(lbphbins):
                    csv_idx.append("LBP-Hist-" + str(i))

            # cv.imwrite(outfolder + img_out + name + "-lbp_seg" + ".png", lbp_seg.astype("uint8"))

    # formatting feature list
    # NaN values are replaced with 0
    FL = [0.0 if np.isnan(x) else x for x in FL]
#    FL = [round(x, 6) for x in FL]

    if (not feat_list.closed):
        feat_list.write(str(flname))
        total_feature = len(flname) - 2

        if (hum_f):
            total_feature += 6

        print >>feat_list, "\n \n Total number of features: {}".format(
            total_feature)

        print >>feat_list, "\n \n Dataset ID: {}".format(
            str(config['info']['dataset']).encode('ascii', 'ignore'))

        if (resize_f):
            print >>feat_list, ("\n Images downsized to: 512x256 or lower")

        feat_list.close()
        flname_on = 0

    if (not colout.closed):
        colout.write(str(csv_idx))
        colout.close()
        csv_idx_on = 0

    # dump features modularly
    if (filecount % 50 == 0):
        # save in temporary file for modular processing of large number of files
        with open(fextout + '/' + 'fltmp-' + str(filecount) + '.pkl', 'ab+') as tmpfile:
            cPickle.dump(FL, tmpfile, protocol=cPickle.HIGHEST_PROTOCOL)

        FL = []

    # print ("Length of FL (feature vector) = {}").format(len(FL))

# dump the rest of the features
tmpfile = open(fextout + '/' + 'fltmp-' + str(filecount + 1) + '.pkl', 'ab+')
cPickle.dump(FL, tmpfile, protocol=cPickle.HIGHEST_PROTOCOL)
tmpfile.close()

# print ("\n Number of feature column = {}").format(feat_num)

# concateate all the temprorary feature files
tmppath = glob.glob(fextout + "/" + "*.*")
FL = []
for f in tmppath:
    name, ext = os.path.splitext(os.path.basename(f))
    if (str(name).startswith("fltmp-")):
        with open(f, "rb") as ffile:
            FL.extend(cPickle.load(ffile))

        os.remove(f)

feat = np.reshape(FL, (-1, int(feat_num)))

df = pd.DataFrame(feat, columns=csv_idx)
with open(fextout + '/' + 'FL-' + timestamp + '.csv', 'w') as csvout:
    df.to_csv(csvout, sep=',', mode='w', decimal='.')

with open(fextout + '/' + 'FL-' + timestamp + '.pkl', 'wb') as outfile:
    cPickle.dump(feat, outfile, protocol=cPickle.HIGHEST_PROTOCOL)

outfile.close()

# remove previous files

remfile = glob.glob(outfolder + '*.*')
for f in remfile:
    os.remove(f)

with open(outfolder + 'feature columns.txt', 'w') as colout:
    colout.write(str(csv_idx))


shutil.copy2(fextout + '/' + 'FL-' + timestamp + '.pkl',
             outfolder + 'FL-' + timestamp + '.pkl')
shutil.copy2(fextout + '/' + 'Feature_List-' + timestamp + '.txt',
             outfolder + 'Feature_List-' + timestamp + '.txt')

# cleanup

fext_time = time.time() - fext_gen  # timing finished
print '\n Feature extraction completed in::: %.2fs.' % fext_time

# cv.waitKey(0)
# cv.destroyAllWindows()

armageddon = time.time() - genesis  # timing finished

print '\n Total time: %.3fs.' % armageddon
print "\n END OF CODE"

# run classifier
#execfile('ms-thesis -CLF.py')
