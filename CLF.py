import os
import glob
import sys
import cPickle
import configparser
import random
import time
import datetime as dt
import numpy as np
import xgboost as xgb
from matplotlib import cm
from matplotlib import pyplot as plt
from prettytable import PrettyTable as prt
from sklearn import metrics, svm
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold


# timing and timestamping functions
genesis = time.time()
timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#
# file and folder names
#
config = configparser.ConfigParser()
config.read('codeconfig.ini')

infolder = config['path']['infolder']
outfolder = config['path']['outfolder']

feat_fname = glob.glob(outfolder + 'FL-*.pkl')

# process the filename string of feature file
#
feat_fname = str(feat_fname[0])
feat_fname = feat_fname.split('\\', 1)[1]
feat_file = feat_fname.split('.', 1)[0]

with open(outfolder + feat_fname, 'rb') as infile:
    feat_in = cPickle.load(infile)

with open(outfolder + 'feature columns.txt', "r") as inp:
    fcols = inp.read()

# matplotlib plotting global parameters
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

#
# flags
#
svm_f = 0
xgb_f = 1

# global variable for comparison
SVM_ACC = 0.0
SVM_CV = 0.0
XGB_ACC = 0.0
XGB_CV = 0.0

#
# class info
#
n_classes = int(config['classinfo']['n_classes'])
clsnames = str(config['classinfo']['target_names']).encode('ascii', 'ignore')

target_names = []
labels = []

for i in range(n_classes):
    target_names.append(clsnames.split(', ')[i])
    labels.append(float(i + 1))


# seed -> score
# 7487 -> 91.67, cv90

#seed_rand = 7487
seed_rand = random.randint(0, 9999)

test_split = 0.2

fold = 10
repeat = 1

X = feat_in[..., 2:]
y, processing_id = feat_in[..., 1], feat_in[..., 0]

# processing_id = classnum + file_num
X_train, X_test, y_train, y_test, prid_train, prid_test = train_test_split(
    X, y, processing_id, test_size=test_split, random_state=seed_rand, stratify=y)

prid_test_files = []
test_class_list = []

#
# process processing_id string to extract class and file number
#
for i, e in enumerate(prid_test):
    prid = int(prid_test[i])
    prid_str = str(prid)
    class_number = prid_str[0]
    test_class_list.append(class_number)
    file_number = prid_str[1:]

    found = 0
    for i in range(n_classes):
        if (class_number == str(int(labels[i]))):
            class_name = target_names[i]
            found = 1

    if (not found):
        print ("Wrong class - {}").format(class_number)
        sys.exit(0)

    filename = class_name + " (" + file_number + ")"
    prid_test_files.append(filename.lower())


def evaluate_clf(clfload, clf_name, feat_file, clf, clfrep, X, y, X_train, y_train, X_test, y_test, y_pred):
    #
    # method for classifier evaluation and reporting
    #
    clf_report = metrics.classification_report(
        y_test, y_pred, labels=labels, target_names=target_names)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    conf = metrics.confusion_matrix(y_test, y_pred)

    print ("\n Feature file ::: \n {}").format(feat_file)
    print >>clfrep, ("\n Feature file ::: \n {}").format(feat_file)

    print ("\n \n Classifier ::: \n {}").format(clf)
    print >>clfrep, ("\n \n Classifier ::: \n {}").format(clf)

    print ("\n \n Classification report ::: \n{}").format(clf_report)
    print >>clfrep, ("\n \n Classification report ::: \n{}").format(clf_report)

    table = prt(["Seed", "Test split"])
    table.add_row([seed_rand, test_split])

    print ("\n \n {}").format(table)
    print >>clfrep, ("\n \n {}").format(table)

    table = prt(["Accuracy (%)"])
    table.add_row([round(accuracy, 4) * 100])

    print ("\n \n {}").format(table)
    print >>clfrep, ("\n \n {}").format(table)

    print ("\n \n Confusion matrix ::: \n {} \n").format(conf)
    print >>clfrep, ("\n \n Confusion matrix ::: \n {} \n").format(conf)

    #
    # map test files to prediction
    #
    table = prt(["Test file", "Real class", "Predicted class"])
    table.align["Test file"] = "l"
    table.align["Real class"] = "c"
    table.align["Predicted class"] = "c"

    for i, e in enumerate(X_test):
        testfile = prid_test_files[i]
        realclass = target_names[int(test_class_list[i]) - 1]
        predclass = target_names[int(y_pred[i]) - 1]

        # marking misclassified data
        if (realclass != predclass):
            table.add_row([testfile, realclass, predclass])

    print ("\n Misclassified samples :::")
    print >>clfrep, ("\n Misclassified samples :::")

    print ("\n {}").format(table.get_string(sortby="Test file"))
    print >>clfrep, ("\n {}").format(table.get_string(sortby="Test file"))

    # print SVM best parameters from GridSearchCV
    svm_best_param = 0
    if (clf_name == "SVM-RBF-"):
        if not clfload:
            svm_best_score = clf.best_score_
            svm_best_param = clf.best_params_

            print("\n \n Best: {} using {}").format(
                svm_best_score, svm_best_param)
            print >>clfrep, ("\n \n Best: {} using {}").format(
                round(svm_best_score, 5), svm_best_param)
    #
    # classwise accuracy calculation
    #
    cmat, classwise_acc, accu, clf_new = eval_classwise(
        clf_name, feat_file, clf, clfrep, X, y, svm_best_param)

    table = prt(["Class", "Accuracy (%)"])
    table.align["Class"] = "l"
    table.align["Accuracy (%)"] = "c"

    classwise_acc_avg = []

    for i in range(n_classes):
        cls_i_acc_avg = round(classwise_acc[i], 4) * 100
        classwise_acc_avg.append(cls_i_acc_avg)
        table.add_row([target_names[i], cls_i_acc_avg])

    print ("\n {}").format(table)
    print >>clfrep, ("\n {}").format(table)

    #
    # plot classwise accuracy
    #
    x_pos = np.arange(len(target_names))
    y_pos = range(0, 105, 5)
    perf = classwise_acc_avg

    my_cmap = cm.get_cmap('Blues')
    plt.clf()
    plt.bar(x_pos, perf, align='center', width=0.2, color=my_cmap(perf))

    plt.ylabel('Average accuracy')
    plt.xlabel('Class')
    plt.xticks(x_pos, target_names)
    plt.yticks(y_pos, y_pos)
    plt.title('Class-wise accuracy rate')
    plt.plot()
    plt.savefig(clfout + '/' + 'CLF-' + timestamp + '-classwise accuracy.pdf')

    #
    # cross validate with new optimized classifier
    #
    clf = clf_new

    cv_score = cross_val_score(
        clf, X, y, cv=RepeatedStratifiedKFold(fold, repeat))

    table = prt(["Mean of " + str(repeat) + " times " + str(fold) +
                 "-fold CV accuracy (%)"])
    table.add_row([round(cv_score.mean(), 4) * 100])

    print ("\n \n {}").format(table)
    print >>clfrep, ("\n \n {}").format(table)

    # cleanup
    clfrep.close()
    with open(clfout + '/' + 'CLF-' + timestamp + '.pkl', 'wb') as outfile:
        cPickle.dump(clf, outfile, protocol=cPickle.HIGHEST_PROTOCOL)

    clf_time = time.time() - clf_gen  # timing finished

    print '\n Classification finished in::: %.2fs.' % clf_time

    return [(round(accuracy, 4) * 100), (round(cv_score.mean(), 4) * 100)]
    # method finished


def eval_classwise(clf_name, feat_file, clf, clfrep, X, y, svm_best_param):
    cmat = []
    accu = []

    # new SVM classifier with optimized parameters
    if (clf_name == "SVM-RBF-"):
        if not clfload:
            old_clf = clf
            clf = svm.SVC(C=svm_best_param["C"], gamma='scale', kernel='rbf')

    #
    # manual m-times k-fold CV
    #
    for i in range(repeat):
        # random.seed(i**2)
        #seed_rand = random.randint(0, 999)

        skf = StratifiedKFold(fold, shuffle=True, random_state=seed_rand)

        for train_idx, test_idx in skf.split(X, y):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            clf = clf.fit(X_train, y_train)

            # new XGB prediction with optimized parameters
            if (clf_name == "XGB-"):
                y_pred = clf.predict(X_test, ntree_limit=clf.best_ntree_limit)
            else:
                y_pred = clf.predict(X_test)

            acc = metrics.accuracy_score(y_test, y_pred, normalize=True)
            accu.append(acc)

            # calculating confusion matrix every round
            conf = metrics.confusion_matrix(y_test, y_pred)
            cm = conf.astype("float").diagonal() / conf.sum(axis=1)
            cmat.append(cm)

    cmat = np.ravel(cmat)
    cmat = np.reshape(cmat, (-1, n_classes))

    classwise_acc = []

    for i in range(n_classes):
        cls_i_acc = np.array(cmat)[:, i]
        cls_i_acc_avg = sum(cls_i_acc) / len(cls_i_acc)
        classwise_acc.append(cls_i_acc_avg)

    return cmat, classwise_acc, accu, clf


if (svm_f):
    print "\n Beginning image classification... \n"
    clf_gen = time.time()

    clf_name = "SVM-RBF-"

    clfout = os.path.join(outfolder + 'CLF-' + clf_name + timestamp)
    os.makedirs(clfout)

    clfrep = open(clfout + '/' + 'Classification_Report-' +
                  timestamp + '.txt', 'w')

    # SVM
    clfload = 0
    if clfload:
        # load classifier
        clfpath = glob.glob(
            outfolder + "CLF-SVM-RBF-2020-01-01_00-00-06/CLF-2020-01-01_00-00-06.pkl")

        clfpath = str(clfpath[0])
        with open(clfpath, "rb") as clfin:
            clf = cPickle.load(clfin)
    else:
        # create classifier
        # gamma ='scale' => 1 / (n_features * X.var())
        svc = svm.SVC(gamma='scale', kernel='rbf', cache_size=100)
        # parameter tuning
        param_grid = [{'C': [2**9, 2**10, 2**15, 2**20]}]
        clf = GridSearchCV(svc, param_grid,
                           cv=StratifiedKFold(fold), iid=False, verbose=0, error_score=np.nan)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    SVM_ACC, SVM_CV = evaluate_clf(clfload, clf_name, feat_file,
                                   clf, clfrep, X, y, X_train, y_train, X_test, y_test, y_pred)


if (xgb_f):
    print "\n Beginning image classification... \n"
    clf_gen = time.time()

    clf_name = "XGB-"

    clfout = os.path.join(outfolder + 'CLF-' + clf_name + timestamp)
    os.makedirs(clfout)

    clfrep = open(clfout + '/' + 'Classification_Report-' +
                  timestamp + '.txt', 'w')

    # xGBoost
    clfload = 0
    if clfload:
        # load classifier
        clfpath = glob.glob(
            outfolder + "CLF-XGB-2020-01-01_00-00-06/CLF-2020-01-01_00-00-06.pkl")

        clfpath = str(clfpath[0])
        with open(clfpath, "rb") as clfin:
            clf = cPickle.load(clfin)
    else:
        clf = xgb.XGBClassifier(min_child_weight=5)

    eval_set = [(X_test, y_test)]
    clf = clf.fit(X_train, y_train, early_stopping_rounds=10,
                  eval_metric="mlogloss", eval_set=eval_set, verbose=0)

    y_pred = clf.predict(X_test, ntree_limit=clf.best_ntree_limit)

    #
    # extracting feature columns (from file)
    #
    # processing file name string
    fcols = fcols.replace('\'', '')
    fcols = fcols.replace('[', '')
    fcols = fcols.replace(']', '')

    fc = []
    for i in range(feat_in.shape[1]):
        a = fcols.split(', ')[i]
        fc.append(a)

    fcols = fc[2:]  # cut out first two columns (prid, class)

    impr = clf.feature_importances_
    impr = np.array(impr)
    # find largest n features
    impr_sort = impr.argsort()[-20:][::-1]

    imp_feat = []
    # map important features to name
    for i, e in enumerate(impr_sort):
        imp_feat.append(fcols[impr_sort[i]])

    imp_f = imp_feat[::-1]  # reverse the list (descending order)

    # dump the XGB model
    clf.get_booster().dump_model(clfout + '/' + 'CLF-' + timestamp +
                                 '-xgb-dump.txt', with_stats=True, dump_format='text')

    # plot feature importance by gain
    plt.clf()
    xgb.plot_importance(clf, max_num_features=20,
                        importance_type='gain', show_values=False, grid=False)

    plt.yticks(range(len(imp_feat)), imp_f, rotation=45)
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.tick_params(axis='y', which='major', labelsize=4)
    plt.xlabel("Gain")
    plt.ylabel("Features")
    plt.plot()
    plt.savefig(clfout + '/' + 'CLF-' + timestamp + '-feature importance.pdf')

    XGB_ACC, XGB_CV = evaluate_clf(clfload, clf_name, feat_file,
                                   clf, clfrep, X, y, X_train, y_train, X_test, y_test, y_pred)


# final report
print ("\n \n Summing up...")

table = prt(["*****", "Support Vector Machine", "Extreme Gradient Boosting"])

table.add_row(["Accuracy", SVM_ACC, XGB_ACC])
table.add_row([str(fold) + "-fold CV", SVM_CV, XGB_CV])
print ("\n \n {}").format(table)


# finishing up
armageddon = time.time() - genesis  # timing finished

print '\n Total time: %.3fs.' % armageddon
print "\n END OF CODE"
