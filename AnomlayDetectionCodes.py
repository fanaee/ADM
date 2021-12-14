######################################################################
# Python codes for Anomaly Detection Lecture 
#
# Applied Data Mining Course, Fall 2021
# School Of Information Technology
# Halmstad University
#
# Hadi Fanaee, Ph.D., Assistant Professor
# hadi.fanaee@hh.se
# www.fanaee.com
######################################################################


#**********************************************************************
#Slide-23: Boxplot
#**********************************************************************

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("weight-height.csv")
plt.boxplot(df['Height'])


#**********************************************************************
#Slide-26: Histogram
#**********************************************************************

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("weight-height.csv")
plt.hist(df['Height'])
hist, bin_edges = np.histogram(df['Height'].to_numpy(), density=False, bins = 24)



#**********************************************************************
#Slide-29: Gaussian Model 
#**********************************************************************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.stats import norm
df=pd.read_csv("weight-height.csv")
h=df['Height'].to_numpy()
mu, std = norm.fit(h)
plt.hist(h, bins=24, density=True, color='w')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)



#**********************************************************************
#Slide-34: Gaussian Mixture Model 
#**********************************************************************

import pandas as pd
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import matplotlib.pyplot as plt
df=pd.read_csv("weight-height.csv")
h=df['Height'].to_numpy()
hp=h.reshape(-1, 1)
gmm = GaussianMixture(n_components = 2).fit(hp)
plt.figure()
plt.hist(hp, bins=24,  density=True)
plt.xlim(0, 360)
f_axis = hp.copy().ravel()
f_axis.sort()
a = []
for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
    a.append(weight*norm.pdf(f_axis, mean, np.sqrt(covar)).ravel())
    plt.plot(f_axis, a[-1])
plt.plot(f_axis, np.array(a).sum(axis =0), 'k-')
plt.xlabel('Height')
plt.ylabel('PDF')
plt.tight_layout()
plt.show()


#**********************************************************************
#Slide-41: Marginal Boxplot 
#**********************************************************************

from pandas import read_csv
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
df=pd.read_csv("weight-height.csv")
left = 0.1
bottom = 0.1
top = 0.8
right = 0.8
fig = plt.figure(figsize=(10, 10), dpi= 80)
main_ax = plt.axes([left,bottom,right-left,top-bottom])
top_ax = plt.axes([left,top,right - left,1-top])
plt.axis('off')
right_ax = plt.axes([right,bottom,1-right,top-bottom])
plt.axis('off')
main_ax.plot(df['Height'],  df['Weight'], 'ko', alpha=0.5)
right_ax.boxplot(df['Height'], notch=True, widths=.6)
top_ax.boxplot(df['Weight'], vert=False, notch=True, widths=.6)
plt.show()

#**********************************************************************
#Slide-48: Histogram-based Outlier Score (HBOS) 
#**********************************************************************

import pandas as pd
from pyod.models.hbos import HBOS
from pyod.utils.utility import standardizer
from pyod.utils.example import visualize
from pyod.utils.data import evaluate_print
df=pd.read_csv("weight-height.csv")
df = df.drop('Gender',1)
labels=np.zeros(10000)
labels[2014]=1 # We already know that this sample is an anomaly
dfn = standardizer(df)
# 24 bins, 1 outlier out of 10000 examples
clf = HBOS(n_bins=24, contamination=0.0001)
clf.fit(dfn)
anomaly_labels = clf.labels_
anomaly_scores = clf.decision_scores_
visualize('HBOS', dfn, labels, dfn, labels,anomaly_labels,
          anomaly_labels, show_figure=True, save_figure=False)
evaluate_print('HBOS', labels, anomaly_scores)


#**********************************************************************
#Slide-64: K-Nearest Neighbors (kNN)  
#**********************************************************************

import pandas as pd
from pyod.models.knn import KNN
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
df=pd.read_csv("height-weight_knn_example.csv")
dfn = standardizer(df.drop('ID',1))
clf = KNN(n_neighbors=2,contamination=1/25,method='mean')
clf.fit(dfn)
df['AnomalyScore'] = clf.decision_scores_
df['AnomalyLabel'] = clf.labels_


#**********************************************************************
#Slide-80: Local Outlier Factor (LOF)  
#**********************************************************************

import pandas as pd
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
df=pd.read_csv("height-weight_lof_example.csv")
dfn = standardizer(df.drop('ID',1))
clf = LOF(n_neighbors=3,contamination=1/25)
clf.fit(dfn)
df['AnomalyScore'] = clf.decision_scores_
df['AnomalyLabel'] = clf.labels_

#**********************************************************************
#Slide-112: Connectivity Outlier Factor (COF)   
#**********************************************************************

import pandas as pd
from pyod.models.cof import COF
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
df=pd.read_csv("height-weight_cof_example.csv")
dfn = standardizer(df.drop('ID',1))
clf = COF(n_neighbors=3,contamination=2/25)
clf.fit(dfn)
anomaly_labels = clf.labels_
anomaly_scores = clf.decision_scores_
df['AnomalyScore'] = clf.decision_scores_
df['AnomalyLabel'] = clf.labels_

#**********************************************************************
#Slide-117: One-Class SVM   
#**********************************************************************

import pandas as pd
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
df=pd.read_csv("height-weight_ocsvm_example.csv")
dfn = standardizer(df.drop('ID',1))
clf = OCSVM(kernel ='linear',contamination=3/25)
clf.fit(dfn)
df['AnomalyScore'] = clf.decision_scores_
df['AnomalyLabel'] = clf.labels_


#**********************************************************************
#Slide-124: DBSCAN   
#**********************************************************************

from sklearn.cluster import DBSCAN
from pyod.utils.utility import standardizer
import pandas as pd
from pyod.utils.utility import standardizer
df=pd.read_csv("height-weight_clustering_example.csv")
dfn = standardizer(df.drop('ID',1))
clustering = DBSCAN(eps=0.5, min_samples=4).fit(dfn)
anomaly_labels=clustering.labels_
anomaly_scores=clustering.labels_
df['AnomalyScore'] = anomaly_labels

#**********************************************************************
#Slide-148: PCA  
#**********************************************************************

import pandas as pd
from pyod.models.pca import PCA
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
# Generate high-dimensional artifical Data
# Normal data is generated by a multivariate Gaussian distribution and outliers are generated by a uniform distribution
X_train, y_train = generate_data(
    n_train=200, n_test=100, n_features=2000, train_only=True, contamination=0.05)
clf = PCA(contamination=0.05,n_selected_components=2)
clf.fit(X_train)
y_train_scores=clf.labels_
df = pd.DataFrame({'TrueLabel': y_train, 'AnoamlyLabel': y_train_scores})
evaluate_print('PCA', y_train, y_train_scores)

#**********************************************************************
# Slide-159: AutoEcndoer 
#**********************************************************************

import pandas as pd
from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.utils.data import evaluate_print
from pyod.utils.data import generate_data
# Generate high-dimensional artifical Data
# Normal data is generated by a multivariate Gaussian distribution and outliers are generated by a uniform distribution
X_train, y_train = generate_data(
    n_train=200, n_test=100, n_features=2000, train_only=True, contamination=0.05)
clf = AutoEncoder(contamination=0.05)
clf.fit(X_train)
y_train_scores=clf.labels_
df = pd.DataFrame({'TrueLabel': y_train, 'AnoamlyLabel': y_train_scores})
evaluate_print('AutoEncoder', y_train, y_train_scores)


#**********************************************************************
# Slide-173: ABOD  
#**********************************************************************

import pandas as pd
from pyod.models.abod import ABOD 
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
X=pd.read_csv("arrhythmia.csv")
true_label=pd.read_csv("arrhythmia_true_labels.csv").to_numpy()
outliers_fraction = np.count_nonzero(true_label) / len(true_label)
X= standardizer(X)
clf = ABOD(contamination=outliers_fraction)
clf.fit(X)
anomaly_label=clf.labels_
evaluate_print('ABOD', true_label, anomaly_label)


#**********************************************************************
# Slide-180: Isolation Forest   
#**********************************************************************

import pandas as pd
from pyod.models.iforest import IForest
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
X=pd.read_csv("arrhythmia.csv")
true_label=pd.read_csv("arrhythmia_true_labels.csv").to_numpy()
outliers_fraction = np.count_nonzero(true_label) / len(true_label)
X= standardizer(X)
# 100 trees, 256 subsamples
clf = IForest(contamination=outliers_fraction,n_estimators=100,max_samples =256)
clf.fit(X)
anomaly_label=clf.labels_
evaluate_print('IsolationForest', true_label, anomaly_label)


#**********************************************************************
# Slide-189: Feature Bagging 
#**********************************************************************

import numpy as np
import pandas as pd
from pyod.models.feature_bagging import FeatureBagging
from pyod.utils.utility import standardizer
from pyod.utils.data import evaluate_print
X=pd.read_csv("arrhythmia.csv")
true_label=pd.read_csv("arrhythmia_true_labels.csv").to_numpy()
outliers_fraction = np.count_nonzero(true_label) / len(true_label)
X= standardizer(X)
clf = FeatureBagging(contamination=outliers_fraction, contamination=outliers_fraction,n_estimators=100)
clf.fit(X)
anomaly_label=clf.labels_
evaluate_print(‘Feature Bagging', true_label, anomaly_label)

