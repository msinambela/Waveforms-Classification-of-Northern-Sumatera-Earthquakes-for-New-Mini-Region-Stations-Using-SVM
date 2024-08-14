#!/usr/bin/env python
# coding: utf-8

# In[10]:


#get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
import sys, os

warnings.filterwarnings('ignore')
random_state = 10
np.random.seed(random_state)


# In[11]:


dir_path = '../../pipeline_modules/'
sys.path.append(dir_path)


# In[12]:


## data62 = 'seismogram_data_62_new.csv'
## data66 = 'seismogram_data_66_new.csv'
data73 = 'seismogram_data_disertasi.csv'
df = pd.read_csv(data73)
print(df.shape)
print(df.columns)


# In[13]:


df.head()


# ### Processing the data

# In[14]:


df = shuffle(df, random_state = random_state)
df_train, df_test = train_test_split(df, test_size = 0.20, random_state= random_state)
mms = StandardScaler()
X_train = mms.fit_transform(df_train.drop(['id', 'target', 'moment', 'variation'], axis=1))
Y_train = df_train['target']


# ## plot t-SNE

# In[20]:


from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_train)
X_embedded.shape


# In[22]:


plt.scatter(X_embedded[:,0], X_embedded[:,1], c=Y_train)
plt.show()


# # Support Vector machine
# 
# In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

# ## Find best parameter using GridSearchCV
# 
# Exhaustive search over specified parameter values for an estimator. GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used. The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# In[15]:


from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = random_state)
svm_clss = svm.SVC(class_weight = 'balanced', random_state = random_state) 

param_dist = {'C': np.linspace(0.1, 10, 20), 
              'gamma': np.linspace(0.1, 0.00008, 30)}

# run randomized search
n_iter_search = 20

grid_clf = RandomizedSearchCV(estimator = svm_clss, param_distributions = param_dist, n_iter = 20, cv = cv, n_jobs=-1)


# In[30]:


grid_clf.fit(X_train, Y_train.astype(int))


# In[31]:


print(grid_clf.best_params_)
print(grid_clf.best_score_)
print(grid_clf.best_estimator_)


# ## Use best parameters to make final model 

# In[16]:


from sklearn import svm

svm_model = svm.SVC(C=4, cache_size=200, coef0=0.0, class_weight='balanced',
  decision_function_shape=None, degree=3, gamma=0.08, kernel='rbf',
  max_iter=-1, probability=False, random_state = random_state, shrinking=True,
  tol=0.001, verbose=True)

svm_model.fit(X_train, Y_train)


# In[17]:
import numpy as np
from sklearn.metrics import roc_auc_score


X_test  = mms.fit_transform(df_test.drop(['id', 'target', 'moment', 'variation'], axis=1))
Y_test = df_test['target']

Y_pred = svm_model.predict(X_test)
try:
    roc_auc_score(Y_pred, Y_test)
except ValueError:
    pass

#Confusion matrix
#import numpy as np
#import seaborn as sns
#from sklearn.metrics import classification_report
#import pandas as pd
#from sklearn.metrics import classification_report
#classificationReport = classification_report(Y_test, Y_pred)

#plot_classification_report(classification_report)
#plot_classification_report(classification_report, with_avg_total=True)

print('1. The accuracy of the model is {}\n'.format(accuracy_score(Y_test, Y_pred)))
print('2. Classification report \n {} \n'.format(classification_report(Y_test, Y_pred)))
print('3. Confusion matrix \n {} \n'.format(confusion_matrix(Y_pred, Y_test)))
#print('4. roc_auc score \n {}'.format(roc_auc_score(Y_pred, Y_test)))
#plot.show('3. Confusion matrix \n {} \n')

# In[16]:


classified_df = pd.DataFrame([], columns=['stations', 'original', 'predicted'])
misclassified_df = pd.DataFrame([], columns=['stations', 'original', 'predicted'])

for i, label in enumerate(Y_test):
    result = svm_model.predict([X_test[i, :]])
    
    if result[0] == label:    
        classified_df.loc[i] = [df_test.iloc[i, 0], label, result[0]]
    else:
        misclassified_df.loc[i] = [df_test.iloc[i, 0], label, result[0]]


# In[22]:


print(classified_df.head())
print('\n----------------------------------------------------------------------\n')
print(misclassified_df.head())


# In[21]:


import pickle
data = {'miss': misclassified_df, 'good':classified_df }
output = open('classification_data_new.pkl', 'wb')
pickle.dump(data, output)
output.close()


# ### robustness of the model

# In[18]:


data_robust = 'seismogram_data_disertasi.csv'
df_robust = pd.read_csv(data_robust)
df_robust = shuffle(df_robust)


# In[19]:


df_robust_test = df_robust.drop(['id', 'target', 'moment', 'variation'], axis=1)
df_robust_target = df_robust['target']

X_test_robust  = mms.fit_transform(df_robust_test)
Y_test_robust = df_robust_target

Y_pred_robust = svm_model.predict(X_test_robust)

print('1. The accuracy of the model is {}\n'.format(accuracy_score(Y_test_robust, Y_pred_robust)))
print('2. Classification report \n {} \n'.format(classification_report(Y_test_robust, Y_pred_robust)))
print('3. Confusion matrix \n {} \n'.format(confusion_matrix(Y_pred_robust, Y_test_robust)))
#print('4. Roc_Auc score \n {}'.format(roc_auc_score(Y_pred_robust, Y_test_robust)))


# ## Refinement history

# <table style="border: 1px solid black;">
#   <tr>
#     <th>Step</th>
#     <th>Method</th> 
#     <th>Description</th>
#     <th>Number of Features</th>
#     <th>Accuracuy</th>
#     <th>Pos Recall</th>
#     <th>Pos F1 score</th>
#     <th>ROC score</th>
#   </tr>
#   
#   <tr>
#     <td>Step 0</td>
#     <td>Benchmark</td> 
#     <td>At this step I used five of supervised algorithoms to find out the best one that gives highest model performance. I have found Support vector machine is the one gives best model performances interms of accuracy, Recall, F-1 and ROC score.</td>
#      <td>61</td>
#     <td>77%</td> 
#     <td>77%</td>
#     <td>76%</td>
#     <td>77.5%</td> 
#   </tr>
#   
#   <tr>
#     <td>Step 1</td>
#     <td>Grid search</td> 
#     <td>After I select SVM as the model towards final solution, I used scikit-sklearn GridSearcgCV model to find best C and gamma parameters. After exhaustive search with many hours, the best C and gamma are 4.0 and 0.08</td>
#      <td>61</td>
#     <td>84.5%</td> 
#     <td>81%</td>
#     <td>81%</td>
#     <td>84.5%</td> 
#   </tr>
#   
#     <tr>
#     <td>Step 2.1 </td>
#     <td>Feature engineering</td> 
#     <td>At this stage, I work with feature manipulations. I have tried adding some extra features like amplitude of the signal, mel coefficients etc. BUt the did not improve.</td>
#      <td>61-1285</td>
#     <td>84.50% - 75.00%</td> 
#     <td>81.00% - 74.00%</td>
#     <td>81.00% - 78.00%</td>
#     <td>84.50% -69.00% </td> 
#   </tr>
# </table>

# ## Learning curve to check if more data is required to improve performance
# 
# In this step of the project, I will check the learning curve of the support vector machine. A learning curve shows the validation and training score of an estimator for varying numbers of training samples. It is a tool to find out how much we benefit from adding more training data and whether the estimator suffers more from a variance error or a bias error. If both the validation score and the training score converge to a value that is too low with increasing size of the training set, we will not benefit much from more training data. If the training score is much higher than the validation score for the maximum number of training samples, adding more training samples will most likely increase generalization. In the following plot of SVM estimator, we see that the model could benefit from more training examples. We have to add training data to get improved performance.

# In[1]:


# %reload_ext autoreload
# %autoreload 2
# import visuals as vs
# vz = vs.vizualization(df)
# vz.check_model_learning(X_train, Y_train, svm_model)


# ## Bagging of SVM
# 
# In this bagging classification ithe above SVM model is the meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator, by introducing randomization into its construction procedure and then making an ensemble out of it.

# In[130]:


from sklearn.ensemble import BaggingClassifier
bagging_svm = BaggingClassifier(svm_model, max_samples=0.4, n_estimators=100, n_jobs=-1)
bagging_svm.fit(X_train, Y_train.astype(int))


# In[131]:


Y_pred = bagging_svm.predict(X_test)

print('1. The accuracy of the model is {}\n'.format(accuracy_score(Y_test, Y_pred)))
print('2. Classification report \n {} \n'.format(classification_report(Y_test, Y_pred)))
print('3. Confusion matrix \n {} \n'.format(confusion_matrix(Y_pred, Y_test)))
#print('4. Roc_Auc score \n {}'.format(roc_auc_score(Y_pred, Y_test)))


# In[ ]:




