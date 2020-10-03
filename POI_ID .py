
# coding: utf-8

# In[1]:

import sys
import pickle
sys.path.append("../tools/")
import helper_function as hf
from time import time
import keras
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score,precision_score,accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit


# In[2]:

'''Initial Features which will be used '''

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'deferral_payments',
                 'total_payments', 
                 'loan_advances', 
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income', 
                 'total_stock_value', 
                 'expenses',
                 'exercised_stock_options',
                 'other',
                 'long_term_incentive', 
                 'restricted_stock',
                 'director_fees',
                 'to_messages',
                 'from_poi_to_this_person', 
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#print data_dict.keys()  ## to view the names of al


# In[3]:

'''Outliner Removal'''

print "scatter plot Before outlier removal"   
for point in data_dict:
    salary = data_dict[point]["salary"]
    bonus = data_dict[point]["bonus"]
    matplotlib.pyplot.scatter( salary, bonus)


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")

matplotlib.pyplot.show()

### Task 2: Remove outliers
out=['TOTAL','LOCKHART EUGENE E','THE TRAVEL AGENCY IN THE PARK']


'''Uncomment to see why they are being removed'''
#print data_dict['THE TRAVEL AGENCY IN THE PARK']
#print data_dict['LOCKHART EUGENE E']
#print data_dict['TOTAL']
for i in out:
    data_dict.pop(i)
    
print "Scatter plot After outlier removal"   
for point in data_dict:
    salary = data_dict[point]["salary"]
    bonus = data_dict[point]["bonus"]
    matplotlib.pyplot.scatter( salary, bonus)


matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.savefig("bonus_vs_salary.png")
matplotlib.pyplot.show()


# In[4]:

for point in data_dict:
    salary = data_dict[point]["from_poi_to_this_person"]
    bonus = data_dict[point]["from_this_person_to_poi"]
    matplotlib.pyplot.scatter( salary, bonus)


matplotlib.pyplot.xlabel("from_poi_to_this_person")
matplotlib.pyplot.ylabel("from_this_person_to_poi")
matplotlib.pyplot.savefig("image.png")
matplotlib.pyplot.show()


# In[5]:

'''Creating new features'''

'''
New features are being created by me using the helper function the new features are networth ,proportional_from_poi and proportional_to_poi
'''    
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_dict = hf.net_worth(data_dict)

data_dict=hf.pfp(data_dict)
data_dict=hf.ptp(data_dict)
features_list+=['net_worth','proportion_from_poi','proportion_to_poi']

#print (data_dict["ALLEN PHILLIP K"]) ##Uncomment to see the example
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

for point in data_dict:
    salary = data_dict[point]["proportion_from_poi"]
    bonus = data_dict[point]["proportion_to_poi"]
    matplotlib.pyplot.scatter( salary, bonus)


matplotlib.pyplot.xlabel("proportion_from_poi")
matplotlib.pyplot.ylabel("proportion_to_poi")
matplotlib.pyplot.savefig("from_to_plot.png")
matplotlib.pyplot.show()


# In[6]:


features,features_list,f_scores=hf.select_features(features,labels,features_list,k=6)
# call the function with uses selectkbest
print ("features_list---" ,features_list)
print("feature scores")
for i in f_scores:
    print (i)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[7]:


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

clf = GaussianNB()
test_classifier(clf,my_dataset,features_list)
clf1=tree.DecisionTreeClassifier()
test_classifier(clf1,my_dataset,features_list)
clf2 = AdaBoostClassifier()
test_classifier(clf2,my_dataset,features_list)
clf3=KNeighborsClassifier(n_neighbors = 4)
test_classifier(clf3,my_dataset,features_list)


# In[8]:

from sklearn.neighbors.nearest_centroid import NearestCentroid
clf4 = NearestCentroid()
test_classifier(clf4,my_dataset,features_list)



# In[7]:

'''Final Algorithm'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)
    
   
t0 = time()
pipe1 = Pipeline([('pca',PCA()),('classifier',GaussianNB())])
param = {'pca__n_components':[4,5,6]}
gsv = GridSearchCV(pipe1, param_grid=param,n_jobs=2,scoring = 'f1',cv=2)
gsv.fit(features_train,labels_train)
clf = gsv.best_estimator_
print("GausianNB with PCA fitting time: %rs" % round(time()-t0, 3))
pred = clf.predict(features_test)

t0 = time()
test_classifier(clf,my_dataset,features_list,folds = 1000)
print("GausianNB  evaluation time: %rs" % round(time()-t0, 3))
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)





# In[12]:

'''
Adaboost tuned for comparision with final algorithm

'''
abc = AdaBoostClassifier(random_state=40)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
dt = []
for i in range(6):
    dt.append(DecisionTreeClassifier(max_depth=(i+1)))
ab_params = {'base_estimator': dt,'n_estimators': [60,45, 101,10]}
t0 = time()
abt = GridSearchCV(abc, ab_params, scoring='f1',)
abt = abt.fit(features_train,labels_train)
print("AdaBoost fitting time: %rs" % round(time()-t0, 3))
abc = abt.best_estimator_
t0 = time()
test_classifier(abc, data_dict, features_list, folds = 100)
print("AdaBoost evaluation time: %rs" % round(time()-t0, 3))






# In[12]:

'''Honorable Mention'''
#GradientBoosting  tuning for better performance
#Remove docstring for running 
#It takes time

'''
from sklearn.ensemble import GradientBoostingClassifier
t0 = time()
params = {'n_estimators': [250], 'max_depth': [3,2],  'min_samples_split':[2,1,3],
          'learning_rate': [0.1,0.2], 'min_samples_leaf':[3,5,6] , 'random_state': [2,3]}
clf5 = GradientBoostingClassifier()
gbc = GridSearchCV(clf5, param_grid=params)
model = gbc.fit(features_train, labels_train)
cf=model.best_estimator_
print("GradientBoosting fitting time: %rs" % round(time()-t0, 3))
print"Will take a little more time"
t0 = time()
test_classifier(cf,my_dataset,features_list)
print("GradientBoosting evaluation time: %rs" % round(time()-t0, 3))
'''




