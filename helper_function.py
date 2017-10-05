
# coding: utf-8

# In[2]:

from sklearn.feature_selection import SelectKBest,f_classif
f_scores =[]



def pfp(data_dict):
    for k, v in data_dict.iteritems():
#Assigning value to the feature 'proportion_from_poi'

        if v['from_poi_to_this_person'] != 'NaN' and  v['from_messages'] != 'NaN':
            v['proportion_from_poi'] = float(v['from_poi_to_this_person']) / v['from_messages'] 
        else:    
            v['proportion_from_poi'] = 0.0
    return (data_dict)       
            
def ptp(data_dict):
    for k, v in data_dict.iteritems():
        #Assigning value to the feature 'proportion_to_poi'        
        if v['from_this_person_to_poi'] != 'NaN' and  v['to_messages'] != 'NaN':
            v['proportion_to_poi'] = float(v['from_this_person_to_poi'] )/ v['to_messages']   
        else:
            v['proportion_to_poi'] = 0.0
    return (data_dict)

# In[ ]:

def net_worth (data_dict) :
    features = ['total_payments','total_stock_value']
    
    for key in data_dict :
        name = data_dict[key]
        
        is_null = False 
        
        for feature in features:
            if name[feature] == 'NaN':
                is_null = True
        
        if not is_null:
            name['net_worth'] = name[features[0]] + name[features[1]]
        
        else:
            name['net_worth'] = 'NaN'
            
    return data_dict                
            
def select_features(features,labels,features_list,k=10) :
    clf = SelectKBest(f_classif,k)
    new_features = clf.fit_transform(features,labels)
    features_l=[features_list[i+1] for i in clf.get_support(indices=True)]
    f_scores = zip(features_list[1:],clf.scores_[:])
    f_scores = sorted(f_scores,key=lambda x: x[1],reverse=True)
    return new_features, ['poi'] + features_l, f_scores
