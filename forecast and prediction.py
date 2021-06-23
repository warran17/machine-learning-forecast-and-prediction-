#!/usr/bin/env python
# coding: utf-8

# # Review
# Hi, my name is Dmitry and I will be reviewing your project.
# 
# You can find my comments in colored markdown cells:
# 
# <div class="alert alert-success">
#     If everything is done succesfully.
# </div>
# 
# <div class="alert alert-warning">
#     If I have some (optional) suggestions, or questions to think about, or general comments.
# </div>
# 
# <div class="alert alert-danger">
#     If a section requires some corrections. Work can't be accepted with red comments.
# </div>
# 
# Please don't remove my comments, as it will make further review iterations much harder for me.
# 
# Feel free to reply to my comments or ask questions using the following template:
# 
# <div class="alert alert-info">
#     For your comments and questions.
# </div>
# 
# First of all, thank you for turning in the project! You did a great job! The project is accepted. Good luck on the final sprint!

# **Project description**
# 
# The gym chain Model Fitness is developing a customer interaction strategy based on analytical data.
# One of the most common problems gyms and other services face is customer churn. How do you know if a customer is no longer with you? You can calculate churn based on people who get rid of their accounts or don't renew their contracts. However, sometimes it's not obvious that a client has left: they may walk out on tiptoes.
# 

# ### Step 1. Download the data

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier


# In[2]:


fitness= pd.read_csv('/datasets/gym_churn_us.csv')


# In[3]:


fitness.head()


# In[4]:


fitness.columns= fitness.columns.str.lower()


# In[5]:


fitness.info()


# > **Required libraries are imported and data is downloaded**

# ### Step 2. Carry out exploratory data analysis (EDA)

# In[6]:


fitness.isna().any()


# > no missing value

# In[7]:


fitness.describe()


# **gender :**                              slightly more men than female
# 
# **near_location:**                        about 84.5 % of users lives or works in the neighborhood where the gym is located
# 
# **partner :**                             About 48.6% of users are employees of a partner company
# 
# **promo_friends :**                       about 30.8 % of users originally signed up through a "bring a friend" offer 
# 
# **phone        :**                        about 90.3% of users provided their phone number
# 
# **contract_period**:                     contract period last about 4.68 months in average
# 
# **group_visits :**                        49.2 % of users takes part in group sessions
# 
# **age          :**                        users average age is 29.18
# 
# **avg_additional_charges_total :**        the total amount of money spent on other gym services: cafe, athletic goods, cosmetics, massages, etc.is
#                                         146.94
#         
# **month_to_end_contract:**                the months remaining until the contract expires is about 4.32 months on average
# 
# **lifetime :**                            the time (in months) since the customer first came to the gym is 3.72 months on average
# 
# **avg_class_frequency_total:**            average frequency of visits per week over the customer's lifetime is 1.87 times on average
# 
# **avg_class_frequency_current_month:**    average frequency of visits per week over the preceding month is 1,76 times on average
# 
# **churn  :**                              26.52% users left to use gym

# In[8]:


print(fitness.groupby(['churn']).mean())


# > More proportion of users from neighborhood stayed using the gym compared to those who are not living or working in neighbourhood. Likewise,
#  more proportion of users who are emplyoed in partner company stayed using gym compared to those sign up through a "bring a friend" offer. people who didnot left have longer contract period on average compared to 
#     those  who left the gym. Also , less number of people who participate in group session left the gym compared to those who dont participate in group session.
#     Likewise, who stayed using gym has higher average age, more additional spending, more month to end contract, longer lifetime, avg_additional_charges_total,
#       churn  than who left the gym.

# In[9]:


fitness.columns


# In[10]:


fitness["churn"].value_counts()


# > About 26% users left the gym

# In[11]:


churn_left= fitness[fitness['churn']==1]


# In[12]:


churn_stayed= fitness[fitness['churn']==0]


# In[13]:


fitness.columns.to_series().groupby(fitness.dtypes).groups


# In[14]:


list_numerical=['gender', 'near_location', 'partner', 'promo_friends','phone']


# In[15]:


dataset2 = fitness[['avg_additional_charges_total', 'month_to_end_contract',
        'avg_class_frequency_total', 'avg_class_frequency_current_month', 'contract_period', 'age','lifetime','churn']]
dataset_left= dataset2[dataset2['churn']== 1]
dataset_stayed= dataset2[dataset2['churn']== 0]
#Histogram:
    
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns\n',horizontalalignment="center",fontstyle = "normal", fontsize = 24, fontfamily = "sans-serif")
for i in range(dataset2.shape[1]):
    plt.subplot(5, 2, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 12:
        vals = 12
    plt.hist(dataset_left.iloc[:, i], bins=vals, color = 'red', alpha= 0.7, label = 'gym_left')
    plt.hist(dataset_stayed.iloc[:, i], bins=vals, color = 'green', alpha=0.3, label = 'gym_stayed')
    

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# > Pattern of additional spending is similar in some extent but who stayed in gym have higher proportion users spending more than 
# average additional spending compared to the proportion in users who left.
# 
# > Users who continue to using gym have higher propertion users with longer contract period compared to who left the gym
# 
# > users who continue to use gym have more propportion of users have higher frequency than average in total compared to proportion of those 
# who left. The proportion of users frequency was even larger in preceeding month
# 
# > Contract period, lifetime and age seems to have more in those who continue to use the gym.
# 
# 

# In[16]:



dataset = fitness[['gender', 'near_location', 'partner', 'promo_friends', 'phone',
         'group_visits',   'churn']]

#Distribution:
    
fig = plt.figure(figsize=(15, 17))
plt.suptitle('Distribution of categorical  Columns\n',horizontalalignment="center",fontstyle = "normal", fontsize = 24, fontfamily = "sans-serif")
for i in range(dataset.shape[1]):
    plt.subplot(3, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset.columns.values[i])
    
    sns.distplot(dataset[dataset['churn']==0].iloc[:, i], color = 'green')
    sns.distplot(dataset[dataset['churn']==1].iloc[:, i], color = 'red')


# More proportion of users from neighborhood stayed using the gym compared to those who are not living or working in neighbourhood
# . Likewise, more proportion of users who are emplyoed in partner company stayed using gym compared to those sign up through a 
# "bring a friend" offer.  Also,
# less number of people who participate in group session left the gym compared to those who dont participate in group session. 
# Likewise, who stayed using gym has  more additional spending, more month to end contract, longer lifetime, 
# avg_additional_charges_total, churn than who left the gym.
# 
# > More people who sign up through a "bring a friend" offer seems to have higher churn rate than continuing to use the gym. Gender doesnot seem to 
# have any impact on churn rate.

# In[17]:


corr = fitness.corr()
corr=corr.round(2)


# In[18]:


sns.set(style='white')
plt.figure(figsize=(13, 9))
plt.title('Cohorts: User Retention')

sns.heatmap(corr, annot=True)


# > gender and providing phone number have no relation with churn rate.
# 
# >Age, contract period and average class frequency in precceding month have comparatively strong correlation with churn rate. Likewise 
#  month remain to end contract period also have stronger corelation with churn rate

# <div class="alert alert-success">
#     <b>Reviewer's comment</b><br>
#     Great, you explored the data and made some interesting findings! Visualizations are used appropriately.
# </div>

# ### Step 3. Build a model to predict user churn

# In[19]:


X = fitness.drop('churn', axis = 1)# write your code here
y = fitness['churn']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[21]:


# create a StandardScaler object and apply it to the train set
scaler =StandardScaler()# write your code here
X_train_st = scaler.fit_transform(X_train) # train the scaler and transform the matrix for the train set

# apply standardization to the feature matrix for the test set
X_test_st = scaler.transform(X_test)


# In[22]:


models = [RandomForestClassifier(n_estimators=100, random_state=0), LogisticRegression(random_state=0)]


# In[23]:


def metrics(y_test, y_pred):
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Precision: {:.2f}'.format(precision_score (y_test, y_pred)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))


# In[24]:


def make_prediction(m, X_train, y_train, X_test, y_test):
    model = m
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics(y_test, y_pred)
    print('\n')
    
   


# In[25]:


for i in models:
    print(i)
    make_prediction(m=i,X_train = X_train_st, y_train= y_train,
                    X_test=X_test_st, y_test = y_test)
    


# > Logistic regression classification has slightly higher precision and Recall compared to random forest classifier in the given dataset

# <div class="alert alert-warning">
#     <b>Reviewer's comment</b><br>
#     The data was split into train and test. The models were trained and evaluated correctly. I have question for you to think about. What metric is more important in our case: precision or recall? What is worse: if someone is likely to leave and the model marks them as not likely to leave (false negative) or someone is not likely to leave and the model marks them as likely to leave (false positive). Higher recall means lower false negatives, while higher precision means lower false positives.
# </div>

# ### Step 4. Create user clusters

# In[26]:


X_sc = scaler.fit_transform(X)


# In[27]:


linked = linkage(X_sc, method = 'ward')


# In[28]:


plt.figure(figsize=(15, 10))  
dendrogram(linked, orientation='top')
plt.title('Hierarchical clustering for GYM')
plt.show() 


# > Dendogram shows four different clusters suggested by four colors and  4 level of branch ( distance) from each cluster.  since
# branch with purple color have many users and it also can be divided in two clusters

# In[29]:


km = KMeans(n_clusters = 5) # setting the number of clusters as 5
labels = km.fit_predict(X_sc) # applying the algorithm to the data and forming a cluster vector 


# In[30]:


fitness['clusters'] = labels
 
# get the statistics of the mean feature values per cluster
data_grouped = fitness.groupby('clusters').mean().reset_index()
 
# print the grouped clusters
data_grouped


# >Contract_period, partner, promo_friends, month to end contract have high variation in feature values among the clusters

# In[31]:


pd.DataFrame(fitness['clusters'].value_counts())


# In[32]:


fig = plt.figure(figsize=(15, 25))
plt.suptitle('Distributions of features for the clusters \n',horizontalalignment="center",fontstyle = "normal", fontsize = 24, fontfamily = "sans-serif")
for i in range(fitness.shape[1]):
    plt.subplot(5, 3, i + 1)
    f = plt.gca()
    f.set_title(fitness.columns.values[i])    
    sns.distplot(fitness[fitness['clusters']==0].iloc[:, i], color = 'green',  kde_kws={"color": "green", "lw": 2, "label": "0"})
    sns.distplot(fitness[fitness['clusters']==1].iloc[:, i], color = 'red',  kde_kws={"color": "red", "lw": 2, "label": "1"})
    sns.distplot(fitness[fitness['clusters']==2].iloc[:, i], color = 'k',  kde_kws={"color": "k", "lw": 2, "label": "2"})
    sns.distplot(fitness[fitness['clusters']==3].iloc[:, i], color = 'blue',  kde_kws={"color": "blue", "lw": 2, "label": "3"})
    sns.distplot(fitness[fitness['clusters']==3].iloc[:, i], color = 'yellow',  kde_kws={"color": "yellow", "lw": 2, "label": "4"})
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])   


# > All clusters are symmetrically distributed among both gender but these clusters varied in size.
# 
# > Based on feature contract period, cluster '2','4' and '0' has 1 month , 6 month and 12 month contract period. Month to end contract 
# also have similar varition among clusters.
# 
# > Average class frequency and average total frequencies aretwo features which differentiate cluster two, cluster 4 and remaining clusters
# 
# > phone is one feature which contibute to differentiate cluster '0' and remaining clusters

# In[33]:


print(fitness.groupby('clusters')['churn'].mean())


# By far highest loyality can be seen in cluster 3. cluster 4 users  followed by cluster 2 users are more prone to leave. The remaining cluster lies in middle of two

# <div class="alert alert-warning">
#     <b>Reviewer's comment</b><br>
#     Clusters were identified and studied. In the future I suggest setting the random_state parameter of KMeans to ensure reproducibility of your research. Otherwise you will get different clusters every time you run the code, because initial cluster centers are selected randomly.
# </div>

# ### Step 5. Come up with conclusions and basic recommendations on working with customers

# In[34]:


forest = ExtraTreesClassifier(n_estimators=250, criterion ='entropy', random_state=0)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(20,10))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[35]:


print(X_train.columns)


# **'month_to_end_contract' and 'avg_class_frequency_total' are most important features in determining churn rate.** 

# ### Conclusion

# > 'month_to_end_contract' and 'avg_class_frequency_total' are most important features in determining churn rate.
# 
# > about 84.5 % of users lives or works in the neighborhood where the gym is located
# 
# >More proportion of users from neighborhood stayed using the gym compared to those who are not living or working in neighbourhood. Likewise, more proportion of users who are emplyoed in partner company stayed using gym compared to those sign up through a "bring a friend" offer. people who didnot left have longer contract period on average compared to those who left the gym. Also , less number of people who participate in group session left the gym compared to those who dont participate in group session. Likewise, who stayed using gym has higher average age, more additional spending, more month to end contract, longer lifetime, avg_additional_charges_total, churn than who left the gym.
# 
# >gender and providing phone number have no relation with churn rate.
# 
# >Age, contract period and average class frequency in precceding month have comparatively strong correlation with churn rate. Likewise month remain to end contract period also have stronger corelation with churn rate
# 
# >**Recommendation** :
#     
# > 'Longer contract','avg_class_frequency_total', 'group_visits', 'age' have strong impact on churn rate. So, policy should be driven
# towards making longer contract, engaging users more days to gym, encouraging group visits and enrolling users with higher age. So, discounts
# and events can be offer to favor these features is recommended.

# <div class="alert alert-success">
#     <b>Reviewer's comment</b><br>
#     Nice summary! Recommendations make sense and are consistent with the data. Well done!
# </div>

# In[ ]:




