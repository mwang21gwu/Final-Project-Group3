train_1 = train.iloc[:,0:50]
train_2 = train.iloc[:,51:100]
train_3 = train.iloc[:,101:150]
train_4 = train.iloc[:,150:204]

msno.matrix(train_1,labels= True)
plt.show()
msno.matrix(train_2,labels = True)
plt.show()
msno.matrix(train_3,labels = True)
plt.show()
msno.matrix(train_4,labels = True)
plt.show()


obj_cols = list(combined_train_test.select_dtypes(include=['object']).columns)

for col in obj_cols:
     combined_train_test[col] = combined_train_test[col].fillna(value = combined_train_test[col].mode()[0])

num_cols = list(combined_train_test.select_dtypes(include = ['int64']).columns)
num_cols.append(list(combined_train_test.select_dtypes(include = ['float64']).columns))

for col in num_cols:
    combined_train_test[col] = combined_train_test[col].fillna(value = combined_train_test[col].mean())

combined_train_test = combined_train_test.drop('UserInfo_9',axis=1)

list_total_1 = list_total[list_total['target'].isin([1])]
list_total_0 = list_total[list_total['target'].isin([0])]
list_total_1 = list_total_1[['ListingInfo']]
list_total_0 = list_total_0[['ListingInfo']]

#change the date in 'object' type to 'date'
list_total_1 = pd.to_datetime(list_total_1['ListingInfo'],format='%Y %m %d')
list_total_0 = pd.to_datetime(list_total_0['ListingInfo'],format='%Y %m %d')

#count with the date
list_total_1 = list_total_1.value_counts()
list_total_0 = list_total_0.value_counts()

# remane the label of date and taget,which have been counted and grouped
list_total_1 = list_total_1.rename_axis('date').reset_index(name='counts')
list_total_0 = list_total_0.rename_axis('date').reset_index(name='counts')

#rank the date from samll to big
list_total_1 = list_total_1.sort_values(by = 'date')
list_total_0 = list_total_0.sort_values(by = 'date')

#plot the results
plt.figure()
plt.plot(list_total_1['date'],list_total_1['counts'],color = '#900302')
plt.plot(list_total_0['date'],list_total_0['counts'])
plt.title('combined_train_test set')
plt.legend(('overdue','good'))
plt.show()


#see the pattern of numbers of cities
plt.figure()
plt.plot(count_city_1['city'],np.log(count_city_1['counts']))
plt.show()
# from the plot we can see that we can get three categories
count_city_1['counts'] = count_city_1['counts'].apply(np.log)
print(count_city_1)

count_city_1['counts'] = count_city_1['counts'].astype(int)
print(count_city_1)

# from the plot we choose 3 and 4.5 as change point
for i in range(len(count_city_1['counts'])):
   if count_city_1['counts'][i] >= 4.5:
      count_city_1['counts'][i] = 3

   elif count_city_1['counts'][i] <3:
      count_city_1['counts'][i] = 1

   else:
      count_city_1['counts'][i] = 2
print(count_city_1)

#the 3 rank
col_3 = list(count_city_1[count_city_1['counts']==3]['city'])
col_2 = list(count_city_1[count_city_1['counts']==2]['city'])
col_1 = list(count_city_1[count_city_1['counts']==1]['city'])
print(col_3)
print(col_2)
print(col_1)


def dummy_change(data):
  dummy_UserInfo_22 = pd.get_dummies(data['UserInfo_22'],prefix='UserInfo_22')
  dummy_UserInfo_23 =pd.get_dummies(data['UserInfo_23'],prefix='UserInfo_23')
  dummy_UserInfo_24 =pd.get_dummies(data['UserInfo_24'],prefix='UserInfo_24')
  dummy_Education_Info2 =pd.get_dummies(data['Education_Info2'],prefix='Education_Info2')
  dummy_Education_Info3 =pd.get_dummies(data['Education_Info3'],prefix='Education_Info3')
  dummy_Education_Info4 =pd.get_dummies(data['Education_Info4'],prefix='Education_Info4')
  dummy_Education_Info6 =pd.get_dummies(data['Education_Info6'],prefix='Education_Info6')
  dummy_Education_Info7 =pd.get_dummies(data['Education_Info7'],prefix='Education_Info7')
  dummy_Education_Info8 =pd.get_dummies(data['Education_Info8'],prefix='Education_Info8')
  dummy_WeblogInfo_19 =pd.get_dummies(data['WeblogInfo_19'],prefix='WeblogInfo_19')
  dummy_WeblogInfo_20 =pd.get_dummies(data['WeblogInfo_20'],prefix='WeblogInfo_20')
  dummy_WeblogInfo_21 =pd.get_dummies(data['WeblogInfo_21'],prefix='WeblogInfo_21')


  data = pd.concat([data,dummy_UserInfo_22,dummy_UserInfo_23,dummy_UserInfo_24,dummy_Education_Info2
                   ,dummy_Education_Info3,dummy_Education_Info4,dummy_Education_Info6,dummy_Education_Info7
                   ,dummy_Education_Info8,dummy_WeblogInfo_19,dummy_WeblogInfo_20,dummy_WeblogInfo_21
                   ],axis = 1)
  return(data)

combined_train_test = dummy_change(combined_train_test)
print(combined_train_test)
print(combined_train_test.info())
#delete features that we dealt with before
delete_cols_1 = ['UserInfo_22','UserInfo_23', 'UserInfo_24', 'Education_Info2', 'Education_Info3', 'Education_Info4','Education_Info6','Education_Info7'
               ,'Education_Info8','WeblogInfo_19','WeblogInfo_20','WeblogInfo_21']
for i in delete_cols_1:
    combined_train_test= combined_train_test.drop(i, axis=1)

print(combined_train_test)

def get_dataset():
   X, y = make_classification(n_samples=3000, n_features=50, n_informative=15, n_redundant=5)
   return X, y

def get_stacking():
   # define the base models
   level0 = list()
   level0.append(('lr', LogisticRegression()))
   level0.append(('knn', KNeighborsClassifier()))
   level0.append(('cart', DecisionTreeClassifier()))
   level0.append(('svm', SVC()))
   level0.append(('bayes', GaussianNB()))
   level0.append(('nn',MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                   solver='lbfgs', alpha=0.9, batch_size='auto',
                   learning_rate='constant')))
   # define meta learner model
   level1 = XGBClassifier(learning_rate= 0.1,
                     n_estimators= 1000,
                     max_depth= 10,
                     min_child_weight= 1,
                     subsample= 0.7,
                     colsample_bytree= 0.9,
                  )
   # define the stacking ensemble
   model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
   model.fit(X, y)
   return model
we choose the logisticregression, knn, decisiontree, svm, neuron network, bayes,xgb, also, we use the10-fold cross-validation, also we can plot the result with the box-plot.

# get a list of models to evaluate
def get_models():
   models = dict()
   models['lr'] = LogisticRegression()
   models['knn'] = KNeighborsClassifier()
   models['cart'] = DecisionTreeClassifier()
   models['svm'] = SVC()
   models['bayes'] = GaussianNB()
   models['nn'] = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                   solver='lbfgs', alpha=0.9, batch_size='auto',
                   learning_rate='constant')
   models['xgb'] = XGBClassifier(learning_rate= 0.1,
                     n_estimators= 1000,
                     max_depth= 10,
                     min_child_weight= 1,
                     subsample= 0.7,
                     colsample_bytree= 0.9,
                    )
   models['stacking'] = get_stacking()
   return models
# evaluate a give model using cross-validation
def evaluate_model(model):
   cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
   scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
   return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
   scores = evaluate_model(model)
   results.append(scores)
   names.append(name)
   print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

predict = models['stacking'].predict(test_data_X)
print(predict)
predict = pd.DataFrame(predict)
predict.to_csv('C:/project/data/predict.csv')
Conclude:

