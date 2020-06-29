#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import missingno as msno
import math
import re
import datetime
from collections import Counter
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import os,random
import operator
from sklearn.preprocessing import OneHotEncoder
train_url = 'C:/project/Training Set/PPD_Training_Master_GBK_3_1_Training_Set.csv'
train_df_org = pd.read_csv(train_url, encoding='gb18030')
test_url = 'C:/project/Training Set/test.csv'
test_df_org = pd.read_csv(test_url, encoding='gb18030')
print(train_df_org)
print(test_df_org)

combined_train_test = train_df_org.append(test_df_org)
print(combined_train_test)
### plot columns with over 70% missing value ###
def plot_missing_values(df):
    delete_percent = 0.6
    cols = df.columns
    count = [df[col].isnull().sum() for col in cols]  # isnull will generate true,false matrix
    percent = [i / len(df) for i in count]
    missing = pd.DataFrame({'number': count, 'proportion': percent}, index=cols)  # create dataframe
    fig = plt.figure(figsize=(20, 7))

    missing_count = list(zip(cols, count, percent)) #join the data set

    null_col = []
    null_percent = []
    for i in range(len(missing_count)):
        if missing_count[i][1] != 0 and missing_count[i][2] > delete_percent:
            null_col.append(missing_count[i][0])
            null_percent.append(missing_count[i][2])
    plt.bar(null_col, null_percent)
    plt.xlabel('Feature')
    plt.ylabel('Percent')
    title = 'Columns with ' + str(delete_percent * 100) + '% missing value'
    plt.title(title)
    plt.show()
    print(null_col)
    return null_col

if __name__ == '__main__':
    train_delete_missing = plot_missing_values(combined_train_test)


# delete column based on missing value
combined_train_test = combined_train_test.drop(train_delete_missing, axis=1)


### calculate std for every numerical column ###
def standard_d(df):
    cols = df.columns
    delete_std = 0.1
    st_d = []
    for col in cols:
        if df[col].dtypes in ('float64', 'int64'):
            st_d.append(df[col].std())
        else:
            st_d.append('none')
    name_w_std = list(zip(cols,st_d))
    delete_list = []
    for i in range(len(name_w_std)):
        if name_w_std[i][1] != 'none' and name_w_std[i][1] < delete_std:
            delete_list.append(name_w_std[i][0])
    return delete_list


if __name__ == '__main__':
    train_delete_std = standard_d(combined_train_test)

    combine = list(set(train_delete_std))

# delete column based on std
combined_train_test = combined_train_test.drop(combine, axis=1)


###count the missing value of every feaure and sort the number of missing value###
def count_missing(df):
    cols = df.columns
    count = [df[col].isnull().sum() for col in cols]
    count = sorted(count)
    x = range(len(count))
    plt.scatter(x, count)
    plt.title('test set')
    # plt.ylim(0, 170)
    plt.xlabel('Order Number(sort increasingly)')
    plt.ylabel('Number of Missing Attributes')
    plt.show()

if __name__ == '__main__':
    count_missing(combined_train_test)



# ##---------------------------------------------------------------------
#
#
### fillin na value of float with mean, while object value with mode ###
obj_cols = list(combined_train_test.select_dtypes(include=['object']).columns)

for col in obj_cols:
     combined_train_test[col] = combined_train_test[col].fillna(value = combined_train_test[col].mode()[0])

num_cols = list(combined_train_test.select_dtypes(include = ['int64']).columns)
num_cols.append(list(combined_train_test.select_dtypes(include = ['float64']).columns))

for col in num_cols:
    combined_train_test[col] = combined_train_test[col].fillna(value = combined_train_test[col].mean())



### data cleaning on city's name "重庆"="重庆市", keep the first city, remove others 3 ###
def clean_city(df, col):
    for i, name in enumerate(df[col]):
        name = list(name)
        if '市' in name:
            index = name.index('市')
            name = name[0:index]
            df[col].iloc[i] = ''.join(name)

info_city = ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']
for i in info_city:
    clean_city(combined_train_test,i)



# delete '中国联通', cause this is a feature like AT&T
#People using different telecom operators have nothing to do with repayment
combined_train_test = combined_train_test.drop('UserInfo_9',axis=1)


# def function to view overdue rate of different cities
def overdue_rate_city(data, num):
    new = pd.DataFrame({'state/city': data, 'target': combined_train_test.target})
    grouped = new.groupby('state/city')
    count = grouped['target'].agg(np.sum) / grouped['target'].agg(np.size)
    ordered = count.sort_values(ascending=False)
    slicing = ordered[0:num]
    print(slicing)

### see the top 6 overdue rate of citys and state ###
if __name__ == '__main__':
    # num= first num city , means top 6 overdue rate citys
    num = 6
    #overdue rate for every state
    overdue_rate_city(combined_train_test.UserInfo_19, num)
    overdue_rate_city(combined_train_test.UserInfo_7, num)
    #overdue rate for every city
    overdue_rate_city(combined_train_test.UserInfo_2, num)
    overdue_rate_city(combined_train_test.UserInfo_4, num)
    overdue_rate_city(combined_train_test.UserInfo_8, num)
    overdue_rate_city(combined_train_test.UserInfo_20, num)

## overdue rate cutoff = 0.85, we decide to focus on 7 provinces (天津 山东 湖南 辽宁 四川 吉林 海南)
def city_or_not(df, prov_name, col):
    title = prov_name + '_or_not_' + col[-1]
    df[title] = 0
    for i in range(len(df[col])):
        if prov_name in df[col].iloc[i]:
            df[title].iloc[i] = 1
    print(df[title].agg(np.sum)) # how many observation in that prov

if __name__ == '__main__':
    prov_names = ['天津','山东', '湖南', '辽宁', '四川', '吉林', '海南']
    cols = ['UserInfo_7', 'UserInfo_19']
    for i in prov_names:
        for j in cols:
            city_or_not(combined_train_test, i, j)


    print(combined_train_test.head())



# find all the unique city
def uniq_city(col):
    train_uniq = list(combined_train_test[col].unique())
    combine = list(set(train_uniq))
    return combine

if __name__ == '__main__':
    all_city = []
    info_city = ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']
    for i in info_city:
        partial_uniq_city = uniq_city(i)
        all_city = list(set(all_city + partial_uniq_city))
    print(all_city)
    print(len(all_city)) # 423 different cities in four features


# give rank of the city instead of name
def rank_city(df, col):
    title = col + '_rank'
    df[title] = 3
    rank_1 = ['北京', '上海', '广州', '深圳']
    rank_2 = ['成都', '杭州', '武汉', '天津', '南京', '重庆', '西安', '长沙', '青岛', '沈阳', '大连', '厦门', '苏州', '宁波', '无锡']
    for i, name in enumerate(df[col]):
        if name in rank_1:
            df[title].iloc[i] = 1
        elif name in rank_2:
            df[title].iloc[i] = 2
    new_num_diff_city = len(np.unique(df[title]))
    print(new_num_diff_city)  # now 395

info_city = ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']
for i in info_city:
    rank_city(combined_train_test, i)

print(combined_train_test)


### see the overdue change with time of train data set###
#combine taget nad date into one dataframe
list_total = combined_train_test[['ListingInfo','target']]
print(list_total)

#group by the overdue and un-overdue
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


#from the plot we can see there exist some relationship between time and target
#so we change the date to a feature, for 1-10 equal to 1, 11-20 eual to 2.....untill 3000

#change the type of date for train
print(train_df_org['ListingInfo'].head())
print(test_df_org['ListingInfo'].head())
print(combined_train_test['ListingInfo'].head(50))
combined_train_test['ListingInfo'] = pd.to_datetime(combined_train_test['ListingInfo'], format='%Y %m %d')
    # sort the whole data by date
combined_train_test = combined_train_test.sort_values(by=['ListingInfo'])
    # create a new dataframe, which represent date
index = list(range(0, len(combined_train_test)))
print(len(combined_train_test))
print(index)
for i in index:
    index[i] = math.ceil(i / 10)
index.remove(0)
index.append(math.ceil(len(combined_train_test)/ 10))
index = pd.DataFrame(index, columns=["date"])
index = pd.DataFrame(index)
combined_train_test = pd.DataFrame(combined_train_test)
print(index)
print(combined_train_test)
index.reset_index(drop=True, inplace=True)
combined_train_test.reset_index(drop=True, inplace=True)
combined_train_test = pd.concat([combined_train_test, index], axis=1)

#after we create a new date, we delete the old date
combined_train_test = combined_train_test.drop('ListingInfo',axis=1)
print(combined_train_test)


## test whether the cities are the same in two features
def len_name(col):
    if len(col) == 10:
        return col[-1]
    else:
        return col[-2:]

def test_same(df,col1,col2):
    first_num = len_name(col1)
    second_num = len_name(col2)
    title = 'diff_'+first_num+second_num
    df[title] = 0
    for i in range(len(df[col1])):
        if df[col1].iloc[i] == df[col2].iloc[i]:
            df[title].iloc[i] = 1

data = [combined_train_test]
for i in data:
    test_same(i,'UserInfo_2','UserInfo_4')
    test_same(i,'UserInfo_2','UserInfo_8')
    test_same(i,'UserInfo_2','UserInfo_20')
    test_same(i,'UserInfo_4','UserInfo_8')
    test_same(i,'UserInfo_4','UserInfo_20')
    test_same(i,'UserInfo_8','UserInfo_20')

print(combined_train_test.head(10))


###view the city, and count the numbers of city.then change it into 6 different level of category by number of cities
def train_city_count(data,label):
  list = data.loc[:,label].value_counts()
  list = list.rename_axis('city').reset_index(name='counts')
  return list

count_city_1 = train_city_count(combined_train_test,'UserInfo_2')
print(count_city_1)
print(count_city_1.head(20))
count_city_2 = train_city_count(combined_train_test,'UserInfo_4')
count_city_3 = train_city_count(combined_train_test,'UserInfo_8')
count_city_4 = train_city_count(combined_train_test,'UserInfo_20')
count_city = pd.concat([count_city_1,count_city_2,count_city_3,count_city_4],axis = 0)
count_city = count_city.groupby('city').sum()
count_city = pd.DataFrame(count_city)
count_city = count_city.sort_values(by = 'counts',ascending= False)
print(count_city)
print(count_city.head(20))

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

# # change the city name into 0, 1
def change_number(data,correspond):
    data['number_rank_1'] = 0
    data['number_rank_2'] = 0
    data['number_rank_3'] = 0
    for i, name in enumerate(data[correspond]):
        if name in col_1:
            data['number_rank_1'].iloc[i] = 1
        elif name in col_2:
            data['number_rank_2'].iloc[i] = 1
        else:
            data['number_rank_3'].iloc[i] = 1

change_number(combined_train_test, 'UserInfo_2')






# delete features that we dealt with before

delete_cols = ['Idx','UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20', 'UserInfo_7','UserInfo_19']
for i in delete_cols:
    combined_train_test = combined_train_test.drop(i, axis=1)

print(combined_train_test.head(10))


object = combined_train_test.select_dtypes(include='object')
print(object.columns)

# print(train['UserInfo_22'])
# print(train['UserInfo_23'])
# print(train['UserInfo_24'])
# print(train['Education_Info2'])
# print(train['Education_Info3'])
# print(train['Education_Info4'])
# print(train['Education_Info6'])
# print(train['Education_Info7'])
# print(train['Education_Info8'])
# print(train['WeblogInfo_19'])
# print(train['WeblogInfo_20'])
# print(train['WeblogInfo_21'])
#
#
# print(train['UserInfo_22'].value_counts())
# print(train['UserInfo_23'].value_counts())
# print(train['UserInfo_24'].value_counts())
# print(train['Education_Info2'].value_counts())
# print(train['Education_Info3'].value_counts())
# print(train['Education_Info4'].value_counts())
# print(train['Education_Info6'].value_counts())
# print(train['Education_Info7'].value_counts())
# print(train['Education_Info8'].value_counts())
# print(train['WeblogInfo_19'].value_counts())
# print(train['WeblogInfo_20'].value_counts())
# print(train['WeblogInfo_21'].value_counts())


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




combined_train_test.to_csv('C:/project/data/combined_train_test.csv')


