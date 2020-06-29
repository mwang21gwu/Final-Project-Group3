#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#### encoding='gb18030' deal with Chinese ###
train = pd.read_csv('D:/Download/PPD_Training_Master_GBK_3_1_Training_Set.csv', encoding='gb18030')
test = pd.read_csv('D:/Download/test.csv', encoding='gb18030')
combine = pd.concat([train,test])


# target 1,0 ratio, we can find from the graph that the ratio is unbalance. we should deal with it later
combine['target'].value_counts().plot.pie(autopct = '%1.2f%%')
plt.show()

def comulation(col):
    combine.groupby([col,'target'])['target'].count()
    combine[[col,'target']].groupby(['target']).mean().plot.bar()
    title = 'Overdue vs '+ col
    plt.title(title)
    plt.xlabel(['Not overdue', 'Overdue'])
    plt.show()

# 'Education_Info1','Education_Info5' overdue information
edu = ['Education_Info1','Education_Info5']
for i in edu:
    comulation(i)

def count_missing(first_col, last_col):
    new = combine.loc[:,first_col:last_col]
    cols = list(new.columns)
    count = [combine[col].isnull().sum() for col in cols]
    miss = pd.DataFrame({'cols':cols, 'count':count})
    print(miss)

## all the count of education is 0, shows that the education information is really
## significant for decision making
count_missing('Education_Info1','Education_Info8')


# show the number of missing data
missing=train.isnull().sum().reset_index().rename(columns={0:'missNum'})
# caculate the rate of missing data
missing['missRate']=missing['missNum']/train.shape[0]
# show by sort
miss_analy=missing[missing.missRate>0].sort_values(by='missRate',ascending=False)

fig = plt.figure(figsize=(18, 6))
plt.bar(np.arange(miss_analy.shape[0]), list(miss_analy.missRate.values), align='center',
        color=['red', 'green', 'yellow', 'steelblue'])

plt.title('Histogram of missing value of variables')
plt.xlabel('variables names')
plt.ylabel('missing rate')
plt.xticks(np.arange(miss_analy.shape[0]), list(miss_analy['index']))
plt.xticks(rotation=90)

for x, y in enumerate(list(miss_analy.missRate.values)):
    plt.text(x, y + 0.12, '{:.2%}'.format(y), ha='center', rotation=90)
plt.ylim([0, 1.2])

plt.show()


