import math
import numpy as np
import os
import pandas as pd

# historical length of typhoon record using for 2D domain-expert knowledge building
forward_seq = 4
# forward_seq + residual_seq is a valid typhoon for pre-processing. it means that a typhoon's lifetime over 12 (3 days) is selected
residual_seq = 8
# predictive typhoon record length, 4 is predicted TI at the leading time of 24 hours
predict_seq_num = 4
# training year
trainYear = (2018, 2021)
# testing year
testYear = (2022,2022)

# # Typhoon Class
# In[3]:
class TyphoonHeader:

    def __init__(self, typhoonYear, tid):
        # the year that typhoon occurs
        self.typhoonYear = typhoonYear
        # typhoon id, identify a typhoon
        self.tid = tid
        # typhoon record id, identify a typhoon record
        self.typhoonRecordNum = 0
        # typhoon record list
        self.typhoonRecords = []

    def printer(self):
        print("tid: %d, typhoonYear: %d, typhoonYear: %s, typhoonRecordNum: %d" %
              (self.tid, self.typhoonYear, self.typhoonName, self.typhoonRecordNum))
        for typhoonRecord in self.typhoonRecords:
            typhoonRecord.printer()

# In[4]:
class TyphoonRecord:

    def __init__(self, typhoonTime, lat, long, wind, pres, totalNum):
        self.typhoonTime = typhoonTime
        self.lat = lat
        self.long = long
        self.wind = wind
        self.pres = pres
        # typhoon record id, identify a typhoon record
        self.totalNum = totalNum

    def printer(self):
        print("totalNum: %d, typhoonTime: %d, lat: %d, long: %d, wind: %d" %
              (self.totalNum, self.typhoonTime, self.lat, self.long, self.wind))

# # Functions For 2D Domain-Expert Knowledge Building
# In[5]:
def buildup_feature(typhoonRecordsList, tid, fileXWriter):
    for start in range(forward_seq, len(typhoonRecordsList) - predict_seq_num):
        # typhoon id
        # typhoon time YYYYMMDDYY
        # label
        # typhoon year YYYY
        strXLine = str(typhoonRecordsList[start].totalNum) + \
                   ',' + str(tid) + \
                   ',' + str(typhoonRecordsList[start].typhoonTime) + \
                   ',' + str(typhoonRecordsList[start + predict_seq_num].wind) + \
                   ',' + str(typhoonRecordsList[start].typhoonTime//1000000)
        
        # persistence factors
        strXLine += ',' + str(typhoonRecordsList[start].lat)
        strXLine += ',' + str(typhoonRecordsList[start].long)
        strXLine += ',' + str(typhoonRecordsList[start].pres)
        strXLine += ',' + str(typhoonRecordsList[start].wind)
        
        strXLine += ',' + str(typhoonRecordsList[start-1].lat)
        strXLine += ',' + str(typhoonRecordsList[start-1].long)
        strXLine += ',' + str(typhoonRecordsList[start-1].pres)
        strXLine += ',' + str(typhoonRecordsList[start-1].wind)
        
        strXLine += ',' + str(typhoonRecordsList[start-2].lat)
        strXLine += ',' + str(typhoonRecordsList[start-2].long)
        strXLine += ',' + str(typhoonRecordsList[start-2].pres)
        strXLine += ',' + str(typhoonRecordsList[start-2].wind)
        
        strXLine += ',' + str(typhoonRecordsList[start-3].lat)
        strXLine += ',' + str(typhoonRecordsList[start-3].long)
        strXLine += ',' + str(typhoonRecordsList[start-3].pres)
        strXLine += ',' + str(typhoonRecordsList[start-3].wind)
        
        strXLine += ',' + str(typhoonRecordsList[start-4].lat)
        strXLine += ',' + str(typhoonRecordsList[start-4].long)
        strXLine += ',' + str(typhoonRecordsList[start-4].pres)
        strXLine += ',' + str(typhoonRecordsList[start-4].wind)
        
        # climatology factors
        strXLine += ',' + str(typhoonRecordsList[start].lat-typhoonRecordsList[start-1].lat)
        strXLine += ',' + str(typhoonRecordsList[start-1].lat-typhoonRecordsList[start-2].lat)
        strXLine += ',' + str(typhoonRecordsList[start-2].lat-typhoonRecordsList[start-3].lat)
        strXLine += ',' + str(typhoonRecordsList[start-3].lat-typhoonRecordsList[start-4].lat)
        
        strXLine += ',' + str(typhoonRecordsList[start].long-typhoonRecordsList[start-1].long)
        strXLine += ',' + str(typhoonRecordsList[start-1].long-typhoonRecordsList[start-2].long)
        strXLine += ',' + str(typhoonRecordsList[start-2].long-typhoonRecordsList[start-3].long)
        strXLine += ',' + str(typhoonRecordsList[start-3].long-typhoonRecordsList[start-4].long)
        
        strXLine += ',' + str(typhoonRecordsList[start].pres-typhoonRecordsList[start-1].pres)
        strXLine += ',' + str(typhoonRecordsList[start-1].pres-typhoonRecordsList[start-2].pres)
        strXLine += ',' + str(typhoonRecordsList[start-2].pres-typhoonRecordsList[start-3].pres)
        strXLine += ',' + str(typhoonRecordsList[start-3].pres-typhoonRecordsList[start-4].pres)
        
        strXLine += ',' + str(typhoonRecordsList[start].wind-typhoonRecordsList[start-1].wind)
        strXLine += ',' + str(typhoonRecordsList[start-1].wind-typhoonRecordsList[start-2].wind)
        strXLine += ',' + str(typhoonRecordsList[start-2].wind-typhoonRecordsList[start-3].wind)
        strXLine += ',' + str(typhoonRecordsList[start-3].wind-typhoonRecordsList[start-4].wind)
        
        # brainstorm features
        strXLine += ',' + str(math.cos(typhoonRecordsList[start].lat))
        strXLine += ',' + str(math.cos(typhoonRecordsList[start-1].lat))
        strXLine += ',' + str(math.cos(typhoonRecordsList[start-2].lat))
        strXLine += ',' + str(math.cos(typhoonRecordsList[start-3].lat))
        strXLine += ',' + str(math.cos(typhoonRecordsList[start-4].lat))
        
        strXLine += ',' + str(math.cos(typhoonRecordsList[start].lat)-math.cos(typhoonRecordsList[start-1].lat))
        strXLine += ',' + str(math.cos(typhoonRecordsList[start].lat)-math.cos(typhoonRecordsList[start-2].lat))
        strXLine += ',' + str(math.cos(typhoonRecordsList[start].lat)-math.cos(typhoonRecordsList[start-3].lat))
        strXLine += ',' + str(math.cos(typhoonRecordsList[start].lat)-math.cos(typhoonRecordsList[start-4].lat))
        
        strXLine += ',' + str(typhoonRecordsList[start].wind**2)
        strXLine += ',' + str(typhoonRecordsList[start-1].wind**2)
        strXLine += ',' + str(typhoonRecordsList[start-2].wind**2)
        strXLine += ',' + str(typhoonRecordsList[start-3].wind**2)
        strXLine += ',' + str(typhoonRecordsList[start-4].wind**2)
        
        strXLine += ',' + str(2*math.log(typhoonRecordsList[start].wind))
        strXLine += ',' + str(2*math.log(typhoonRecordsList[start-1].wind))
        strXLine += ',' + str(2*math.log(typhoonRecordsList[start-2].wind))
        strXLine += ',' + str(2*math.log(typhoonRecordsList[start-3].wind))
        strXLine += ',' + str(2*math.log(typhoonRecordsList[start-4].wind))
        
        strXLine += ',' + str(typhoonRecordsList[start].wind**3)
        strXLine += ',' + str(typhoonRecordsList[start-1].wind**3)
        strXLine += ',' + str(typhoonRecordsList[start-2].wind**3)
        strXLine += ',' + str(typhoonRecordsList[start-3].wind**3)
        strXLine += ',' + str(typhoonRecordsList[start-4].wind**3)
        
        strXLine += ',' + str(typhoonRecordsList[start].pres/typhoonRecordsList[start].wind)
        strXLine += ',' + str(typhoonRecordsList[start-1].pres/typhoonRecordsList[start-1].wind)
        strXLine += ',' + str(typhoonRecordsList[start-2].pres/typhoonRecordsList[start-2].wind)
        strXLine += ',' + str(typhoonRecordsList[start-3].pres/typhoonRecordsList[start-3].wind)
        strXLine += ',' + str(typhoonRecordsList[start-4].pres/typhoonRecordsList[start-4].wind)
        
        strXLine += ',' + str(typhoonRecordsList[start].wind/typhoonRecordsList[start].pres)
        strXLine += ',' + str(typhoonRecordsList[start-1].wind/typhoonRecordsList[start-1].pres)
        strXLine += ',' + str(typhoonRecordsList[start-2].wind/typhoonRecordsList[start-2].pres)
        strXLine += ',' + str(typhoonRecordsList[start-3].wind/typhoonRecordsList[start-3].pres)
        strXLine += ',' + str(typhoonRecordsList[start-4].wind/typhoonRecordsList[start-4].pres)
        
        strXLine += ',' + str(typhoonRecordsList[start].pres/typhoonRecordsList[start].wind**2)
        strXLine += ',' + str(typhoonRecordsList[start-1].pres/typhoonRecordsList[start-1].wind**2)
        strXLine += ',' + str(typhoonRecordsList[start-2].pres/typhoonRecordsList[start-2].wind**2)
        strXLine += ',' + str(typhoonRecordsList[start-3].pres/typhoonRecordsList[start-3].wind**2)
        strXLine += ',' + str(typhoonRecordsList[start-4].pres/typhoonRecordsList[start-4].wind**2)
        
        strXLine += ',' + str(typhoonRecordsList[start].wind**2/typhoonRecordsList[start].pres)
        strXLine += ',' + str(typhoonRecordsList[start-1].wind**2/typhoonRecordsList[start-1].pres)
        strXLine += ',' + str(typhoonRecordsList[start-2].wind**2/typhoonRecordsList[start-2].pres)
        strXLine += ',' + str(typhoonRecordsList[start-3].wind**2/typhoonRecordsList[start-3].pres)
        strXLine += ',' + str(typhoonRecordsList[start-4].wind**2/typhoonRecordsList[start-4].pres)
        
        strXLine += ',' + str(typhoonRecordsList[start].pres/typhoonRecordsList[start].wind**3)
        strXLine += ',' + str(typhoonRecordsList[start-1].pres/typhoonRecordsList[start-1].wind**3)
        strXLine += ',' + str(typhoonRecordsList[start-2].pres/typhoonRecordsList[start-2].wind**3)
        strXLine += ',' + str(typhoonRecordsList[start-3].pres/typhoonRecordsList[start-3].wind**3)
        strXLine += ',' + str(typhoonRecordsList[start-4].pres/typhoonRecordsList[start-4].wind**3)
        
        strXLine += ',' + str(typhoonRecordsList[start].wind**3/typhoonRecordsList[start].pres)
        strXLine += ',' + str(typhoonRecordsList[start-1].wind**3/typhoonRecordsList[start-1].pres)
        strXLine += ',' + str(typhoonRecordsList[start-2].wind**3/typhoonRecordsList[start-2].pres)
        strXLine += ',' + str(typhoonRecordsList[start-3].wind**3/typhoonRecordsList[start-3].pres)
        strXLine += ',' + str(typhoonRecordsList[start-4].wind**3/typhoonRecordsList[start-4].pres)
        
        strXLine += ',' + str(math.log(typhoonRecordsList[start].pres))
        strXLine += ',' + str(math.log(typhoonRecordsList[start-1].pres))
        strXLine += ',' + str(math.log(typhoonRecordsList[start-2].pres))
        strXLine += ',' + str(math.log(typhoonRecordsList[start-3].pres))
        strXLine += ',' + str(math.log(typhoonRecordsList[start-4].pres))
        
        # month
        strXLine += ',' + str(typhoonRecordsList[start].typhoonTime//10000 % 100)
        
        fileXWriter.write(strXLine + '\n')

# # Read CMA Folder
# In[6]:
def read_cma_dfs(cma_list):
    totalNum = 1
    typhoonHeaderList = []
    for df in cma_list:
        tid = df.loc[0, 'TID']
        typhoonYear = df.loc[0, 'YEAR']
        typhoonHeader = TyphoonHeader(typhoonYear, tid, )
        # 读取文件
        for i in range(len(df)):
            typhoonTime = int(
                str(df.loc[i, 'YEAR'])+
                str(df.loc[i, 'MONTH']).zfill(2)+
                str(df.loc[i, 'DAY']).zfill(2)+
                str(df.loc[i, 'HOUR']).zfill(2)
            )
            lat = df.loc[i, 'LAT'] * 0.1
            long = df.loc[i, 'LONG'] * 0.1
            wind = df.loc[i, 'WND']
            pres = df.loc[i, 'PRES']
            typhoonRecord = TyphoonRecord(typhoonTime, lat, long, wind, pres, totalNum)
            totalNum += 1
            typhoonHeader.typhoonRecords.append(typhoonRecord)
        typhoonHeader.typhoonRecordNum = len(typhoonHeader.typhoonRecords)
        typhoonHeaderList.append(typhoonHeader)
    return typhoonHeaderList

# In[7]:
# 文件夹目录
path = "./CMA/"
files = os.listdir(path)
files.sort()

# 将所有CMABSTdata按文件读取
pd_list = []
for file in files:
    cma_pd = pd.read_csv(path+'//'+file, delim_whitespace=True,
                         names=['TROPICALTIME', 'I', 'LAT', 'LONG', 'PRES', 'WND' , 'OWD', 'NAME', 'RECORDTIME'])
    pd_list.append(cma_pd)

df = pd.concat(pd_list, axis=0)
print(df)

# 合并后重置索引
df = df.reset_index(drop=True)

# delete missing item
df = df.drop(columns=['OWD','RECORDTIME'])

# add typhoon id and typhoon record id
df = pd.concat([df, pd.DataFrame(columns=['TID','YEAR','MONTH','DAY','HOUR'])], axis=1)

# recoloumn
df = df[['TID','YEAR','MONTH','DAY','HOUR','TROPICALTIME', 'I', 'LAT', 'LONG', 'WND', 'PRES', 'NAME']]
print(df)

tid = 0
name = None
for i in range(0, len(df)):
    if df.at[i, 'TROPICALTIME'] == 66666:
        tid += 1
        name = df.loc[i, 'NAME']
    else:
        df.at[i, 'TID'] = tid
        df.at[i, 'NAME'] = name
        df.at[i, 'YEAR'] = df.loc[i, 'TROPICALTIME'] // 1000000
        df.at[i, 'MONTH'] = df.loc[i, 'TROPICALTIME'] // 10000 % 100
        df.at[i, 'DAY'] = df.loc[i, 'TROPICALTIME'] // 100 % 100
        df.at[i, 'HOUR'] = df.loc[i, 'TROPICALTIME'] % 100

df = df.drop(df[df['TROPICALTIME']==66666].index, axis=0)
df = df.drop(columns=['TROPICALTIME'])

# reindex
df=df.reset_index(drop=True)

df.loc[df['NAME']=='In-fa', 'NAME'] = 'Infa'
print(df)


df['KEY'] = None
years = df['YEAR'].unique()
years_dict = dict(zip(years, np.ones(years.shape)))
result_list = []
# search each typhoon
for tid in df['TID'].unique():
    temp_df = df[df['TID'] == tid].copy()
    # if a typhoon record includes two contiguous years, it will classified into the smaller year.
    tid_year = temp_df['YEAR'].unique()[0]
    cy = int(years_dict[tid_year])
    years_dict[tid_year] += 1
    temp_df['KEY'] = str(tid_year) + '-' + str(cy).zfill(2)
    result_list.append(temp_df)

df = pd.concat(result_list, axis=0)
df = df.reset_index(drop=True)
print(df)

# # Drop Duplicate
# #### in fact no duplicate record is found
# In[12]:
df = df.drop(df[~df['HOUR'].isin([0,6,12,18])].index, axis=0)
df = df.reset_index(drop=True)


# In[13]:
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print(df)

# In[14]:
df.to_csv('./cma_data/raw.csv', index=False)
print('./cma_data/raw.csv is saved!')


# Preprocessing
df = pd.read_csv('./cma_data/raw.csv')

tids = df['TID'].unique()
cma_list = []

for tid in tids:
    temp_df = df[df['TID']==tid]
    temp_df = temp_df.reset_index(drop=True)
    cma_list.append(temp_df)

# drop the invalid typhoon
valid_tropical_len = forward_seq + residual_seq
temp = []
for df in cma_list:
    if df.shape[0] >= valid_tropical_len:
        temp.append(df)
cma_list = temp
# print(cma_list)

# buiding training and testing
df = pd.concat(cma_list, axis=0)
df = df.reset_index(drop=True)
print(df)

df['number']=range(1,len(df)+1)
print(df)
df.to_csv('./cma_data/cma_list.csv', index=False)

print('./cma_data/cma_list.csv is saved!')


train_range = [str(x) for x in range(trainYear[0], trainYear[1]+1)]
train_keys = [v for i, v in enumerate(df['KEY'].unique()) if any(s in v for s in train_range)]

test_range = [str(x) for x in range(testYear[0], testYear[1]+1)]
test_keys = [v for i, v in enumerate(df['KEY'].unique()) if any(s in v for s in test_range)]

df = df[(df['KEY'].isin(train_keys)) | (df['KEY'].isin(test_keys))]
df = df.reset_index(drop=True)
print(df)

# In[18]:
tname = pd.read_csv('./cma_data/typhoon_name.csv')

# In[19]:
dict_name = {}
for i in range(len(tname)):
    dict_name[tname.at[i, 'en'].lower()] = tname.at[i, 'en']
dict_name['(nameless)']='missing'

# In[20]:
df['CN_NAME'] = None
for i in range(len(df)):
    try:
        df.at[i, 'CN_NAME'] = dict_name[df.at[i, 'NAME'].lower()]
    except KeyError:
        print(df.at[i, 'NAME'].lower())

# In[21]:
df.to_csv('./cma_data/pre_processing.csv', index=False)
print(df)
# In[22]:
typhoonHeaderList = read_cma_dfs(cma_list)

trainXFile = open('./cma_data/CMA_train_'+str(predict_seq_num*6)+'h.csv', 'w')
testXFile = open('./cma_data/CMA_test_'+str(predict_seq_num*6)+'h.csv', 'w')
for typhoonHeader in typhoonHeaderList:
    typhoonRecordsList = typhoonHeader.typhoonRecords
    if typhoonHeader.typhoonYear in range(trainYear[0], trainYear[1]+1):
        buildup_feature(typhoonRecordsList, typhoonHeader.tid, trainXFile)
    elif typhoonHeader.typhoonYear in range(testYear[0], testYear[1]+1):
        buildup_feature(typhoonRecordsList, typhoonHeader.tid, testXFile)

trainXFile.close()
testXFile.close()





