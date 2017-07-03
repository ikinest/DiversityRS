# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import math
from sklearn.cluster import KMeans
import time



def DisimlarityMatrixReclist(sim,reclist):
    leng = len(reclist)
    
    simreclist = np.zeros((leng, leng))
    for i in range(0,leng):
        for j in range(i,leng):
            if reclist[i]==reclist[j]:
                simreclist[i][j]=0.0
            else:
                simreclist[i][j]=sim[reclist[i]][reclist[j]]
                simreclist[j][i]=sim[reclist[i]][reclist[j]]
    return simreclist

def simlaryOfReclist(matrix):
    size = len(matrix)
    if size<=1:
        return 0
    else:
        return np.sum(matrix)/size/(size-1)
#new add

def CandinateItemSimilaryWithOthersInR(sim,candinateItem,R):
    simDF = pd.DataFrame(sim)
    similarysum = np.sum(simDF.loc[candinateItem][R])- 1.0
    return similarysum/len(R)

def overallSimilaryWithoutCandinateItem(sim,candinateItem):
    return np.sum(pd.DataFrame(sim).loc[candinateItem]) - 1.0

#end new add

def label_item_map(label):
    clusters = list(set(label))
    cnt = 0
    label_item ={}
    item_label ={}
    for cluster in clusters:
        label_item[cluster]=[]
        
    for i in label:
        label_item.get(i).append(cnt)
        item_label[cnt]=i
        cnt +=1
    return label_item,item_label


#��ȡ��ʼ�Ƽ�����Ʒ���Ե�����
def find_init_label(item_label,initidxs):
    initlabel=[]
    for item in initidxs:
        initlabel.append(item_label.get(item))
    return initlabel
  

 
#ͳ�Ƴ�ʼ�Ƽ�����������
def countInitlalbe(initlabel):
    initlabeNum = {}
    label = set(initlabel)
    count = 0
    for lab in label:
        for l in range(0,len(initlabel)):
            if lab == initlabel[l]:
                count +=1
        initlabeNum[lab]=count
        count = 0
    return initlabeNum
            

#ͳ��ÿ�������е���Ʒ��Ŀ
def countNumPerCluster(label_item):
    numPerCluster = {}
    for k in label_item.keys():
        numPerCluster[k] = len(label_item.get(k))-1
    return numPerCluster
   
def findMin(weightdict):
    minValueKey=[]
    minValue = sorted(list(set(weightdict.values())))[0]
    for k in weightdict.keys():
        if weightdict.get(k)==minValue:
            minValueKey.append(k)
        else:
            pass
    return minValueKey
    
def weightRejust1(InitlabelCount,threshod,numPerCluster):
    weight = InitlabelCount.copy()
    diffkeys = list(set(numPerCluster.keys()).difference(set(weight.keys())))
    for diff in diffkeys:
        weight[diff]=0
    for k in weight.keys():             
        while (weight.get(k)>threshod and len(findMin(weight))>0):
            minValueId = findMin(weight) 
            idx = random.choice(minValueId)
            if(numPerCluster.get(idx)>weight.get(idx)):
                weight[idx] = weight.get(idx) + 1
            else:
                minValueId.pop(minValueId.index(idx))#ɾ���ѵ���Ŀ����key
                if len(minValueId)>0:
                    idx1 = random.choice(minValueId)
                    weight[idx1] = weight.get(idx1) + 1
            weight[k] = weight.get(k) - 1
        print ("this label have been rejusted.")
    return weight

def weightRejust(InitlabelCount,threshod,numPerCluster):
    zerokey=[]
    weight = InitlabelCount.copy()
    diffkeys = list(set(numPerCluster.keys()).difference(set(weight.keys())))
    for diff in diffkeys:
        weight[diff]=0
        
         
    for k in weight.keys():             
        while( weight.get(k)>threshod and len(findMin(weight))>0):
            minValueId = findMin(weight) 
            idx = random.choice(minValueId)
            if(numPerCluster.get(idx)>weight.get(idx)):
                weight[idx] = weight.get(idx) + 1
            else:
                minValueId.pop(minValueId.index(idx))#ɾ���ѵ���Ŀ����key
                if len(minValueId)>0:
                    idx1 = random.choice(minValueId)
                    weight[idx1] = weight.get(idx1) + 1
            weight[k] = weight.get(k) - 1
    for ke in weight.keys():
        if weight.get(ke)==0:
            zerokey.append(ke)
    for zeroke in zerokey:
        weight.pop(zeroke)
    return weight

    
def WI(ri,zi):
    return math.exp(-(ri-zi)/ri)


def MaxsumDivIintMaxScore(weight,label_item,topk,aUserRatings):
    result = []
    ri = weight.copy()
    wi = {}
    zi = {}
    MaxConstraint = -999999999
    tempitem = -1
    templabel = -1
    initdict = [0,0]
    for k in ri.keys():
        wi[k]=0
        zi[k]=0 
        initscore = dict(aUserRatings[list(label_item.get(k))].sort_values(ascending=False).head(n=1)).values()[0]
        if(initscore > initdict[1]):
            initdict[0]=k
            initdict[1]=initscore
    InitResult = dict(aUserRatings[list(label_item.get(initdict[0]))].sort_values(ascending=False).head(n=1)).keys()[0]
    result.append(InitResult)
    initlabel = item_label.get(InitResult)
    zi[initlabel]= zi.get(initlabel)+1
    wi[initlabel]= WI( ri.get(initlabel) ,zi.get(initlabel) )     
    result_label = []
    counter = 1

  
    while(counter<topk):     

        for k in ri.keys():
                while (zi.get(k) < ri.get(k)):
                    dict_k = dict(aUserRatings[list(set(label_item.get(k)).difference(set(result)))].sort_values(ascending=False))
                    #print(dict_k)
                    temp = result
                    delete = None
                    
                    for dictkey in dict_k.keys():
                        #print("dict",dictkey)
                        temp.append(dictkey)
                        delete = dictkey                
                        disreclist = CandinateItemSimilaryWithOthersInR(sim,dictkey,temp) 
                        tempConstraint = (1-wi.get(k)*disreclist) * dict_k.get(dictkey)
                        if tempConstraint > MaxConstraint:
                            MaxConstraint = tempConstraint
                            tempitem = dictkey
                            templabel = k
                        temp.remove(delete)
                    result.append(tempitem)                
                    result_label.append(templabel)
                    zi[templabel]= zi.get(templabel)+1
                    wi[templabel]= WI( ri.get(templabel) ,zi.get(templabel) )
                    counter += 1
                    MaxConstraint = -999999999
            
    return result,result_label 

def cluDivRec(weight,item_label,topN,aUserRatings):
    result = []
    #tempweight = weight
    itemratings = aUserRatings.sort_values(ascending=False)
    #print "itemratings",itemratings
    while(len(result)<topN):
        for item in itemratings.index:
            if weight.get(item_label.get(item))>0:
                result.append(item)
                weight[item_label.get(item)]=weight.get(item_label.get(item))-1          
    
    return result

def cluDivRecOld(weight,label_item,aUserRatings):
    result = []
    for k in weight.keys():
        if weight.get(k)>0:
            dict_k = dict(aUserRatings[label_item.get(k)].sort_values(ascending=False).head(n=weight.get(k)))
            for dictkey in dict_k:
                result.append(dictkey)
        else:
            pass
    
    return result   

def paraCreate(topn):

    #for alg in preAlg.keys():
    #preAlg={'SVD':'MovieLensSVD.txt','IBCF':'MovieLensIBCF.txt','UBCF':'MovieLensUBCF.txt','ALS':'MovieLensALS.txt'}
    path = "D:\\recommendation\\new\\MovieLensALS.txt" 
    
    preScoredf = pd.read_csv(path, sep=',', names=header,index_col=False)
    als_prediction =preScoredf.as_matrix() #all_data_matrix


 
        #topn = int(topN)
    timefile = open("D:\\[Time]CluDivonMovielensThreshold3[BasedonALS]Top["+str(topn)+"].txt",'a')
    resultfile =   open("D:\\CluDivonMovielensThreshold3[BasedonALS]Top["+str(topn)+"].txt",'a')
    
    
    cnt = 0
    for user in users[:500]:
        cnt = cnt +1
        print (topn,cnt)
        t1 = time.time()
        aUserRatings = pd.DataFrame(als_prediction).loc[user]
        init = aUserRatings.sort_values(ascending=False).head(n=topn)
        initidxs = dict(init).keys()
    
        initlabel = find_init_label(item_label,initidxs)
    
        InitlabelCount = countInitlalbe(initlabel) 
        
        ALSuser20tims =[]
        result =[]
        for threshod in threshods:  
            weight = weightRejust(InitlabelCount,threshod,numPerCluster)            
            LGDivItem = cluDivRec(weight,item_label,topn,aUserRatings)
            result.append(LGDivItem)
            t2 = time.time()        
            ALSuser20tims.append(t2-t1)     
        resultfile.write(str(result)+","+"\n")
        timefile.write(str(ALSuser20tims)+","+"\n")  
    
    timefile.close()
    resultfile.close()
header1 = ['user_id', 'item_id', 'rating']

df = pd.read_csv('D://recommendation//new//movielens1M.txt', sep='\t', names=header1,index_col=False)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)) 

from sklearn import cross_validation as cv
dataset,_ = cv.train_test_split(df, test_size=0)
originalMatrix = np.zeros((n_users, n_items))
for line in dataset.itertuples():
    originalMatrix[int(line[1])-1, int(line[2])-1] = line[3]  

from sklearn.metrics.pairwise import pairwise_distances
cosine_distance = pairwise_distances(originalMatrix.T, metric='cosine')


sim = 1-cosine_distance




#based on similarity
label = np.array(
    [
        5,16,9,12,12,16,7,15,19,16,7,12,15,6,19,1,1,9,9,9,5,1,19,9,1,8,15,8,6,2,12,5,10,5,8,1,10,15,5,2,6,9,8,9,1,6,5,12,2,5,17,1,10,15,8,10,6,6,10,9,19,1,15,15,12,15,2,2,9,16,19,8,8,15,10,19,2,8,19,2,6,8,2,2,6,19,15,12,9,10,4,19,12,6,16,13,2,2,2,6,6,12,19,7,6,10,12,2,1,5,11,16,8,13,15,2,8,15,2,10,2,9,8,2,6,15,10,10,2,10,2,19,10,10,12,2,13,13,10,12,7,3,12,6,9,2,6,8,8,5,1,8,16,4,8,8,12,12,6,16,16,6,16,6,16,8,10,9,15,9,8,9,16,12,6,6,0,2,8,9,12,8,8,8,16,12,8,0,15,8,19,10,9,6,12,16,2,16,17,2,8,2,12,9,8,8,12,16,15,2,2,10,2,2,6,9,15,12,2,18,13,6,1,1,9,10,19,2,6,1,7,6,6,12,5,9,9,15,12,15,15,2,10,15,2,6,6,12,6,15,15,9,16,2,15,9,19,15,19,5,6,8,2,2,6,1,12,8,8,15,8,6,9,15,12,12,19,8,8,9,6,6,10,8,2,10,2,16,12,8,15,16,1,8,15,5,15,2,8,1,2,8,16,15,6,6,6,6,2,10,10,12,15,6,9,16,7,5,6,8,2,6,15,2,15,8,9,0,16,0,2,0,7,8,8,10,1,9,7,19,2,1,15,7,6,6,8,6,16,1,12,8,16,19,9,5,5,19,2,12,9,12,2,5,8,0,5,16,8,9,9,1,6,12,8,9,5,12,16,5,6,9,9,19,13,2,19,8,2,2,15,2,19,15,2,10,10,10,10,2,3,10,15,10,19,10,0,15,15,7,15,6,9,12,9,19,6,15,12,9,15,19,9,19,8,0,19,6,12,15,1,7,15,16,16,9,12,12,10,5,1,16,2,8,19,6,8,6,15,9,8,2,15,5,9,8,5,19,19,15,8,2,19,9,8,16,15,6,8,15,1,8,12,5,6,2,6,15,19,5,1,19,8,15,16,15,15,8,12,9,9,6,19,16,2,2,1,2,19,7,8,12,13,19,12,8,9,1,1,15,19,0,19,9,6,9,9,9,19,7,6,8,8,9,2,2,5,19,1,13,6,9,9,6,6,19,6,6,7,9,5,12,7,19,10,9,19,9,8,12,5,16,16,2,1,8,10,12,10,10,2,1,13,15,8,13,2,2,12,2,8,10,10,6,12,10,15,10,3,8,8,10,2,13,7,7,5,5,5,5,10,5,5,7,7,7,7,10,4,8,3,2,15,15,12,0,13,5,15,16,0,19,8,2,2,12,2,13,15,2,8,19,10,10,4,15,6,1,13,13,15,2,2,10,8,10,12,13,8,19,10,10,3,3,2,14,1,16,10,8,3,2,16,10,10,15,10,10,2,3,9,9,9,13,2,13,15,2,2,2,9,3,9,14,9,8,2,6,13,17,2,10,17,10,2,8,10,12,15,10,15,19,15,19,19,10,19,10,9,12,13,2,10,15,2,10,14,7,12,15,15,13,18,8,2,15,10,2,9,6,14,2,17,9,19,13,10,6,10,10,8,13,16,10,18,16,9,19,2,2,19,0,12,10,6,17,15,16,13,11,2,18,2,10,13,10,10,10,2,2,19,9,13,8,12,6,2,9,10,10,10,18,9,10,13,10,19,1,15,16,8,9,9,7,1,16,10,16,3,10,10,10,10,2,15,13,2,9,9,11,12,7,8,12,9,8,10,15,19,12,13,2,15,10,10,17,4,12,17,13,10,4,10,2,13,10,13,15,12,7,2,16,12,10,19,9,12,6,19,15,18,0,10,10,10,2,15,6,16,2,6,7,10,2,15,2,2,5,3,13,9,2,10,10,10,1,15,10,8,15,15,10,3,13,15,10,2,10,0,9,15,19,10,13,15,19,10,15,10,1,0,8,8,10,3,2,17,4,11,4,4,11,11,11,4,4,17,11,4,11,4,11,11,11,4,4,8,17,5,11,4,4,11,5,13,4,17,4,17,4,4,17,4,14,17,17,17,4,13,14,17,4,17,4,4,17,17,4,4,4,4,4,11,4,4,2,2,13,17,13,2,13,15,17,4,13,2,14,11,17,4,13,17,13,10,17,3,17,18,10,10,17,10,8,10,12,8,8,10,19,6,15,2,6,18,9,8,19,9,15,18,10,9,19,12,19,14,17,14,7,12,14,7,12,12,12,14,12,14,7,12,7,14,12,9,13,7,5,7,9,7,7,12,6,11,5,16,13,10,13,6,7,19,2,8,8,16,19,9,6,6,2,8,8,19,8,6,10,1,1,1,10,2,12,4,17,13,17,17,13,13,14,5,4,8,13,11,4,5,11,11,4,14,11,17,4,2,7,5,5,7,5,1,11,1,4,5,13,4,9,16,10,11,4,18,17,10,17,16,4,8,8,15,8,3,2,13,3,2,1,2,9,2,11,11,9,5,0,16,0,4,4,10,10,7,5,9,10,10,1,2,3,18,2,3,14,6,6,10,4,2,2,13,13,19,19,19,1,14,13,4,17,10,10,10,10,12,18,2,10,6,4,6,10,6,8,6,4,11,2,2,2,1,8,4,11,6,6,4,6,6,8,11,14,0,5,5,5,11,5,11,17,11,11,19,5,11,11,4,5,4,4,5,5,16,8,4,6,11,5,11,5,6,4,11,4,4,11,7,11,11,2,11,11,11,8,4,4,8,5,18,5,6,11,6,5,11,4,1,11,4,11,14,4,18,11,7,5,5,4,14,11,11,4,5,5,11,1,4,5,1,11,8,9,16,11,6,5,6,8,4,11,4,4,5,14,11,11,4,6,5,11,11,11,4,4,7,1,11,4,14,5,4,11,4,8,5,2,2,2,10,2,10,19,10,10,2,15,8,16,11,18,18,18,18,0,0,18,18,0,18,0,11,14,18,18,18,12,16,14,18,0,1,4,11,14,16,4,10,14,8,19,12,6,19,16,1,1,12,10,8,17,12,2,8,8,12,8,2,16,16,16,16,5,16,16,16,16,9,7,9,19,10,13,16,3,5,16,0,12,16,6,5,5,11,5,8,15,8,19,6,16,15,8,7,2,16,16,7,19,6,2,2,6,2,6,2,15,17,10,10,9,10,10,12,15,19,2,9,10,9,19,10,10,8,13,2,9,15,19,7,8,2,10,12,8,8,8,1,13,19,0,12,8,2,15,12,8,9,12,9,0,15,6,8,1,2,19,18,15,10,8,15,12,8,7,2,17,16,6,14,8,6,8,7,10,12,9,15,15,19,17,2,10,15,10,15,8,16,5,8,10,12,8,17,19,8,8,10,13,2,18,1,13,9,2,5,9,2,10,15,10,15,1,10,12,16,10,8,8,13,10,2,13,8,14,6,10,10,0,12,6,2,16,2,2,2,13,2,12,15,16,10,8,10,9,9,3,19,15,10,16,2,2,15,12,2,2,7,2,2,2,16,10,2,6,0,8,10,5,15,15,10,5,2,16,16,12,1,16,16,15,12,6,15,2,16,19,15,8,19,15,16,15,19,19,17,16,15,5,6,8,2,7,9,16,5,2,9,9,19,13,0,2,1,19,19,10,12,3,8,15,6,17,1,10,8,10,1,15,5,19,6,9,16,15,19,6,8,8,13,10,5,15,0,2,10,6,10,6,19,2,5,10,12,2,19,8,2,8,2,1,1,11,2,16,8,1,15,6,19,5,6,2,13,19,9,12,12,16,10,10,1,6,2,8,8,19,8,13,6,12,15,5,6,7,12,16,10,15,1,2,9,10,2,19,9,10,6,10,1,16,2,10,2,19,6,8,1,6,15,5,8,8,6,15,17,17,15,17,10,19,10,19,9,15,5,16,13,15,10,9,12,9,2,18,9,8,2,12,15,0,19,3,15,15,8,9,9,2,18,12,10,8,18,2,7,14,16,10,14,10,19,5,19,18,10,8,18,0,8,16,2,8,10,10,8,19,9,10,9,18,11,19,1,12,10,2,2,1,2,2,13,2,10,8,10,18,10,13,12,15,10,19,15,15,8,10,2,10,16,10,9,1,9,6,15,7,15,6,19,10,10,2,1,2,2,12,10,10,10,3,2,15,12,8,2,19,2,2,10,0,12,2,2,9,15,10,19,15,10,13,19,10,6,16,10,10,10,2,15,16,1,1,1,15,15,12,8,13,10,9,2,9,12,8,8,2,10,2,10,2,2,8,10,2,9,3,16,2,12,1,4,6,10,6,16,16,15,9,1,2,5,14,17,17,4,17,17,13,4,17,17,17,17,17,17,17,4,17,17,17,17,4,4,4,11,4,4,4,4,11,11,5,11,11,11,11,4,11,5,11,4,4,11,6,7,5,0,0,0,0,0,0,0,0,0,0,0,0,0,14,0,0,0,0,0,0,18,18,0,0,0,5,0,0,11,0,0,5,16,16,5,9,7,16,15,13,14,4,5,16,14,12,14,12,12,7,11,11,16,6,1,6,6,0,12,5,8,10,15,15,12,14,15,12,15,12,10,12,15,12,14,19,15,9,15,12,2,12,12,12,9,5,15,10,15,16,12,9,10,2,10,11,11,17,11,17,4,4,6,9,2,17,17,11,12,7,2,7,7,9,12,15,7,15,7,7,12,12,12,12,12,16,12,7,14,17,14,5,2,14,15,8,16,12,0,5,7,11,7,1,18,1,5,14,14,14,0,0,0,0,12,7,9,16,10,2,10,4,17,4,7,7,14,14,7,14,9,16,7,12,9,7,7,7,19,0,0,11,19,15,9,19,6,10,2,10,6,11,7,12,14,2,6,19,16,8,12,15,8,10,17,5,13,4,4,4,17,17,17,17,4,4,17,4,17,19,2,19,13,10,16,5,12,15,10,10,10,13,2,4,4,17,2,4,13,17,13,17,17,17,10,13,13,17,10,10,13,7,13,8,13,15,10,10,17,0,4,10,1,19,3,10,3,19,2,17,17,14,14,13,11,10,7,2,11,1,9,8,10,9,9,10,19,18,15,10,14,15,12,14,9,19,12,12,8,5,9,4,8,8,16,10,2,8,3,16,0,8,10,6,8,2,17,15,14,14,11,4,5,15,2,7,8,12,9,15,13,11,7,5,4,10,2,15,15,10,10,8,14,11,11,8,0,9,13,6,2,9,5,9,2,1,19,2,0,0,1,2,6,10,6,9,9,1,8,0,15,9,8,10,10,14,17,14,19,6,4,15,17,11,16,12,5,8,8,10,1,8,4,17,14,0,19,11,14,0,1,14,7,9,9,7,7,16,0,7,9,9,12,12,12,6,12,15,9,10,0,6,1,15,16,9,1,5,8,11,12,15,14,16,16,9,16,5,5,9,16,16,9,9,7,14,8,7,8,6,8,7,16,9,7,7,8,2,1,9,9,14,9,12,1,2,8,8,8,10,6,8,2,8,8,13,19,19,9,0,15,16,0,10,12,14,5,0,9,12,0,0,18,18,7,10,18,10,14,7,7,5,7,9,9,1,14,14,15,7,8,10,2,2,2,3,12,13,10,17,8,16,15,2,2,2,18,9,19,15,2,12,1,1,10,12,9,15,2,19,15,10,17,2,0,0,18,18,0,7,18,14,15,15,19,14,0,18,14,16,5,14,14,14,14,18,14,18,15,10,7,19,9,1,10,2,13,15,2,0,19,14,14,15,14,18,15,10,10,12,15,19,19,10,10,9,4,15,9,19,10,8,5,9,2,12,8,10,8,10,2,1,9,15,6,3,2,2,19,6,8,8,10,10,2,8,2,19,15,9,1,6,10,10,19,0,9,19,2,10,2,2,6,4,14,18,9,16,16,2,10,2,8,12,10,2,19,2,10,16,15,2,13,10,14,18,18,18,18,18,6,5,16,16,9,14,18,18,18,14,18,18,18,18,18,18,18,18,5,10,8,17,18,14,18,14,18,18,18,0,17,17,7,9,10,2,2,19,8,10,10,19,2,8,7,10,10,6,12,9,2,6,2,1,6,9,10,8,2,19,16,1,9,6,10,2,10,1,9,2,12,1,2,1,9,15,8,5,7,9,9,12,2,9,16,12,8,17,2,11,4,4,17,17,19,14,16,14,15,8,11,18,19,3,2,2,14,5,14,19,8,4,8,14,8,18,8,15,8,2,6,10,9,5,16,14,10,2,8,2,8,1,15,19,10,2,10,10,10,19,11,18,18,18,13,18,18,12,0,11,0,0,5,7,0,7,7,9,5,12,12,15,8,9,9,5,12,19,15,9,15,2,13,2,2,13,9,19,19,15,4,17,13,9,10,2,10,9,19,15,19,13,10,9,10,2,19,19,2,2,10,19,9,2,2,8,10,12,17,17,8,2,14,4,18,10,18,2,11,5,4,12,19,19,11,2,2,4,0,0,10,4,11,11,2,17,6,15,14,18,19,15,9,15,6,15,8,10,10,15,12,5,8,2,2,2,10,17,2,0,17,0,0,0,0,13,2,19,15,6,10,3,13,6,10,10,5,5,11,5,4,17,14,14,2,2,17,11,17,19,4,10,10,4,13,10,17,17,17,2,17,17,4,7,8,11,8,4,11,11,11,7,14,6,9,13,15,14,2,10,1,10,19,10,2,2,15,6,17,5,17,17,4,2,4,0,15,6,15,15,15,2,10,18,17,10,5,16,5,4,16,16,16,18,14,2,0,8,5,2,10,6,10,2,10,15,9,1,6,2,2,2,4,10,18,15,14,0,0,0,11,16,18,4,14,0,10,10,10,4,14,17,15,14,16,9,4,14,11,17,5,7,19,19,15,1,6,2,2,12,17,10,8,1,19,15,8,10,10,18,0,1,4,14,19,15,10,14,4,11,17,14,1,11,17,14,17,17,2,8,8,2,1,16,6,10,10,17,7,4,4,4,17,10,4,8,4,17,17,11,4,1,1,14,8,11,1,2,16,5,17,17,4,8,9,1,8,10,2,2,10,12,13,17,10,10,8,13,8,8,6,19,10,17,2,17,4,10,2,8,13,10,8,19,17,17,6,12,1,1,10,2,10,4,14,13,19,12,12,2,12,1,2,2,6,10,19,10,4,11,14,18,17,10,9,1,5,1,15,9,2,15,6,8,2,13,8,6,10,2,19,19,19,2,15,4,2,4,14,4,17,4,4,10,14,14,10,19,13,9,10,5,8,3,12,8,13,10,17,8,9,3,2,17,2,2,15,13,15,10,13,17,17,10,10,17,19,10,10,19,2,2,2,10,7,4,10,1,7,12,9,9,4,1,5,7,5,16,16,7,9,6,1,6,7,7,19,8,1,12,9,12,1,6,9,16,19,10,10,4,4,10,2,10,2,17,19,15,15,10,2,10,10,13,10,10,10,4,10,6,15,9,9,2,10,17,10,13,4,11,13,17,17,3,15,13,10,19,6,2,2,2,3,3,10,19,15,15,8,6,2,4,8,2,13,4,10,2,13,2,18,18,17,4,8,10,2,10,4,6,2,17,18,10,10,9,19,10,2,1,11,11,5,11,11,17,4,14,17,17,10,8,17,17,10,13,17,10,13,13,17,13,10,10,10,17,9,1,9,7,15,15,12,12,15,9,19,7,7,7,15,15,19,2,19,19,17,17,3,1,19,10,2,14,13,17,10,17,17,5,13,14,5,6,6,11,8,6,17,15,6,14,19,19,19,15,4,12,10,16,12,12,16,19,10,9,4,2,17,5,8,7,11,19,15,15,2,2,8,10,0,10,14,4,10,10,8,12,4,4,4,17,5,10,2,2,17,1,12,7,16,6,1,10,15,19,10,10,17,18,16,18,15,13,3,14,2,8,19,4,5,6,14,19,17,11,14,14,11,14,2,9,15,6,19,2,8,17,13,8,14,15,6,13,2,7,7,5,16,6,14,10,2,13,10,19,6,6,2,2,2,10,13,13,11,19,4,4,8,4,4,14,11,5,15,2,9,6,19,10,17,6,15,2,10,15,15,6,2,2,10,10,8,18,18,18,10,14,8,5,2,2,2,2,6,19,10,2,18,17,19,19,7,10,19,15,10,15,10,8,17,17,10,2,10,17,17,17,10,5,10,2,10,2,17,7,19,15,19,6,19,8,10,17,16,9,10,2,17,17,4,6,10,17,14,17,14,10,2,16,14,13,13,13,17,13,2,15,10,10,14,17,10,10,17,14,17,10,10,18,18,0,18,18,18,18,18,10,15,4,12,13,5,12,15,15,4,4,2,17,2,2,14,14,11,11,11,16,3,7,12,12,15,18,0,18,18,18,16,16,5,4,16,16,16,16,9,11

    ]

)



label_item,item_label = label_item_map(label)    
numPerCluster = countNumPerCluster(label_item)

users = random.sample([x for x in range(0,n_users)],500)

threshods = [3]

header = [x for x in range(0,n_items)]





from multiprocessing import Pool
if __name__ == '__main__':    
    TopN =[5,10,20,30,40,50,60]   
    pool = Pool(7)
    pool.map(paraCreate,TopN)
    pool.close()
    pool.join()