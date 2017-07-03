# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:22:40 2017

@author: April
"""
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool

def loadRatingsFile(sliceFlag):
    header = ['user_id','item_id','rating']
    #movielens1M
    ratingsdf = pd.read_csv(r'D:\recommendation\new\Books.txt',sep=sliceFlag, names=header,index_col=False)
    return ratingsdf

sliceFlag="::"
ratings = loadRatingsFile(sliceFlag)    
n_users = ratings.user_id.unique().shape[0] 
n_items = ratings.item_id.unique().shape[0] 

def getPositivelinks(ratings):          
    user_positive_items ={}    
    for user in range(0,n_users):
        user_positive_items[user]=[]        
        filterUser = ratings[ratings.user_id==user+1] 
        positive = np.mean(filterUser)[2]           
        for line in filterUser.itertuples():
            if line[3] >= positive:
                user_positive_items.get(user).append(int(line[2])-1)    
    return user_positive_items


def genWeightMatrix(user_positive_items):
    positive_matrix = np.zeros((n_items, n_items))
    for user in user_positive_items.keys():
        items = user_positive_items.get(user)        
        for item in items:
            idx = items.index(item)
            for it in items[idx+1:]:
                positive_matrix[item][it]=positive_matrix[item][it]+1
                positive_matrix[it][item]=positive_matrix[it][item]+1
    return positive_matrix
   

def extractSubgraph(weightMatrix,theUserPositiveItems):
    subGraph = pd.DataFrame(weightMatrix).filter(theUserPositiveItems,axis=0)
          
    return subGraph

def getSubGraphEdges(subGraph,theUserPositiveItems):        
    
    subGraphEdges = []    
    for row in theUserPositiveItems:
        location = dict(subGraph.loc[row])         
        items = [k for k,v in location.items() if v>0]# filter subMatrix =0        
        for col in items:           
            subGraphEdges.append([row,col])
            if col in theUserPositiveItems:
                pos = theUserPositiveItems.index(col)
                subGraph.iloc[pos,row] = 0
            else:
                pass
    return subGraphEdges

def countCo_ratedOrder(edges):
    occurenceValue = {}        
    for item in range(0,n_items):
        cnt = 0
        for i in range(0,len(edges)):
            if (edges[i][1]==item):
                cnt +=1
        occurenceValue[item]=cnt
    return occurenceValue

def itemHaveEdges(WeightMatrix): 
     itemEdges ={}
     matrixdf = pd.DataFrame(WeightMatrix)
     for item in range(0,n_items):
        num = len([k for k,v in dict(matrixdf.loc[item]).items() if v>0])
        itemEdges[item]=num
     return itemEdges
 
def caculateEntropy(WeightMatrix,itemEdges):
    entropy = {}
    matrixdf = pd.DataFrame(WeightMatrix)
    for item in range(0,n_items):
        hasEdgesItem = [k for k,v in dict(matrixdf.loc[item]).items() if v>0]
        tempentropy =0
        for j in hasEdgesItem:
            normalized = float(matrixdf[item][j])/ itemEdges.get(j)
            tempentropy -= normalized * np.log2(normalized)
        entropy[item] = tempentropy
    return entropy

def recommendation(occurenceValue,entropy,threshold,topn):
    reldiv = {}
    result=[]
    occurenceDES = sorted(occurenceValue.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    '''
    for item,value in occurenceDES:
        if entropy.get(item)>threshold:
            if len(result)<topn:
                result.append(item)
            else:
                break
    '''     
    for item,value in occurenceDES:
        reldiv[item]=entropy.get(item)+value
    top = sorted(reldiv.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[:topn]
    for i in top:
        result.append(i[0]) 
    return result


if __name__ == '__main__':
    users = [2922, 937, 531, 2718, 1025, 237, 3012, 15, 3188, 1831, 1316, 3481, 415, 2877, 719, 2371, 2906, 893, 2934, 1758, 1278, 2575, 1615, 64, 3452, 2049, 3300, 2118, 3135, 2836, 1135, 1827, 325, 1448, 474, 1952, 2517, 3302, 766, 117, 1574, 336, 979, 2732, 1693, 2316, 3299, 2979, 1040, 910, 2553, 175, 737, 3375, 810, 3398, 1381, 2253, 2040, 1330, 2612, 3231, 3312, 1333, 2147, 2942, 3409, 1243, 774, 3184, 2065, 3269, 3037, 2135, 2981, 1516, 2531, 3057, 4, 2296, 2383, 2952, 897, 952, 2832, 262, 3207, 2098, 2775, 2570, 1380, 3139, 331, 1931, 1044, 859, 1475, 1134, 842, 725, 1580, 309, 2162, 438, 2567, 3208, 76, 1519, 1147, 2502, 56, 2692, 475, 885, 3200, 2792, 2194, 2412, 158, 2299, 686, 131, 315, 2786, 1083, 2848, 3022, 433, 546, 534, 3238, 1241, 2189, 1253, 2421, 1317, 2406, 1117, 2477, 3157, 3272, 2380, 1020, 1341, 1027, 101, 696, 3250, 1608, 949, 1924, 1813, 2021, 34, 2565, 1269, 623, 1318, 2133, 2984, 2428, 1930, 1115, 3495, 3258, 646, 165, 182, 543, 2278, 387, 1759, 712, 1834, 73, 515, 1992, 441, 337, 997, 3166, 527, 3203, 1015, 2589, 535, 2527, 3196, 1212, 1219, 2503, 2264, 1520, 566, 2032, 1572, 3096, 2047, 2568, 2627, 3419, 469, 2168, 3046, 3458, 2921, 710, 1403, 2824, 3159, 3122, 3306, 2947, 1052, 1401, 1733, 1339, 3263, 1089, 1836, 863, 2026, 247, 1999, 2715, 894, 1708, 2023, 2645, 1583, 59, 3336, 2933, 940, 2755, 2420, 2146, 1922, 332, 2184, 2711, 2198, 3267, 163, 1252, 1066, 1410, 2298, 923, 2871, 2215, 1794, 1363, 708, 1222, 1064, 2183, 10, 604, 777, 1175, 780, 2238, 265, 1350, 2171, 1329, 1665, 1691, 1787, 2924, 3024, 2748, 1681, 2727, 1639, 1272, 1291, 47, 329, 697, 1842, 2293, 1426, 961, 3443, 929, 2905, 2438, 3459, 927, 2324, 989, 814, 836, 1632, 2540, 1981, 16, 2128, 1377, 1885, 443, 2688, 1313, 1224, 211, 576, 2003, 792, 17, 1983, 549, 2491, 1640, 1026, 2456, 915, 1211, 124, 1290, 2731, 397, 3510, 3204, 2311, 1846, 2596, 1578, 647, 1859, 1518, 2584, 467, 2625, 503, 2180, 718, 2746, 2015, 2093, 1271, 1650, 1054, 1891, 1080, 3315, 2678, 1828, 652, 1084, 941, 1451, 3340, 214, 1112, 2272, 224, 2880, 954, 273, 212, 2446, 3460, 1294, 2100, 3107, 1778, 753, 3192, 2009, 2011, 502, 2354, 788, 1998, 2573, 3538, 2031, 2614, 579, 2360, 653, 3153, 2405, 1902, 1180, 116, 1436, 1331, 3324, 3559, 2158, 3051, 3053, 2057, 1959, 3352, 3234, 2805, 22, 1216, 389, 1822, 2861, 106, 1190, 1327, 2154, 42, 345, 2615, 3108, 176, 2379, 2222, 2230, 3514, 3130, 2966, 1903, 1984, 778, 2042, 1856, 148, 598, 1442, 1429, 827, 2033, 2056, 2328, 1536, 288, 1576, 2588, 3449, 142, 2929, 699, 3480, 3520, 2563, 2543, 2683, 2207, 292, 613, 890, 202, 648, 2580, 3519, 138, 2460, 3350, 2699, 856, 1092, 2261, 3, 518, 2887, 1550, 1752, 1173, 1647, 1986, 3417, 2348, 702, 3091, 1723, 2985, 1276, 1014, 3124, 1687, 373, 1043, 769, 2638, 2419, 2996, 743, 465, 217, 1110, 956, 839, 2352, 2007, 820, 994, 2851, 2855, 1198, 991, 2598]
    topN = [5,10,20,30,40,50,60]
    threshold = 100 
    #t1 = time.time()
    positive = getPositivelinks(ratings)
    #t2 = time.time()
    #print "positive",t2-t1
    weightMatrix = genWeightMatrix(positive)
    #t3 = time.time()
    #print "weightMatrix",t3-t2
    itemHaveEdges = itemHaveEdges(weightMatrix)
    #t4 = time.time()
    #print "itemHaveEdges",t4-t3
    shanno_entropy = caculateEntropy(weightMatrix,itemHaveEdges)
    #t5 = time.time()
    #print "shanno_entropy",t5-t4
    
    for user in users[:500]:
        print users.index(user)
        theUserPositiveItems = sorted(positive.get(user))
        subGraph =extractSubgraph(weightMatrix,theUserPositiveItems)  
        #t6 = time.time()
        #print "subGraph",t6-t5
        edges = getSubGraphEdges(subGraph,theUserPositiveItems)
        #t7=time.time()    
        #print "edges",t7-t6
        occurence = countCo_ratedOrder(edges)    
        #t8 = time.time()
        #print "occurence",t8-t7,
        for topn in topN:
            timefile = open("D:\\recommendation\\new\\graphRecBooksTop["+str(topn)+"]Time.txt",'a')
            recofile = open("D:\\recommendation\\new\\graphRecBooksTop["+str(topn)+"]Result.txt",'a')
            t1 = time.time()
            result = recommendation(occurence,shanno_entropy,threshold,topn)
            t2 = time.time()
            timefile.write(str(t2-t1))
            timefile.write("\n")
            recofile.write(str(result)+",")
            recofile.write("\n")
            timefile.close()
            recofile.close()
            #print str(topn),str(result)    
        #t9 = time.time()
        #print "recommendation",t9-t8
        #print "total",t9-t1
    print "finished!!!"