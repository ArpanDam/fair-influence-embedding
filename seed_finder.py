# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 10:42:50 2024

@author: HP
"""


#import tensorflow as tf
#import torch
#from torch.utils.data import random_split,DataLoader
#import torchvision.transforms as transforms
#from torchvision.datasets import CIFAR10
#from torch import nn
#import numpy as np
#import torch.optim as optim
import random
#from random import uniform
import pickle
#from numpy.linalg import norm

#import matplotlib.pyplot as plt
#import embedding_capture_influenced
#import embedding_capture_influencial
#import matplotlib.pyplot as plt
#import math


#influencial_embed=embedding_capture_influencial.influencial_embeding()


#pickle.dump(dict_probability_of_influence, open('dict_probability_of_influence', 'wb'))

probability_of_influence=pickle.load(open("dict_probability_of_influence","rb"))


#influenced_multiple_hop=pickle.load(open("influenced_multiple_hop","rb"))

def func(k):
    seed=[]
    influenced_member=set()
    influenced_member_2=set()
    #dict1={} # key influencer and value number of members influenced
    #dict2={} # key influencer value list of influenced members
    while(len(seed)<k):
        
        dict1={} # key influencer and value number of members influenced
        dict2={} # key influencer value list of influenced members
        for key in probability_of_influence:
            if key not in seed:
                list1=[] # for each influencer
                
                mc=10
                simulation=0
                while simulation < mc:
                    
                    sum1=0   # for each simulation
                    for follower in probability_of_influence[key]:
                        
                        for edges in probability_of_influence[key][follower]: 
                            if (follower not in influenced_member_2): # checking if follower is already influenced
                            
                                if(random.uniform(0, 1)<edges[0]):
                                    sum1=sum1+1 # number of member influenced
                                    if simulation ==mc-1:
                                        if key in dict2:  # storing the influenced member in dict2
                                            list2=dict2[key]
                                            list2.append(follower)
                                            list2=set(list2)
                                            list2=list(list2)
                                            dict2[key]=list2
                                        else:
                                            list2=[]
                                            list2.append(follower)
                                            dict2[key]=list2
                    list1.append(sum1) # storing the number of member influenced in list1
                    simulation=simulation+1
                dict1[key]=sum(list1)/len(list1)      
                
                
            # sort dict1 and take the 1st key into seed    
        sorted_dict = dict(sorted(dict1.items(), key=lambda item: item[1], reverse=True))
        #if (len(seed)==4):
            #print("")
        first_key = next(iter(sorted_dict))
        seed.append(first_key)
        #print(seed)
       
        for follower in dict2[first_key]:
            influenced_member_2.add(follower) # infliuenced member 2 should contain all the member influenced for k-1 seed
        for m in influenced_member:
            influenced_member_2.add(m)
        for m in influenced_member_2:
            influenced_member.add(m)
    
    #return seed, len(influenced_member),influenced_member
    #return seed,influenced_member
    return seed
    
#result=func(5)
#print(result)

