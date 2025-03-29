# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:38:02 2024

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

#import matplotlib.pyplot as plt
#import math
import seed_finder


import sys

if(len(sys.argv) == 3):
    
    r=int(sys.argv[2]) # Number of influence tags
    k=int(sys.argv[1]) # Number of Influencial users

if(len(sys.argv) != 3):
    
    k=5 # Number of influencial users
    r=2# Number of Influence tags


probability_of_influence=pickle.load(open("dict_probability_of_influence","rb"))
seed= seed_finder.func(k)




print("The top influencial members are",seed)
def remove_all_other_nodes(probability_of_influence,seed):
    # Keep only the seeds in the graph
    dict1={}
    tags=set()
    for key in seed:
        for follower in probability_of_influence[key]:
            for i in probability_of_influence[key][follower]:
                tags.add(i[1])
        
   
    for key in seed:
        dict1[key]= probability_of_influence[key]               
                #if follower in influenced_member:
                #dict1[key]=probability_of_influence[key]
        #print("")
    #print("")
    return dict1,tags
        
dict_only_seed_graph,tags=remove_all_other_nodes(probability_of_influence,seed)

def best_tags(dict_only_seed_graph,tags,r):
    best_tag=set()
    mc=20
    influenced_member_2=set()
    times=5
    times_going=0
    while len(best_tag)<r:
        
        dict_tag={} # key tag and value number of members influenced
        for tag in tags:
             # number of member influenced
            
            dict2={} # key influencer value list of influenced members
            list1=[] # for all influencer
            simulation=0
            while simulation < mc:
                sum1=0   # for each similation 
                for key in dict_only_seed_graph:
                        for key in dict_only_seed_graph:
                        
                            
                            for follower in probability_of_influence[key]:
                                
                                for edges in probability_of_influence[key][follower]: 
                                    if (edges[1]==tag) and (edges[1] not in best_tag):
                                        if (follower not in influenced_member_2): # checking if follower is already influenced
                                        
                                            if(random.uniform(0, 1)<edges[0]):
                                                sum1=sum1+1 # number of member influenced
                                                if simulation ==mc-1:
                                                    influenced_member_2.add(follower)
                                                    '''
                                                    if key in dict2:  # storing the influenced member in dict2
                                                        list2=dict2[key]
                                                        list2.append(follower)
                                                        list2=set(list2)
                                                        list2=list(list2)
                                                        dict2[key]=list2
                                                    else:
                                                        list2=[]
                                                        list2.append(follower)
                                                        dict2[key]=list2'''
                simulation=simulation+1                            
                list1.append(sum1) # storing the number of member influenced in list1
                        #simulation=simulation+1
            dict_tag[tag]=sum(list1)/len(list1) 
        sorted_dict = dict(sorted(dict_tag.items(), key=lambda item: item[1], reverse=True))
        #print(sorted_dict)
        
        first_key = next(iter(sorted_dict))
        if first_key in best_tag:
            times_going=times_going+1
        else:
            times_going=0
        best_tag.add(first_key)
        if times_going ==times:
            break
        #print(best_tag)
    return best_tag        
        
best_tags=best_tags(dict_only_seed_graph,tags,r)  
print("Best tags are ",best_tags)      