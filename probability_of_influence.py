# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 01:32:39 2024

@author: HP
"""

import tensorflow as tf
#import torch
#from torch.utils.data import random_split,DataLoader
#import torchvision.transforms as transforms
#from torchvision.datasets import CIFAR10
#from torch import nn
#import numpy as np
#import torch.optim as optim

import pickle
#from numpy.linalg import norm

#import matplotlib.pyplot as plt
import follower_embedding
import influencial_embedding_v2
import math

dict_probability_of_influence = pickle.load(open('dict_probability_of_influence', 'rb'))

tags_least = pickle.load(open('tags_least_occurrence', 'rb'))
tags_least["The Human Energy Field: A Practical Introduction"]=1

tags_least["Thinking About Divorce: Sort out the Process"]=1
edge_probability_career=pickle.load(open("edge_probability_career","rb"))    
influenced_multiple_hop=pickle.load(open("influenced_multiple_hop","rb"))
dict_tag_influencer_follower_mapping=  pickle.load(open("dict_tag_influencer_follower_mapping","rb"))
tags=set()

def dot_product(vector1, vector2):
    dot_product_result = tf.tensordot(vector1, vector2, axes=1)
    return dot_product_result


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


for key in edge_probability_career:
    for i in edge_probability_career[key]:
        for m in edge_probability_career[key][i]:
            tags.add(m[0])
            
#dict_probabilty_influence={} # key tag  value  list   
dict1={}          
for tag in tags:
    if tag not in tags_least and tag not in "Anyone craving to play a game of real MAHJONG?" and tag not in "Community Service for Age 50's+: Various Projects": 
        print("calculating probablity for badge",tag)
        
        if tag =="\"Cloud Computing\" How to use it for your business":
            tag ="Cloud Computing"
        follower_embedding_tag=follower_embedding.follower_embedding_func(tag)
        influencial_embedding_tag=influencial_embedding_v2.influencial_embeding(tag)
        if tag =="Cloud Computing":
            tag="\"Cloud Computing\" How to use it for your business"
        for i in dict_tag_influencer_follower_mapping[tag]:
            influencer=(list(i.keys())[0])
            influencer_embedding=influencial_embedding_tag[influencer]
            for influenced in i[list(i.keys())[0]]:
                follower=influenced
                follower_embedding_follower=follower_embedding_tag[follower]
                dot_product_result=dot_product(influencer_embedding, follower_embedding_follower)
                probability=sigmoid(dot_product_result) # here calculating the probablity
                dict2={}
                dict2[follower]=[(probability,tag)]
                if influencer in dict1:
                    dict3=dict1[influencer]
                    if follower in dict3:
                        list1=dict3[follower]
                        list1.append((probability,tag))
                        #list1.append(probability)
                        #list1.append(tag)
                        dict3[follower]=list1
                        dict1[influencer]=dict3
                        
                    else:
                        list1=[]
                        list1.append((probability,tag))
                        #list1.append(probability)
                        #list1.append(tag)
                        dict3[follower]=list1
                        dict1[influencer]=dict3
                        
                        
                else:
                    dict1[influencer]=dict2
                    
                        
                        
                
        #dot_product_result=dot_product(influencer_embedding, follower_embedding)
        #probability=sigmoid(dot_product_result)
        

#pickle.dump(dict1, open('dict_probability_of_influence', 'wb')) # saving the badge wise probablity of influence

