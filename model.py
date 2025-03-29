# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:50:57 2024

@author: HP
"""

#import tensorflow as tf
import torch
#from torch.utils.data import random_split,DataLoader
#import torchvision.transforms as transforms
#from torchvision.datasets import CIFAR10
#from torch import nn
#import numpy as np
#import torch.optim as optim
#import input1
#import output
import pickle
#import input_cascade
#import matplotlib.pyplot as plt


dict_tag_influencer_follower_mapping=  pickle.load(open("dict_tag_influencer_follower_mapping","rb"))

def input_output_func(tag):
   
    influencer=set()
    follower=set()
    # find the influencial users
    for i in dict_tag_influencer_follower_mapping[tag]:
        influencer.add(list(i.keys())[0])
        for influenced in i[list(i.keys())[0]]:
            follower.add(influenced)
    #got the influencer now get the tenspr
    dict_influencer_mapping={} # key influencer and value index
    
    index=0
    
    for key in influencer:
        dict_influencer_mapping[key]=index
        index=index+1
    dict_influencer_tensor_mapping={}
    vector_size = len(influencer)
    for key in dict_influencer_mapping:
        index=dict_influencer_mapping[key]
        column_vector = torch.zeros(vector_size,1)
        column_vector[index][0] = 1
        dict_influencer_tensor_mapping[key]=column_vector    # this is the input tensor
    
    
    ################# for follower ####################
    dict1_follower_mapping={}
    index=0
    for influenced in follower:
        dict1_follower_mapping[influenced]=index
        index=index+1  
    vector_size = len(follower) 
    dict_influencial_influenced_tensor_mapping={}
    dict_influencial_cascade_mapping={}
    
    for key in influencer:
        vector_tensor = torch.zeros(len(follower))
        index_set=set()
        for i in dict_tag_influencer_follower_mapping[tag]:
            if list(i.keys())[0] == key:
                dict_influencial_cascade_mapping[key]=len(i[list(i.keys())[0]])
                for influenced in i[list(i.keys())[0]]:
                    index_set.add(dict1_follower_mapping[influenced])
                    vector_tensor[dict1_follower_mapping[influenced]]=1
                dict_influencial_influenced_tensor_mapping[key]=vector_tensor    
                
            
    return dict_influencer_tensor_mapping, dict_influencial_influenced_tensor_mapping,dict_influencial_cascade_mapping   
    #  dict_influencer_tensor_mapping is the input and dict_influencial_influenced_tensor_mapping is the output
    
    print("")    

'''
def input_output_func():
    for tag in dict_tag_influencer_follower_mapping:
        influencer=set()
        follower=set()
        # find the influencial users
        for i in dict_tag_influencer_follower_mapping[tag]:
            influencer.add(list(i.keys())[0])
            for influenced in i[list(i.keys())[0]]:
                follower.add(influenced)
        #got the influencer now get the tenspr
        dict_influencer_mapping={} # key influencer and value index
        
        index=0
        
        for key in influencer:
            dict_influencer_mapping[key]=index
            index=index+1
        dict_influencer_tensor_mapping={}
        vector_size = len(influencer)
        for key in dict_influencer_mapping:
            index=dict_influencer_mapping[key]
            column_vector = torch.zeros(vector_size,1)
            column_vector[index][0] = 1
            dict_influencer_tensor_mapping[key]=column_vector    # this is the input tensor
        
        
        ################# for follower ####################
        dict1_follower_mapping={}
        index=0
        for influenced in follower:
            dict1_follower_mapping[influenced]=index
            index=index+1  
        vector_size = len(follower) 
        dict_influencial_influenced_tensor_mapping={}
        
        
        for key in influencer:
            vector_tensor = torch.zeros(len(follower))
            index_set=set()
            for i in dict_tag_influencer_follower_mapping[tag]:
                if list(i.keys())[0] == key:
                    for influenced in i[list(i.keys())[0]]:
                        index_set.add(dict1_follower_mapping[influenced])
                        vector_tensor[dict1_follower_mapping[influenced]]=1
                    dict_influencial_influenced_tensor_mapping[key]=vector_tensor    
                
            
    return dict_influencer_tensor_mapping, dict_influencial_influenced_tensor_mapping   
    #  dict_influencer_tensor_mapping is the input and dict_influencial_influenced_tensor_mapping is the output
    
    print("")    
'''    