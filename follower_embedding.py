# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:17:25 2024

@author: HP
"""
#import tensorflow as tf
import torch
#from torch.utils.data import random_split,DataLoader
#import torchvision.transforms as transforms
#from torchvision.datasets import CIFAR10
from torch import nn
#import numpy as np
import torch.optim as optim
import model
#import input1
#import output
import pickle
#from numpy import array
#from numpy.linalg import norm
#import matplotlib.pyplot as plt
import os

folder_path = '\saved model'
current_path = os.getcwd()
folder_name = 'saved model'

# Create the full path by joining the current path with the folder name
folder_path = os.path.join(current_path, folder_name)

# List all files and subdirectories in the folder
contents = os.listdir(folder_path)
dict_tag_influencer_follower_mapping=  pickle.load(open("dict_tag_influencer_follower_mapping","rb"))
def follower_embedding_func(tag):
    
    print("Inside follower function")
    print(tag)
    if tag =="Cloud Computing":
        tag = "\"Cloud Computing\" How to use it for your business"
    input1,target,cascade=model.input_output_func(tag)
    influencer=set()
    follower=set()
    # find the influencial users
    for i in dict_tag_influencer_follower_mapping[tag]:
        influencer.add(list(i.keys())[0])
        for influenced in i[list(i.keys())[0]]:
            follower.add(influenced)
            
    dict1_follower_mapping={}
    index=0
    for influenced in follower:
        dict1_follower_mapping[influenced]=index
        index=index+1         
           
            
    
    
    
    
    
    
    
    class NN(nn.Module):
      def __init__(self,input_size,output_size):
        super().__init__()
        
            #nn.Flatten(),  # Flattening the matrix to 1-D
        self.hidden = nn.Linear(input_size, 30)
        self.hidden2 = nn.Linear(30, output_size)
        self.hidden3 = nn.Linear(30, 1)
        nn.init.constant_(self.hidden3.weight, 1)
        nn.init.constant_(self.hidden3.bias, 0)
        self.hidden3.trainable = False
        self.influenced = nn.Linear(output_size, output_size)
        self.cascade = nn.Linear(1, 1)     
      def forward(self,x,task_id):
        x=self.hidden(x)
        x = torch.relu(x)
        
        if task_id ==1:
            x=self.hidden2(x)
            hidden2_output = self.hidden2(x)
            x = torch.relu(hidden2_output)
            x=self.influenced(x)
            x = torch.sigmoid(x)
        if task_id ==2:
            x=self.hidden3(x)
            x = torch.relu(x)
            x=self.cascade(x)
            x = torch.sigmoid(x)
        return x  
        
   # mlp=NN()
    #loaded_model = mlp(input_size=10, output_size=2)
    
    # Load the saved parameters into the model
    #mlp.load_state_dict(torch.load('saved_model_sigmoid_1_hidden_layer_0.001.pth'))
    #mlp.load_state_dict(torch.load('saved_model_sigmoid_0.001_multitask.pth'))
    #mlp.load_state_dict(torch.load('saved_model_sigmoid_0.001_multitask_2_hidden_layer.pth'))
    #mlp.load_state_dict(torch.load('saved_model_sigmoid_0.001_multitask_2_hidden_layer_30_embedding_untrainable_v3.pth'))
    #state_dict = mlp.state_dict()  
    input1,target,cascade=model.input_output_func(tag)
    for key in input1:
        input_tensor=input1[key]
        target_tensor=target[key]
        input_size=input_tensor.shape[0]
        output_size=target_tensor.shape[0]
        mlp=NN(input_size,output_size)
        for param in mlp.hidden3.parameters():
            param.requires_grad = False
            
        infuenced_loss_fn = nn.BCELoss()
        cascade_loss_fn = nn.MSELoss()
        #cascade_loss_fn = nn.MSELoss()
        optimizer = optim.Adam(mlp.parameters(), lr=0.001)
        #mlp.eval()
        os.chdir(folder_path)
        if tag == "\"Cloud Computing\" How to use it for your business":
            tag= "Cloud Computing"
        mlp.load_state_dict(torch.load(tag+'saved_model_0.001.pth'))
        state_dict = mlp.state_dict()
        os.chdir(current_path)
        break
    embedding_follower={} # key is influenced member and value the embedding     
    print("")  
    
    
    for key in follower:
        index=dict1_follower_mapping[key]
        embedding_follower[key]=state_dict['hidden2.weight'][index]
    '''
    for key in index_middle_embedding:
        index=index_middle_embedding[key]
        embedding[key]=state_dict['hidden2.weight'][index]
    
    
    for key in index_bad_embedding:
        index=index_bad_embedding[key]
        embedding[key]=state_dict['hidden2.weight'][index]'''  
    return embedding_follower        
'''
tags_least = pickle.load(open('tags_least_occurrence', 'rb'))
edge_probability_career=pickle.load(open("edge_probability_career","rb"))    
influenced_multiple_hop=pickle.load(open("influenced_multiple_hop","rb"))
tags=set()
dict1={} # key tag value number of occurence

for key in edge_probability_career:
    for i in edge_probability_career[key]:
        for m in edge_probability_career[key][i]:
            tags.add(m[0])
for tag in tags:
    if tag not in tags_least and tag not in "Anyone craving to play a game of real MAHJONG?": 
        d=follower_embedding(tag)'''
print("") 