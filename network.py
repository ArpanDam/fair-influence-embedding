# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:02:14 2024

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
#import input1
#import output
import pickle
#import input_cascade
#import matplotlib.pyplot as plt

import model

tags_least = pickle.load(open('tags_least_occurrence', 'rb'))
edge_probability_career=pickle.load(open("edge_probability_career","rb"))    
influenced_multiple_hop=pickle.load(open("influenced_multiple_hop","rb"))

tags=set()
dict1={} # key tag value number of occurence

for key in edge_probability_career:
    for i in edge_probability_career[key]:
        for m in edge_probability_career[key][i]:
            tags.add(m[0])
            
class NN(nn.Module):
  def __init__(self,input_size,output_size):
    super().__init__()
    
        
    self.hidden = nn.Linear(input_size, 30)
    self.hidden2 = nn.Linear(30, output_size)
    self.hidden3=nn.Linear(30, 1)
    nn.init.constant_(self.hidden3.weight, 1)
    nn.init.constant_(self.hidden3.bias, 0)
    self.influenced=nn.Linear(output_size, output_size)
    self.cascade=nn.Linear(1,1)
       
  def forward(self,x,task_id):
    
    #if self.hidden is None:
     #       raise RuntimeError("Dynamic layer not initialized. Call set_input_size before forward pass.")  
    
    #self.hidden = nn.Linear(input_size, 30)
    x=self.hidden(x)
    x = torch.relu(x)
    
    if task_id ==1:
        x=self.hidden2(x)
        x = torch.relu(x)
        x=self.influenced(x)
        x = torch.sigmoid(x)
    if task_id ==2:
        x=self.hidden3(x)
        x = torch.relu(x)
        x=self.cascade(x)
        #x = torch.sigmoid(x)
    return x
   
    return x 
 
#mlp=NN()
# Make hidden3 layer untrainable


#criterion = nn.BCELoss()
#infuenced_loss_fn = nn.BCELoss()
#cascade_loss_fn = nn.MSELoss()
#optimizer = optim.Adam(mlp.parameters(), lr=0.01)
i=0
max_epoch=150
current_loss=[]
for tag in tags:
    #if tag not in tags_least and tag not in "Anyone craving to play a game of real MAHJONG?":
    #if tag == "The Human Energy Field":    
    #if tag == "Public Speaking as a Means to Market your Business":
    #flag=0
    input1,target,cascade=model.input_output_func(tag)
    print("Training for badge ",tag)
    
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
        break
    #input1,target=model.input_output_func(tag) 
    #print(mlp)
    for epoch in range(max_epoch):
        loss_list=[]
        mlp.train()
        for key in input1:
            input_tensor=input1[key]
            target_tensor=target[key]
            #input_size=input_tensor.shape[0]
            #output_size=target_tensor.shape[0]
           
            reshaped_target_tensor=target_tensor.view(1, -1)
            reshaped_input_tensor=input_tensor.view(1, -1)
            outputs_influenced = mlp(reshaped_input_tensor,task_id=1)
            optimizer.zero_grad()
            loss_influenced = infuenced_loss_fn(outputs_influenced, reshaped_target_tensor)
            
            
            cascade_size=cascade[key]
            outputs_cascade = mlp(reshaped_input_tensor,task_id=2)
            target_cascade_length = torch.tensor(cascade_size)
            target_cascade_length = target_cascade_length.to(torch.float)
            loss_cascade = cascade_loss_fn(outputs_cascade, target_cascade_length)
            
            loss = loss_influenced + loss_cascade
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
        print("For epoch",epoch,"loss is ",sum(loss_list)/len(loss_list));
    print("Final loss of tag "+tag+" is ",sum(loss_list)/len(loss_list))
    #torch.save(mlp.state_dict(), tag+'saved_model_0.001.pth')    Saving the model
    