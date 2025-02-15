﻿import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import os 
import torch.nn.functional as F
import numpy as np
import pandas as pd
import glob
import random
from torch.autograd import Variable
from torch import optim
from models.tdnn import TDNN
from sklearn.metrics import accuracy_score
#from hierarchical_loss import HierarchicalLossNetwork
#from level_dict import hierarchy

nc=15
IP_dim=1024
n_epoch=100

#lan1id={'kon':0,'kan':1,'tam':2,'mar':3}
lan2id={'ka':0,'ng':1,'sg':2,'HK':3,'KK':4,'MK':5,'OM':6,'MT':7,'NT':8,'OT':9,'ST':10,'TT':11,'UT':12,'MA':13,'PU':14}
le=len('/media/data/CygNet_DL2/final_split/train/kar/')
#le1=len('/media/data/CygNet_DL2/trial/hierarchical/train/')
#print(le)
def lstm_data(f):
    X = torch.load(f)   
    f1 = os.path.splitext(f)[0]     
    #lang = f1[le1:le1+3]  
    #print('In LSTM lang',lang)
    #Y1 = lan1id[lang]    
    #Y1 = np.array([Y1]) 
    #Y1 = torch.from_numpy(Y1).long() 
    #print('f1',f1)
    dial = f1[le:le+2]
    #print('In LSTM Dialect',dial)
    Y2 = lan2id[dial]    
    Y2 = np.array([Y2]) 
    #print(Y2)
    #Xdata1 = np.array(X)    
    #Xdata1 = torch.from_numpy(Xdata1).float() 
    #print('file name',f)
    #print('XDATA1',Xdata1)
    Y2 = torch.from_numpy(Y2).long() 
    #print('Y2',Y2)
    return X, Y2  # Return the data and true label
    
#################################################
################ X_vector #######################

class X_vector(nn.Module):
    def __init__(self, input_dim = IP_dim, num_classes=15):
        super(X_vector, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        #self.output = nn.Linear(512, num_classes)
        self.output1= nn.Linear(512,num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        tdnn1_out = F.relu(self.tdnn1(inputs))
        #print(f'shape of tdnn1 is {tdnn1_out.shape}')
        tdnn2_out = self.tdnn2(tdnn1_out)
        #print(f'shape of tdnn2 is {tdnn2_out.shape}')
        tdnn3_out = self.tdnn3(tdnn2_out)
        #print(f'shape of tdnn3 is {tdnn3_out.shape}')
        tdnn4_out = self.tdnn4(tdnn3_out)
        #print(f'shape of tdnn4 is {tdnn4_out.shape}')
        tdnn5_out = self.tdnn5(tdnn4_out)
        #print(f'shape of tdnn5 is {tdnn5_out.shape}')
        
        ### Stat Pooling        
        mean = torch.mean(tdnn4_out,1)
        #print(f'shape of mean is {mean.shape}')
        std = torch.var(tdnn4_out,1,)
        #print(f'shape of std is {std.shape}')
        stat_pooling = torch.cat((mean,std),1)
        #print(f'shape of stat_pooling is {stat_pooling.shape}')
        segment6_out = self.segment6(stat_pooling)
        
        segment6_out1 = segment6_out[-1]

        #print(f'shape of segment6 is {segment6_out1.shape}')
        #ht = torch.unsqueeze(ht, 0)
        segment6_out1 = torch.unsqueeze(segment6_out1, 0)
        #print(f'shape of segment6 is {segment6_out1.shape}')
        x_vec = self.segment7(segment6_out1)
        adl1 = self.segment7(x_vec)
        adl2 = self.segment7(adl1)
        predictions_level2 = self.output1(adl1)
        return predictions_level2
        
        
######################## X_vector ####################
model = X_vector(IP_dim, nc)
model.cuda()

optimizer =  optim.Adam(model.parameters(), lr=0.0001)
loss_lang = torch.nn.CrossEntropyLoss()  # cross entropy loss function for the softmax output

#####for deterministic output set manual_seed ##############
manual_seed = random.randint(1,10000) #randomly seeding
random.seed(manual_seed)
torch.manual_seed(manual_seed)

#######################################################


files_list = []
main_folder_path = '/media/data/CygNet_DL2/final_split/train/'
subfolders = ['kon', 'kan','tam','mar']
for subfolder in subfolders:
    subfolder_path = main_folder_path + subfolder + '/'
    
    for folder in glob.glob(subfolder_path + '*'):
        for f in glob.glob(folder + '/*.pt'):
            files_list.append(f)

print('Total Training files: ', len(files_list))
l = len(files_list)
txtfl = open('/media/data/CygNet_DL2/final_split/baseline/xvec/train1.txt', 'w') # txt file to write training loss and accuracies afte every epoch.

#HLN = HierarchicalLossNetwork(hierarchical_labels=hierarchy, device='cuda')
#new_file = open('/home/administrator/ananya/hierarchical/xvec/train.txt','w')
########################
#model.load_state_dict(torch.load('/mnt/Disk12TB/DID/hierarchical/xvec/dloss/e_18.pth'))

for e in range(n_epoch): # repeat for n_epoch
    i = 0
    cost = 0.0
    random.shuffle(files_list)
    train_loss_list=[]
    full_preds=[]
    full_gts=[]

    for fn in files_list:                          
        XX1, YY1 = lstm_data(fn) # get data from file
        #print("shape of xx1",XX1.shape)
        
        XX1 = torch.unsqueeze(XX1, 1) # Adding one additional dimension at specified position
        #print("shape of xx1",XX1.shape)

        i = i+1  #Counting the number of files

        X1 = np.swapaxes(XX1,0,1)  # changing the axis(similar to transpose)

        # Enable cuda
        X1 = Variable(X1,requires_grad=False).cuda() 
        Y1 = Variable(YY1,requires_grad=False).cuda()

        model.zero_grad() # setting model gradient to zero before calculating gradient
        
        lang_op = model.forward(X1)   # forward propagation the input to the model
        #print("lang_op=", lang_op)
        #print("lang_op shape=", lang_op.shape)

        T_err = loss_lang(lang_op,Y1)  # loss calculation over the model output with true label
               
        T_err.backward()  # calculating the gradient on loss obtained
        
        optimizer.step() # parameter updation based on gradient calculated in previous step 
        
        train_loss_list.append(T_err.item())

        cost = cost + T_err.item()
            
        print("x-vec-bnf.py: Epoch = ",e+1,"  completed files  "+str(i)+"/"+str(l)+" Loss= %.7f"%(cost/i))
        predictions = np.argmax(lang_op.detach().cpu().numpy(),axis=1) #Convert one hot coded result to label
        for pred in predictions:
            full_preds.append(pred)
        for lab in Y1.detach().cpu().numpy():
            full_gts.append(lab)

            ############################

    mean_acc = accuracy_score(full_gts,full_preds)  # accuracy calculation over the true label and predicted label
    mean_loss = np.mean(np.asarray(train_loss_list)) # average loss calculation
    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,e+1))      
    path = "/media/data/CygNet_DL2/final_split/baseline/xvec/model1/e_"+str(e+1)+".pth"
    torch.save(model.state_dict(),path) # saving the model parameters 
    txtfl.write(path)
    txtfl.write('\n')
    txtfl.write("acc: "+str(mean_acc))
    txtfl.write('\n')
    txtfl.write("loss: "+str(mean_loss))
    txtfl.write('\n')

###############################################################
