################## used Library  ############################################################
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
import sklearn.metrics
import h5py
from sklearn.metrics import roc_curve

n_classes = 15 # Number of language classes 
IP_dim = 1024
lan1id={'ka':0,'ng':1,'sg':2,'HK':3,'KK':4,'MK':5,'OM':6,'MT':7,'NT':8,'OT':9,'ST':10,'TT':11,'UT':12,'MA':13,'PU':14}
le=len('/media/data/CygNet_DL2/final_split/test/kar/')

#le=len('/media/data/CygNet_DL2/trial/kannada/wav2vec2/test/')

def input_data(f):
    # print(f)
   # hf = h5py.File(f, 'r')
    #X = np.array(hf.get('feature'))
    #y = np.array(hf.get('target'))
    # print(X.shape, "---", y.shape)
    #hf.close()      
    # print("Y[0]=", y[0])    
    #Y1 = y[0]
    X = torch.load(f)
    f1 = os.path.splitext(f)[0]     
    lang = f1[le:le+2]  
    #print('lang',lang)
    Y1 = lan1id[lang]    
    Y1 = np.array([Y1]) 
    Y1 = torch.from_numpy(Y1).long()

    return X, Y1  # Return the data and true label


############################################### X_vector ######################################################################


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
        x_vec = self.segment7(segment6_out1)
        adl1 = self.segment7(x_vec)
        adl2 = self.segment7(adl1)
        predictions_level2 = self.output1(adl1)
        return predictions_level2

#################################################### X_vector #####################################################################

files_list=[]
main_folder_path = '/media/data/CygNet_DL2/final_split/test/'
subfolders = ['kon', 'kan','tam','mar']
for subfolder in subfolders:
    subfolder_path = main_folder_path + subfolder + '/'
    
    for folder in glob.glob(subfolder_path + '*'):
        for f in glob.glob(folder + '/*.pt'):
            files_list.append(f)

print('Total Test files: ',len(files_list))

l=len(files_list)  
txtfl = open('/media/data/CygNet_DL2/final_split/baseline/xvec/test1-new.txt', 'w')#write mode -- Txt file to write output accuracy

A = []


for e in range(25,100):  ### Test for 30 models --- Model trained for 30 epochs and saved after each epoch
    model = X_vector(IP_dim, n_classes)
    model.cuda()
    random.shuffle(files_list) 
    path =  "/media/data/CygNet_DL2/final_split/baseline/xvec/model/e_"+str(e+1)+".pth"
    print(path)
    model.load_state_dict(torch.load(path))
    model.cuda()
    Actual=[]
    Pred=[]

    i=0
    for fn in files_list:
        X, Y = input_data(fn)
        X = torch.unsqueeze(X, 1)

        X = np.swapaxes(X,0,1)
        X = Variable(X, requires_grad=True).cuda()

        lang_op= model.forward(X)
        
        P = np.argmax(lang_op.detach().cpu().numpy(),axis=1)
        # print(P)
        i+=1
        
       
        Actual.append(int(Y))
        Pred.append(int(P))        
        print("epoch",e+1," Predicted for ",i,"/",l)
        
    CM2=sklearn.metrics.confusion_matrix(Actual, Pred)
    print(CM2)
    txtfl.write(path)
    txtfl.write('\n')
    txtfl.write(str(CM2))
    txtfl.write('\n')
    acc = sklearn.metrics.accuracy_score(Actual,Pred)
    print(acc)
    txtfl.write(str(acc))
    txtfl.write('\n')
    A.append(acc)
    nc1 = 15
    fpr = dict()
    tpr = dict()
    fnr = dict()
    EER = dict()
    y_test  = F.one_hot(torch.as_tensor(Actual), num_classes=nc1)
    y_score = F.one_hot(torch.as_tensor(Pred), num_classes=nc1)
  
    for i in range(nc1):
      fpr[i], tpr[i],_ = roc_curve(y_test[:, i], y_score[:, i],pos_label=1)
      fnr[i] = 1-tpr[i]
      EER[i] = (fpr[i][np.nanargmin(np.absolute((fnr[i]-fpr[i])))] + 
              fnr[i][np.nanargmin(np.absolute((fnr[i]-fpr[i])))])/2
            
    print(EER)
    print('EER dialect:\t',np.mean(list(EER.values())))
    txtfl.write("EER:"+str(EER))
    txtfl.write('\n')
    txtfl.write("Mean EER:"+str(np.mean(list(EER.values()))))
    txtfl.write('\n')
   
print(max(A))    
txtfl.write(str(max(A)))
txtfl.close()
