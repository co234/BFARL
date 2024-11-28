import torch
from torch.autograd import Variable
import numpy as np
import os
from data import *
from sklearn.model_selection import train_test_split
import os


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def compute_expectation(pred,Y,l_func):
    exp_loss = torch.zeros(pred.shape[0],)
    compute_idx = np.where(pred!=0)[0]
    for i in compute_idx:
        temp_matrix = torch.ones(Y.shape)*pred[i] 
        exp_loss[i] = l_func(temp_matrix,Y)

    return exp_loss


def farl_loss(pred,y,a,l_func,b_0,b_1):
    m_idx = np.where(a==0)[0]
    f_idx = np.where(a==1)[0]
    y_m = y[m_idx]
    y_f = y[f_idx]

    Ea = compute_expectation((1-a)*(pred),y_m,l_func)
    Eb = compute_expectation(a*(pred),y_f,l_func)

    l_f = l_func(pred, y) - (b_0*Ea +b_1*Eb)
    
    return l_f



def compute_di(data,pred,s):
    pred_label = np.array(pred)
    if isinstance(s, np.ndarray):
        f_idx,m_idx = np.where(s==0)[0], np.where(s==1)[0]
    else:
        f_idx,m_idx = np.where(data[:,s]==0)[0], np.where(data[:,s]==1)[0]
    f_ratio, m_ratio = (pred_label[f_idx]==1).sum()/(len(f_idx)+1e-15), (pred_label[m_idx]==1).sum()/(len(m_idx)+1e-15)
    dp = np.min([(f_ratio/(m_ratio+1e-15)),(m_ratio/(f_ratio+1e-15))])

    return dp

def compute_deo(data,label,pred,s):
    pred_label = np.array(pred)
    y_pos = np.where(label==1)[0]
    if isinstance(s, np.ndarray):
        f_idx,m_idx = np.where(s==0)[0], np.where(s==1)[0]
    else:
        f_idx,m_idx = np.where(data[:,s]==0)[0], np.where(data[:,s]==1)[0]
    pos_f,pos_m = np.intersect1d(y_pos,f_idx), np.intersect1d(y_pos,m_idx)
    f_eo,m_eo = (pred[pos_f]==1).sum()/(len(pos_f)+1e-15), (pred[pos_m]==1).sum()/(len(pos_m)+1e-15)

    return np.abs(f_eo-m_eo)


def compute_bal_acc(label,pred):
    label[label == 0] = -1
    pred[pred == 0] = -1
    tp = len(np.intersect1d(np.where(label==1)[0],np.where(pred==1)[0]))
    tn = len(np.intersect1d(np.where(label==-1)[0],np.where(pred==-1)[0]))

    tpr = tp/len(np.where(label==1)[0])
    tnr = tn/len(np.where(label==-1)[0])

    bal_acc = (tpr+tnr)/2

    return bal_acc


def compute_min_di(data,pred,s):
    di = compute_di(data,pred,s)
    return 1-np.minimum(di,1/di)



def data_loader(data,test_ratio=0.1,enforce_di = False):
    if data == "adult":
        train_dir = os.path.join("data/adult",'adult.data')
        test_dir = os.path.join("data/adult",'adult.test')
        processer = AdultProcess()
        X_train,y_train,s = processer.process(train_dir)
        X_test,y_test,_ = processer.process(test_dir)

        test_ratio = len(X_test)/(len(X_test)+len(X_train))

        x = np.append(X_train,X_test,axis=0)
        y = np.append(y_train,y_test)

        print("Adult dataset loaded")
    elif data == "compas":
        train_dir = os.path.join("data/compas",'compas-scores-two-years.csv')
        processer = CompasProcess()
        x,y,s = processer.process(train_dir)
  
    elif data == "german":
        train_dir = os.path.join("data/german",'german.data')
        processer = GermanProcess()
        x,y,s = processer.process(train_dir)
  
    elif data == "arrhythmia":
        train_dir = os.path.join("data/arrhythmia",'arrhythmia.csv')
        processer = ArrhythmiaProcess()
        x,y,s = processer.process(train_dir)
   
    elif data == "drug":
        train_dir = os.path.join("data/drug",'drug_consumption.data')
        processer = DrugProcess()
        x,y,s = processer.process(train_dir)
  
    elif data == "bank":
        train_dir = os.path.join("data/bank",'bank-additional-full.csv')
        processer = BankProcess()
        x,y,s = processer.process(train_dir)
   
    elif data == "violent":
        train_dir = os.path.join("data/viloent",'cox-parsed.csv')
        processer = ViloentProcess()
        x,y,s = processer.process(train_dir)
    else:
        print("{} dataset not implemented yet".format(data))

    if enforce_di is True:
        x_,y_ = enforce_di_fair(x,y,s)
    else:
        x_,y_ = x,y

    
    X_train,X_test,y_train,y_test = train_test_split(x_,y_,test_size = test_ratio,random_state = 42)

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    return (X_train,y_train),(X_test,y_test),s
    

def unlabel_dataset(dataset, labels, ratio):
    idx_pool = list(range(0,len(labels)))
    pos_label_idx = np.where(labels==1)[0]
    pos_idx = np.random.choice(pos_label_idx, int(len(labels)*ratio))
    pos_x, pos_y = dataset[pos_idx], labels[pos_idx]
    rest_idx = list(set(idx_pool)-set(pos_idx))

    X_un, y_un_true = dataset[rest_idx], labels[rest_idx]
    print("Positive labeled shape: {}, Unlabeled shape: {}".format(pos_x.shape, X_un.shape))
    print("Original rho+1 = {}".format(len(pos_idx)/len(pos_label_idx)))
    
  
    return (pos_x,pos_y),(X_un,y_un_true)


def flip_datset(dataset, labels, flip_rate):
    idx_pool = list(range(0,len(labels)))
    pos_rate = 1- flip_rate
    pos_label_idx = np.where(labels==1)[0]
    pos_idx = np.random.choice(pos_label_idx, int(len(pos_label_idx)*pos_rate))
    pos_x, pos_y = dataset[pos_idx], labels[pos_idx]
    rest_idx = list(set(idx_pool)-set(pos_idx))

    X_un, y_un_true = dataset[rest_idx], labels[rest_idx]
    print("Positive labeled shape: {}, Unlabeled shape: {}".format(pos_x.shape, X_un.shape))
    print("Original rho+1 = {}".format(1-len(pos_idx)/len(pos_label_idx)))
    
  
    return (pos_x,pos_y),(X_un,y_un_true)


def sigmoid(t):
    s = 1.0/(1.0 + np.exp(-t)+1e-15)
    return s


def flip_real_data(x,y,s,flip_ratio):
    if isinstance(s, np.ndarray):
        f_idx,m_idx = np.where(s==0)[0], np.where(s==1)[0]
    else:
        f_idx,m_idx = np.where(x[:,s]==0)[0], np.where(x[:,s]==1)[0]

    pos_idx = np.where(y==1)[0]
    neg_idx = np.where(y==0)[0]
    
    y_ = np.copy(y)

    f_pos_idx = np.intersect1d(f_idx,pos_idx) # flip protected group y=0 to y=1
    m_neg_idx = np.intersect1d(m_idx,neg_idx) # flip unprotected group y=1 to y=0

    flip_to_one_idx = np.random.choice(m_neg_idx,int(len(m_neg_idx)*flip_ratio))
    flip_to_zero_idx = np.random.choice(f_pos_idx,int(len(f_pos_idx)*flip_ratio))

    for idx in flip_to_one_idx:
        y_[idx] = 1

    for idx in flip_to_zero_idx:
        y_[idx] = 0

    return y_


def enforce_di_fair(data,label,s):
    f_idx,m_idx = np.where(data[:,s]==1)[0], np.where(data[:,s]==0)[0]
    pos_idx, neg_idx = np.where(label==1)[0], np.where(label==0)[0]
    f_ratio, m_ratio = (label[f_idx]==1).sum()/len(f_idx), (label[m_idx]==1).sum()/len(m_idx)
    m_pos,m_neg = np.intersect1d(m_idx,pos_idx), np.intersect1d(m_idx,neg_idx)
    f_pos,f_neg = np.intersect1d(f_idx,pos_idx), np.intersect1d(f_idx,neg_idx)
    
    num_pos = min(len(m_pos),len(f_pos))
    num_neg = min(len(m_neg),len(f_neg))

    m_pos,m_neg = np.random.choice(m_pos,num_pos), np.random.choice(m_neg,num_neg)
    f_pos,f_neg = np.random.choice(f_pos,num_pos), np.random.choice(f_neg,num_neg)

    all_idx = list(m_pos)+list(m_neg)+list(f_pos)+list(f_neg)

    return data[all_idx],label[all_idx]


def add_selection_bias(x,y,s,sigma = 1.5):
    pos_pro_idx = np.intersect1d(np.where(y==1)[0],np.where(x[:,s]==1)[0])

    r = len(pos_pro_idx)/len(np.where(x[:,s]==1)[0])
    a = r/sigma
    c = len(np.where(x[:,s]==1)[0])-len(pos_pro_idx)
    pos_pro_num = int(a*c/(1-a))

    selected_pos_idx = np.random.choice(pos_pro_idx,pos_pro_num)
    exclude_idx = list(set(pos_pro_idx)-set(selected_pos_idx))
    x_ = np.delete(x, exclude_idx,axis=0)
    y_ = np.delete(y, exclude_idx)

    return x_,y_



def add_label_bias(yclean,rho,theta_dict):
    """
    theta_0_p: P(Y=+1|Z=-1,A=0)
    theta_0_m: P(Y=-1|Z=+1,A=0)
    theta_1_p: P(Y=+1|Z=-1,A=1)
    theta_1_m: P(Y=-1|Z=+1,A=1)
    """
    n = len(yclean)

    t_0_p, t_0_m, t_1_p,t_1_m = theta_dict['theta_0_p'],theta_dict['theta_0_m'],theta_dict['theta_1_p'],theta_dict['theta_1_m']


    def locate_group(label,sensitive_attr,a,y):
        return np.intersect1d(np.where(sensitive_attr==a)[0],np.where(label==y)[0])

    g_01, g_00 = locate_group(yclean,rho,0,1),locate_group(yclean,rho,0,-1)
    g_11, g_10 = locate_group(yclean,rho,1,1),locate_group(yclean,rho,1,-1)

    group = [g_01,g_00,g_11,g_10]

    theta = [t_0_m,t_0_p,t_1_m,t_1_p]
    tilde_y = [-1,1,-1,1]

  

    t = yclean.copy()
    for i in range(len(group)):
        for j in range(len(group[i])):
            p = np.random.uniform(0,1)
            if p < theta[i]:
                t[group[i][j]] = tilde_y[i]
            else:
                t[group[i][j]] = yclean[group[i][j]]
        # flip_idx = np.random.choice(group[i],int(theta[i]*len(group[i])))
        # t[flip_idx] = tilde_y[i]

    return t

    

    