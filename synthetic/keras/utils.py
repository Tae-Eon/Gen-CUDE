import numpy as np
import torch
import torch.nn as nn
import random
import datetime
import copy

def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def PRINT(f,s):
    out = str(datetime.datetime.now()) + '\t' + s
    print(out)
    f.write(out+'\n')  
    
    
    
def decision_boundary(hid_states):
    db = np.arange(hid_states)
    db *= 2
    db = db - (hid_states-1)
    return db



def sym_mat(states, tran_prob):  # make transition matrix
    x = np.ones((states,states)) * (tran_prob/(states-1))
    for i in range(states):
        x[i][i] = 1 - (states-1)*x[i][i]
    return x



def source_generator_v2(n, a): # a is transition matrix
    x = np.zeros(n, dtype = np.int)
    hid_states = a.shape[0]
    a_sum = np.copy(a)
    for i in range(1,hid_states):
        a_sum.T[i] += a_sum.T[i-1]   

    prob = np.random.random()
    x[0] = int(prob/(1 / float(hid_states)))
    
    for i in range(1,n):
        prob = np.random.random()
        x[i] = np.argmax(a_sum[x[i-1]] > prob)
        
    x *= 2
    x = x - (hid_states-1)     
    return x


def con_noisy_awgn(x,sigma):  # make noisy y
    y=np.zeros(len(x),dtype=np.float32)
    for i in range(len(x)):
        noise=sigma*np.random.randn() ## y=m+z*sig, Y~N(m,sigma^2)  
        y[i]=x[i]+noise
    return y


def find_nearest_integer(noisy_y, nb_z_classes): # make quantized z
    noisy_y_shifted = noisy_y + (nb_z_classes - 1 )
    noisy_y_scaled =  noisy_y_shifted / 2
    z_tmp = np.int_(np.round(noisy_y_scaled))
    z_tmp[np.where(z_tmp<0)[0]] = 0
    z_tmp[np.where(z_tmp>=nb_z_classes)[0]] = nb_z_classes-1
    z_tmp = z_tmp * 2
    quantized_z = z_tmp - (nb_z_classes - 1)
    return quantized_z



def error_rate(a,b): ## bit error
    a=np.int_(a)
    b=np.int_(b)
    error=np.zeros(len(a))
    for i in range(len(a)):
        if a[i]==b[i]:
            error[i]=0
        else:
            error[i]=1
    return np.sum(error)/len(a)



def p_vector_from_wide_pdf_table(middle_y, pdf_table, nb_x_classes, end):
    shift = end*100
    prob_vector = np.zeros((len(middle_y),nb_x_classes))
    middle_y = np.round(middle_y, 2)
    for i in range(len(middle_y)):
        prob_vector[i]  = pdf_table[:,shift+int(middle_y[i]*100)]
    return prob_vector


def transform_to_narrow(seq,hid_states):
    seq_tmp = seq + (hid_states - 1)
    narrow_seq = seq_tmp/2
    narrow_seq = narrow_seq.astype(seq.dtype)
    return narrow_seq

def transform_to_wide(seq,hid_states):
    seq_tmp = seq * 2
    wide_seq = seq_tmp - (hid_states - 1)
    wide_seq = wide_seq.astype(seq.dtype)
    return wide_seq


def input_context_without_middle_symbol(noisy_y, k):
    data_n = len(noisy_y)
    noisy_context = np.zeros((data_n-2*k, 2*k),dtype = noisy_y.dtype)
    
    for ii in range(k,data_n-k):
        n_ii = np.hstack((noisy_y[ii-k:ii,],noisy_y[ii+1:ii+k+1,]))    # make context without middel symbol
        noisy_context[ii-k,]=n_ii 

    return noisy_context


def middle_y(noisy_y,k):
    n=len(noisy_y)
    middle_y = np.zeros((n-2*k,1),dtype = noisy_y.dtype) 
    for i in range(k,n-k):
        middle_y[i-k]=noisy_y[i] 
    return middle_y


def init_params(model): # weight initialization
    for p in model.parameters():
        if(p.dim() > 1):
            #nn.init.xavier_normal_(p)
            nn.init.kaiming_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)
            

def net_output(net, x, device):
    x_ = torch.from_numpy(x).float().to(device)
    output = net(x_).detach().cpu().numpy()   
    return output


def base_symbol(hid_states):
    base_symbol = np.arange(hid_states)
    base_symbol *= 2
    base_symbol = base_symbol - (hid_states-1)
    return base_symbol


def Quantizer_4ary(noisy_y, nb_z_classes, db_set):
    print('This Quantizer is designed for the "4ary & non-square PI" case in which number of db is double including "Margin" :', (nb_z_classes-1)*2 == len(db_set))
    
    bs = base_symbol(nb_z_classes)
    print('base_symbol :', bs)
    bs_double = np.arange(2*nb_z_classes-1)-3
    print('base_symbol with margin :', bs_double)
    
    z_tmp = copy.deepcopy(noisy_y)
     
    z_tmp[np.where( z_tmp < db_set[0] )[0]] = bs_double[0]  # 왼쪽 끝 대입
    z_tmp[np.where( z_tmp > db_set[len(db_set)-1] )[0]] = bs_double[len(bs_double)-1] # 오른쪽 끝 대입

    for i in range(len(db_set)-1): #0,1 (왼쪽이 기준)
        z_tmp[np.where( (db_set[i] <= z_tmp) & (z_tmp <= db_set[i+1]) )[0]] = bs_double[i+1]

    quantized_z = np.int_(z_tmp)
    return quantized_z