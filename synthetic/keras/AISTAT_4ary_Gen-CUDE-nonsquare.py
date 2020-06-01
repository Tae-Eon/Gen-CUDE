#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
import tensorflow as tf

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import np_utils        
from torch.utils.data import TensorDataset 


import os
import sympy as sp
from sympy import exp, sqrt, pi, Integral, Symbol
import time
from scipy.io import savemat


import matplotlib.pyplot as plt


# In[2]:


seed_value = 0
set_seed(seed_value)


# In[3]:


experiment_directory_name = '4ary-non_square'
if not os.path.isdir('./'+experiment_directory_name):
    os.mkdir('./'+experiment_directory_name)


# In[4]:


# Generate Data


# In[5]:


data_n = int(3e6) #int(3e6)
sigma=1.0
nb_x_classes = 4
nb_z_classes = 4
transition_prob = 0.1


# In[6]:


bs = base_symbol(nb_z_classes)
print(bs)


# In[7]:


# db = decision_boundary(nb_z_classes)
# print(db)


# In[8]:


a = sym_mat(nb_x_classes, transition_prob)
print(a)


# In[9]:


discrete_x_denoising = source_generator_v2(data_n, a)  
print(discrete_x_denoising[10:20])


# In[10]:


noisy_y_denoising = np.round(con_noisy_awgn(discrete_x_denoising,sigma) , 2 ) # upto two decimal point
print(noisy_y_denoising[10:20])


# In[ ]:





# In[11]:


print(min(noisy_y_denoising))
print(max(noisy_y_denoising))


# In[12]:


db1_1 = np.round(np.random.uniform(low = -2.7, high = -2.3, size=1)[0] , 2)
db1_2 = np.round(np.random.uniform(low= -1.7, high= -1.3, size=1)[0], 2)
db2_1 = np.round(np.random.uniform(low= -0.7, high= -0.3, size=1)[0], 2)
db2_2 = np.round(np.random.uniform(low= 0.3, high= 0.7, size=1)[0], 2)
db3_1 = np.round(np.random.uniform(low= 1.3, high= 1.7, size=1)[0], 2)
db3_2 = np.round(np.random.uniform(low= 2.3, high= 2.7, size=1)[0], 2)
print(db1_1,db1_2,db2_1,db2_2,db3_1,db3_2)

db_set = [db1_1, db1_2, db2_1, db2_2, db3_1, db3_2]
print(np.arange(2*nb_z_classes-1)-3)


# In[13]:


print(db_set)

print(bs)

bs_double = np.arange(2*nb_z_classes-1)-3
print('base_symbol with margin :', bs_double)


# In[14]:


quantized_z_denoising = Quantizer_4ary(noisy_y_denoising, nb_z_classes, db_set)
print(db_set)
print(noisy_y_denoising[100:200])
print(quantized_z_denoising[100:200])


# In[15]:


print(min(noisy_y_denoising))
print(max(noisy_y_denoising))
print(min(quantized_z_denoising))
print(max(quantized_z_denoising))


# In[16]:


print( 'Data length : ', len(discrete_x_denoising) )
error_x_z = error_rate(discrete_x_denoising, quantized_z_denoising)
print('error_rate(x vs z) : ' , error_x_z)


# # Channel

# In[17]:


# induced channel matrix


# In[18]:


PI = np.zeros((nb_x_classes,len(bs_double)),dtype=float)
x = Symbol('x')
for i in range(nb_x_classes):
    f = sp.exp(-(x-(bs[i]))**2/(2*((sigma)**2)))/(sigma*sp.sqrt(2*sp.pi))
    for j in range(len(bs_double)):
        if j == 0 :
            PI[i][j] = sp.Integral(f, (x, -float('inf'), db_set[0] )).doit().evalf()
            
        elif j == len(bs_double)-1:
            PI[i][j] = sp.Integral(f, (x, db_set[j-1], float('inf'))).doit().evalf()
            
        else :
            PI[i][j] = sp.Integral(f, (x, db_set[j-1], db_set[j] )).doit().evalf()
    print('proceeding..')
print(PI)
PI_inverse=np.linalg.pinv(PI)
print(PI_inverse)


# In[19]:


# pdf table


# In[20]:


start = - (nb_z_classes-1)*10
print(start)
end = (nb_z_classes-1)*10
print(end)
interval_length = end - start
print(interval_length)

pdf_table = np.zeros((nb_x_classes , interval_length*100),dtype='float64')

for i in range(nb_x_classes):
    f = sp.exp(-(x-(bs[i]))**2/(2*((sigma)**2)))/(sigma*sp.sqrt(2*sp.pi))
    f_ = f.evalf()
    
    for j in range(interval_length*100) :
        pdf_table[i][j] = f_.subs(x, start+(j)/100)
        
    print('proceeding..')
print(pdf_table.shape)
print(pdf_table)


# In[21]:


print(min(noisy_y_denoising))
print(max(noisy_y_denoising))
print('Range :',start, -start)


# In[22]:


np.set_printoptions(linewidth=180)
print('PI :')
print(PI)

print('pdf_table :')
print(pdf_table)


# In[23]:


plt.rcParams["figure.figsize"] = (15,10)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True 
plt.rcParams['font.size'] = 40

x = np.arange(start,end,0.01) # -36.00 ~ 35.99 까지 ~~

y_0 = pdf_table[0]
y_1 = pdf_table[1]
y_2 = pdf_table[2]
y_3 = pdf_table[3]
    
plt.plot(x,y_0,label='x=-3') # A bar chart
plt.plot(x,y_1,label='x=-1') # A bar chart
plt.plot(x,y_2,label='x=1') # A bar chart
plt.plot(x,y_3,label='x=3') # A bar chart

#plt.title("pdf table made by picking points ")
plt.xlabel('y value')
plt.ylabel('pdf value')
plt.xticks([-5,-3,-1,1,3,5])
plt.legend(loc='upper right')
plt.xlim(-6+bs[0],6+bs[nb_z_classes-1])
plt.savefig("./"+experiment_directory_name+"/binary_pdf_table_line.png", dpi = 100)

plt.show()


# In[ ]:





# In[24]:


file_model = 'Gen-CUDE'


# In[25]:


cuda = True
device = torch.device("cuda:0" if cuda else "cpu")
print(device)


# In[26]:


batch_size = 4000
epochs = 10


# In[27]:


loss_matrix = np.int_(np.ones((nb_x_classes, nb_x_classes)) - np.eye(nb_x_classes))
print(loss_matrix)

nb_z_classes_new = len(bs_double)
z=quantized_z_denoising + nb_z_classes - 1  # 0, 1, 2, 3, .., 6
Z=np_utils.to_categorical(  z, nb_z_classes_new ,dtype=np.int32)
print(z)


# In[ ]:





# In[28]:


k_set = []
error_set = []
time_set = []

for k in [1,2,3,5,8,10,15,20,30,50]:
    print('k',k)
    
    start_time = time.time()  
    ###################################################################################
    train_x = input_context_without_middle_symbol(noisy_y_denoising, k)
    train_y = Z[k:len(Z)-k,]
   

    y_middle = middle_y(noisy_y_denoising,k) 
    pdf_vector = p_vector_from_wide_pdf_table(y_middle, pdf_table, nb_x_classes, end)
    
    
    
    
    #----------------------------Neural Net model------------------------------#
    #inputs = layers.Input(shape = (2 * k * nb_z_classes,))  # one hot
    inputs = layers.Input(shape = (2 * k,))  # not one hot
    layer = layers.Dense(200, kernel_initializer = 'he_normal')(inputs)
    layer = layers.Activation('relu')(layer)
    layer = layers.Dense(200, kernel_initializer = 'he_normal')(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Dense(200, kernel_initializer = 'he_normal')(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Dense(200, kernel_initializer = 'he_normal')(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Dense(200, kernel_initializer = 'he_normal')(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Dense(200, kernel_initializer = 'he_normal')(layer)
    layer = layers.Activation('relu')(layer)
    layer = layers.Dense(nb_z_classes_new , kernel_initializer = 'he_normal')(layer)
    outputs = layers.Activation('softmax')(layer)
    model = models.Model(inputs = inputs, outputs = outputs)
    #--------------------------------------------------------------------------#
    
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam)


    
    model.fit(train_x,train_y,epochs=epochs,batch_size=batch_size,
                     verbose=1, validation_data=(train_x,train_y), shuffle=True)

    
    pred_prob = model.predict(train_x, batch_size = batch_size, verbose = 1)
    
    pred_prob_u_0 = pred_prob.dot(PI_inverse)
    new_target = pred_prob_u_0 * pdf_vector    
    denoised_seq = np.argmin(new_target.dot(loss_matrix),axis=1) 
    
    x_nn_hat_sub= transform_to_wide(denoised_seq, nb_x_classes)
    x_hat=np.hstack((quantized_z_denoising[0:k],x_nn_hat_sub,quantized_z_denoising[data_n - k:data_n]))
    ###############################################################################################
    
    duration = time.time() - start_time
    print("time(s)):", duration )  
    
    error_x_hat = error_rate(discrete_x_denoising,x_hat)
    print('x_hat_error : ', error_x_hat)

    
    ################################# save result ################################################ 
    f = open('./'+experiment_directory_name+'/result_'+file_model+'.txt', 'a')
    f.flush()
    PRINT(f, 'window size k : %d'%k)
    PRINT(f, 'time(s) : %f'%duration)
    f.flush()
    PRINT(f, 'x_hat_error_rate : %f'%error_x_hat)
    PRINT(f, '')

    f.flush()
    f.close()
    
    
    k_set.append(k)
    time_set.append(duration)
    error_set.append(error_x_hat)


# In[ ]:


quantized_z_denoising_better = Quantizer_v1(noisy_y_denoising, nb_z_classes, db_set = [-2, 0, 2]) 
error_x_z = error_rate(discrete_x_denoising, quantized_z_denoising_better)


# In[ ]:


saving_list = {'window_size_k':k_set, 'time':time_set, 'error_rate':error_set, 'error_x_z':error_x_z}
saving_list = dict(saving_list)
savemat('./'+experiment_directory_name+'/result_'+file_model+'.mat', saving_list)

