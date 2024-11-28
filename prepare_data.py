import numpy as np
import os
import pickle
from utils import *


def prepare_data(dataset, shuffle_num = 10,test_size = 0.1,enforce_di=False,selection_bias = 1):
    print("start")

    (X_train, z_train), (X_test, y_test), s = data_loader(dataset,test_size,enforce_di)

    

    if enforce_di is True:
        directory = "pre_data_f"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, 'clean_{}.pickle'.format(dataset))
        with open(file_path,'wb') as handle:
            pickle.dump([X_train,z_train,X_test,y_test,s],handle)
    else:
        directory = "pre_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, 'clean_{}.pickle'.format(dataset))
        with open(file_path,'wb') as handle:
            pickle.dump([X_train,z_train,X_test,y_test,s],handle)


    if selection_bias != 1:
        X_train,z_train = add_selection_bias(X_train,z_train,s,sigma=selection_bias)
    
    sensitive_attr = X_train[:,s]
    # Test with asymmetric bias
    t_0_p = 0.25
    t_0_m = 0.05
    t_1_p = 0.05
    t_1_m = 0.25

    thetas = []

    train_data= []
    validation_data=[]
    test_data = []
    for i in range(shuffle_num):
        theta_sum = (t_0_p+t_0_m+t_1_p+t_1_m)/4
        theta_dict = {'theta_0_p':t_0_p,'theta_0_m':t_0_m,'theta_1_p':t_1_p,'theta_1_m':t_1_m}
        y_train = add_label_bias(z_train,sensitive_attr,theta_dict)
        thetas.append(theta_sum)
        Xtr,Xval,ytr,yval = train_test_split(X_train,y_train,test_size = 0.1,random_state = 42)
        train_data.append([Xtr,ytr])
        validation_data.append([Xval,yval])
        test_data.append([X_test,y_test])

    store_data = [train_data,validation_data,test_data,s]
    if enforce_di is True:
        with open('pre_data_f/corrupt_{}_{}.pickle'.format(dataset,selection_bias),'wb') as handle:
            pickle.dump(store_data,handle)
    else:
        with open('pre_data/corrupt_{}_{}.pickle'.format(dataset,selection_bias),'wb') as handle:
            pickle.dump(store_data,handle)
    avg_t = np.mean(theta_sum)
    print("average theta is {:.3f}".format(avg_t))
    print(store_data[3])
    
    
if __name__ == '__main__':
    data = "adult"
    prepare_data(data,test_size=0.1,enforce_di = False,selection_bias=1.3)



