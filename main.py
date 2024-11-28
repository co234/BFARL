import numpy as np
import argparse
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score
import time
import pickle
from model import MLP
from utils import to_var, compute_di, compute_deo, farl_loss

########## Load dataset ########################

parser = argparse.ArgumentParser(description='PyTorch Uncertainty Training')
#Network
parser.add_argument('--lr_1', default=0.05, type=float, help='learning_rate')
parser.add_argument('--lr_2', default=0.001, type=float, help='learning_rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
parser.add_argument('--dataset',default='compas',type=str,help='input_dataset')
parser.add_argument('--b0',default=0.5,type=float,help='beta_0')
parser.add_argument('--b1',default=0.5,type=float,help='beta_1')
parser.add_argument('--n_iter',default=300,type=int,help="number of training iteration")
parser.add_argument('--n_burn',default=50,type=int,help='number of burning')
parser.add_argument('--n_train',default=50,type=int,help='number of training')
parser.add_argument('--lr_b0',default=5e-2,type=float,help='learning rate of b0')
parser.add_argument('--lr_b1',default=5e-2,type=float,help='learning rate of b1')
#Summary
args = parser.parse_args()


############### Build model ######################################
def build_model(d,h):
    mlp =MLP(feature_dim=d,hidsizes=h,dropout=0.1)
    if torch.cuda.is_available():
        mlp.cuda()
        torch.backends.cudnn.benchmark=True

    return mlp


def burn_model(s, xtr, ytr,  xts, yts,b_0,b_1,log_interval = 10,n_burn = 100,dataset = "compas"):
    d = xtr.shape[1]-1
    h = [20,10]
    
    model = build_model(d,h)
    optimizer = torch.optim.SGD(model.params(), lr=args.lr_1, momentum=args.momentum, weight_decay=args.weight_decay)
  
    save_root = './{}_burn.pth'.format(dataset)
 
    best_accuracy = 0
    loss = torch.nn.BCEWithLogitsLoss()

    for i in tqdm(range(n_burn)):
        model.train()

        xtr = to_var(xtr, requires_grad=False)
        ytr = to_var(ytr, requires_grad=False)

        y_pred = model(np.delete(xtr,s,1))
        l_f_meta = loss(y_pred,ytr)

        model.zero_grad()

        optimizer.zero_grad()
        l_f_meta.backward()
        optimizer.step()

        if i % log_interval == 0:
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(np.delete(xts,s,1))).cpu().numpy()
                pred = pred.round()
               
                test_acc = accuracy_score(yts,pred)
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(),save_root)
    print('burn => Data dp: {:.4f}'.format(compute_di(xtr,ytr,s)))



def meta_train(s, xtr, ytr,  xval,yval,xts, yts, b_0, b_1,log_interval=50,n_iter = 500,dataset = 'compas'):

    d = xtr.shape[1]-1
    h = [20,10]

    main_model= build_model(d,h)
    optimizer = torch.optim.SGD(main_model.params(), lr=args.lr_2, momentum=args.momentum, weight_decay=args.weight_decay)

    save_root = './{}_burn.pth'.format(dataset)
    main_model.load_state_dict(torch.load(save_root))

    loss = torch.nn.BCEWithLogitsLoss()
    best_accuracy = 0
    best_b0 = 0
    best_b1 = 0

  
    betas = []
    meta_train_loss = []
    meta_val_loss = []
    train_losses = []
    deos = []
    dis = []
    f1_list = []
    accuracy_list = []
    predictions=[]
    train_accuracy_list = []

    
    m = xtr.shape[0]
    for i in tqdm(range(n_iter)):

        mb_idxes = np.random.choice(m, 128, replace=False)
        mb_X_train, mb_y_train = xtr[mb_idxes], ytr[mb_idxes]
        
        main_model.train()

        meta_net = MLP(d,h)
        meta_net.load_state_dict(main_model.state_dict())
        if torch.cuda.is_available():
            meta_net.cuda()

        y_pred = meta_net(np.delete(mb_X_train,s,1))

        a_tr = mb_X_train[:,s]
        b_0 = torch.tensor(b_0) if isinstance(b_0, float) else b_0
        b_1 = torch.tensor(b_1) if isinstance(b_1, float) else b_1
        b_0 = to_var(b_0.clone().detach())
        b_1 = to_var(b_1.clone().detach())

        # Update meta loss
        l_f_meta = farl_loss(y_pred,mb_y_train,a_tr,loss,b_0,b_1).mean()
        meta_net.zero_grad()
        grads = torch.autograd.grad(l_f_meta,(meta_net.params()))
        meta_net.update_params(1e-4,source_params=grads)
        y_g_hat = meta_net(np.delete(xval,s,1))
        a_val = xval[:,s]
        l_g_meta = farl_loss(y_g_hat,yval,a_val,loss,b_0,b_1).mean()

        # Update b0 and b1
        grad_beta_0 = torch.autograd.grad(l_g_meta,b_0,only_inputs=True)[0]
        grad_beta_1 = torch.autograd.grad(l_g_meta,b_1,only_inputs=True)[0]

        b_0= b_0-args.lr_b0*grad_beta_0
        b_1= b_1-args.lr_b1*grad_beta_1

        if b_0 > 3 or b_1 > 3:
            b_0 = 2.9
            b_1 = 2.9

        y_pred_ = main_model(np.delete(mb_X_train,s,1))
        l_f = farl_loss(y_pred_,mb_y_train,a_tr,loss,b_0,b_1).mean()

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()
        

        
        main_model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(main_model(np.delete(xts,s,1))).cpu().numpy()
            pred = pred.round()
            test_acc = accuracy_score(yts,pred)
            test_f1 = f1_score(yts,pred,average="weighted")
            deo = compute_deo(xts,yts,pred,s)
            di = compute_di(xts,pred,s)
            
        
        if test_f1 > best_accuracy:
            best_accuracy = test_f1
            best_b0 = b_0
            best_b1 = b_1
            torch.save(main_model.state_dict(),save_root)

        if i % log_interval == 0:
            print('Meta train => Best_Test_f1: {:.2f} Test_f1: {:.2f} deo: {:.2f} di: {:.2f}'.format(best_accuracy, test_f1,deo,di))
            print('b0: {:.4f}, b1: {:.4f}'.format(b_0,b_1))

            deos.append(deo)
            dis.append(di)
            train_losses.append(l_f.mean())
            betas.append([b_0,b_1])
            meta_train_loss.append(l_f_meta.mean())
            meta_val_loss.append(l_g_meta.mean())
            accuracy_list.append(test_acc)
            predictions.append(pred)
            f1_list.append(test_f1)

    results_dict={'deo':deos,
                  'di':dis,
                  'train_loss':train_losses,
                  'meta_train_loss':meta_train_loss,
                  'meta_val_loss':meta_val_loss,
                  'accuracy':accuracy_list,
                  'predictions':predictions,
                  'betas':betas,
                  'f1':'f1_list',
                  'train_accuracy':train_accuracy_list}

    current_time = time.strftime("%Y%m%d_%H%M%S")
    with open('results/{}_result_dict_{}.pickle'.format(dataset,current_time), 'wb') as handle:
            pickle.dump(results_dict, handle)

    return best_b0,best_b1


def actual_train(s, xtr, ytr,  xts, yts,b_0,b_1,n_train = 50,dataset = "compas"):

    d = xtr.shape[1]-1
    h = [20,10]
    
    main_model= build_model(d,h)
    optimizer = torch.optim.SGD(main_model.params(), lr=args.lr_2, momentum=args.momentum, weight_decay=args.weight_decay)

    save_root = './{}_burn.pth'.format(dataset)
    main_model.load_state_dict(torch.load(save_root))
 
    loss = torch.nn.BCELoss()
    m = xtr.shape[0]
    for i in tqdm(range(n_train)):
        main_model.train()
        mb_idxes = np.random.choice(m, 516, replace=False)
        mb_X_train, mb_y_train = xtr[mb_idxes], ytr[mb_idxes]

        y_pred = torch.sigmoid(main_model(np.delete(mb_X_train,s,1)))
        a_tr = mb_X_train[:,s]

        # Update loss
        l_f = farl_loss(y_pred,mb_y_train,a_tr,loss,b_0,b_1).mean()

        main_model.zero_grad()

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

    main_model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(main_model(np.delete(xts,s,1))).cpu().numpy()
        pred = pred.round()
        
        test_acc = accuracy_score(yts,pred)
        test_f1 = f1_score(yts,pred,average="weighted")
        deo = compute_deo(xts,yts,pred,s)
        di = compute_di(xts,pred,s)

    print("f1: {:.2f}".format(test_f1*100))
    print("di: {:.2f}".format(di*100))
    print("deo: {:.2f}\n".format(deo*100))

    return test_acc, test_f1, di, deo
 



def run_meta():

    data = pickle.load(open("pre_data/corrupt_{}_1.5.pickle".format(args.dataset),"rb"))

    acc_lists = []
    di_lists = []
    deo_lists = []
    f1_lists = []

    train_set = data[0]
    vali_set = data[1]
    test_set = data[2]
    s = data[3]

    for i in range(10):
        print("Start {} shuffle data".format(i))
        X_train,y_train= train_set[i][0],train_set[i][1]
        X_valid,y_valid = vali_set[i][0],vali_set[i][1]
        X_test,y_test = test_set[i][0],test_set[i][1]

        y_train[y_train==-1]=0
        y_test[y_test==-1]=0
        y_valid[y_valid==-1]=0



        X_train = torch.from_numpy(X_train.astype(np.float32)).type(torch.FloatTensor)
        X_test = torch.from_numpy(X_test.astype(np.float32)).type(torch.FloatTensor)
        y_train = torch.tensor(y_train, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)
    
        X_valid = torch.from_numpy(X_valid.astype(np.float32)).type(torch.FloatTensor)
        y_valid = torch.tensor(y_valid,dtype=torch.float)

        print("dataset description: train set => {}, vali set => {}, test set => {}".format(X_train.shape, X_valid.shape, X_test.shape))
  
        burn_model(s, X_train, y_train, X_test, y_test, args.b0,args.b1,n_burn = args.n_burn,dataset=args.dataset)

        
        b_0,b_1 = meta_train(s, X_train, y_train,X_valid,y_valid, X_test, y_test, args.b0,args.b1,n_iter = args.n_iter,dataset=args.dataset)
        acc,f1,di,deo = actual_train(s, X_train, y_train, X_test, y_test, b_0, b_1,n_train = args.n_train,dataset=args.dataset)

        acc_lists.append(acc)
        f1_lists.append(f1)
        di_lists.append(di)
        deo_lists.append(deo)

    print("Results on noisy data")
    print("f1: {:.2f}+/-{:.2f}".format(np.nanmean(f1_lists)*100,np.nanstd(f1_lists)*100))
    print("deo: {:.2f}+/-{:.2f}".format(np.nanmean(deo_lists)*100,np.nanstd(deo_lists)*100))
    print("di: {:.2f}+/-{:.2f}\n".format(np.nanmean(di_lists)*100,np.nanstd(di_lists)*100))
    



if __name__=='__main__':
    run_meta()


    


