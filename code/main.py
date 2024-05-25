import os
import time
import pandas as pd
import torch
import matplotlib.pyplot as plt
# import seaborn as sns
from random import random
from train import train1,train3
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.neural_network import MLPClassifier
import dataget
from model import GCN_circ, GCN_dis
import evaluation_scores
from param import parameter_parser
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB



def save_plots_base_path(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

def feature_representation(model_cir, model_dis, args, dataset):
    '''training part to get the feature'''
    model_cir.cuda()
    model_dis.cuda()
    optimizer1 = torch.optim.Adam(model_cir.parameters(), lr=0.005)
    model_cir,model_dis = train1(model_cir,model_dis, dataset, optimizer1, args)
    model_cir.eval()
    model_dis.eval()
    with torch.no_grad():
        circ_dis,cir_fea,dis_fea = model_cir(dataset)
        b, dis_fea,drug_fea = model_dis(dataset)

    cir_fea = cir_fea.cpu().detach().numpy()
    drug_fea = drug_fea.cpu().detach().numpy()
    dis_fea = dis_fea.cpu().detach().numpy()

    return  cir_fea, drug_fea, dis_fea


#save data of ROC_curve and draw a draft plot of ROC curve
def plot_and_save_roc_data(fprs, tprs, roc_aucs, photos_path):
    roc_data = pd.DataFrame(columns=['Fold', 'FPR', 'TPR', 'ROC_AUC'])
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'darkgreen', 'darkred']
    for i, (fpr, tpr, roc_auc) in enumerate(zip(fprs, tprs, roc_aucs)):
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'Fold {i+1} ROC curve (area = {roc_auc:.4f})')
        for f, t in zip(fpr, tpr):
            roc_data = roc_data.append({'Fold': i+1, 'FPR': f, 'TPR': t, 'ROC_AUC': roc_auc}, ignore_index=True)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(photos_path, 'combined_roc_curve.png'))
    plt.close()
    roc_data.to_csv(os.path.join(photos_path, 'roc_data.csv'), index=False)
# save data of PR_curve and draw a draft plot of PR curve
def plot_and_save_pr_data(precisions, recalls, aps, photos_path):
    pr_data = pd.DataFrame(columns=['Fold', 'Recall', 'Precision', 'AP'])
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'magenta', 'yellow', 'black', 'orange', 'darkgreen']
    for i, (precision, recall, ap) in enumerate(zip(precisions, recalls, aps)):
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2, label=f'Fold {i+1} PR curve (AP = {ap:.2f})')
        for p, r in zip(precision, recall):
            pr_data = pr_data.append({'Fold': i+1, 'Recall': r, 'Precision': p, 'AP': ap}, ignore_index=True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Combined Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(photos_path, 'combined_pr_curve.png'))
    plt.close()
    pr_data.to_csv(os.path.join(photos_path, 'pr_data.csv'), index=False)

#The function
def circRDRP(n_fold):
    args = parameter_parser()
    dataset, cd_pairs = dataget.dataset(args)

    kf = KFold(n_splits = n_fold, shuffle = True)


    model_cir = GCN_circ(args)
    # model_drug = GCN_drug(args)
    model_dis = GCN_dis(args)
    
    ave_acc = 0
    ave_prec = 0
    ave_sens = 0
    ave_f1_score = 0
    ave_mcc = 0
    ave_auc = 0
    ave_auprc = 0
    ave_spec = 0
    localtime = time.asctime( time.localtime(time.time()) )
    ########Can also directly get from the evaluation_scores.py
    fprs, tprs, roc_aucs = [], [], [] 
    precisions, recalls, aps = [], [], []
    #########
    results_path = '../results'
    photos_path = os.path.join(results_path, 'photos')
    save_plots_base_path(photos_path)  
    
    with open('../results/result.txt', 'a') as f:
        f.write('time:\t'+ str(localtime)+"\n")
        
        for fold_index, (train_index, test_index) in enumerate(kf.split(cd_pairs)):
            c_drugmatix,train_cd_pairs,test_cd_pairs = dataget.C_Dmatix(cd_pairs,train_index,test_index)
            dataset['c_d']=c_drugmatix
         
            cir_fea, drugfea,dis_fea = feature_representation(model_cir,model_dis, args, dataset)
            print(cir_fea.shape)
            print(drugfea.shape)
            print(dis_fea.shape)
            train_dataset = dataget.new_dataset(cir_fea, drugfea,train_cd_pairs)
            test_dataset = dataget.new_dataset(cir_fea, drugfea,test_cd_pairs)

            X_train, y_train = train_dataset[:,:-2], train_dataset[:,-2:][:,0]
            X_test, y_test = test_dataset[:,:-2], test_dataset[:,-2:][:,0]

            print(X_train.shape,X_test.shape)
            #####different classifiers' comparsion

            # clf = RandomForestClassifier(n_estimators=100,max_depth=20)
            # clf = LogisticRegression(random_state=42)
            # clf = SVC(C=0.1,kernel='rbf',gamma=5, probability=True)
            # clf = DecisionTreeClassifier(random_state=42)
            # clf = GaussianNB()#朴素贝叶斯
            # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=20, random_state=2,subsample=0.8)
            clf = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=360, random_state=42,alpha=0.001,learning_rate='constant')
            clf.fit(X_train, y_train)

            ##used to test the effectiveness in the training datasets
            # y_train_pred = clf.predict(X_train)
            ##
            # train_accuracy = accuracy_score(y_train, y_train_pred)
            # train_precision = precision_score(y_train, y_train_pred)
            # train_recall = recall_score(y_train, y_train_pred)
            # train_f1 = f1_score_func(y_train, y_train_pred)

            # print("Training Metrics:")
            # print("Accuracy:", train_accuracy)
            # print("Precision:", train_precision)
            # print("Recall:", train_recall)
            # print("F1 Score:", train_f1)
            # print("-----------Training Scores Above-------------")
            
            y_pred = clf.predict(X_test) 
            y_prob = clf.predict_proba(X_test)
            y_prob = y_prob[:, 1]
            tp, fp, tn, fn, acc, prec, sens, f1_score, MCC, AUC,AUPRC, spec = evaluation_scores.calculate_performace(len(y_pred), y_pred, y_prob, y_test) 
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            roc_aucs.append(roc_auc)

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            precisions.append(precision)
            recalls.append(recall)
            aps.append(ap)

            ###get the prediction results 
            ###you can just use ctrl+/ to get the prediction results
            # positive_indices = [i for i, pred in enumerate(y_pred)]
            # positive_pairs_with_probs = [(test_cd_pairs[i], y_prob[i]) for i in positive_indices]

            # with open('/root/autodl-tmp/CircRDRP/results/positive_circRNA_drug_pairs.txt', 'a') as file2:
            #     for pair, prob in positive_pairs_with_probs:
            #         file2.write(f"circRNA Index: {pair[0]}, Drug Index: {pair[1]}, Probability: {prob}\n")           
            
            ### output the result metrics
            print('RF: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  sens = \t', sens, '\n  f1_score = \t', f1_score, '\n  MCC = \t', MCC, '\n  AUC = \t', AUC,'\n  AUPRC = \t', AUPRC,'\n  spec = \t', spec)
            f.write('RF: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(acc)+'\t  prec = \t'+ str(prec)+ '\t  sens = \t'+str(sens)+'\t  f1_score = \t'+str(f1_score)+ '\t  MCC = \t'+str(MCC)+'\t  AUC = \t'+ str(AUC)+'\t  AUPRC = \t'+ str(AUPRC)+'\n'+'\t  spec = \t'+ str(spec)+'\n')
            ave_acc += acc
            ave_prec += prec
            ave_sens += sens
            ave_f1_score += f1_score
            ave_mcc += MCC
            ave_auc += AUC
            ave_auprc  += AUPRC
            ave_spec += spec

        
        plot_and_save_roc_data(fprs, tprs, roc_aucs, photos_path)
        plot_and_save_pr_data(precisions, recalls, aps, photos_path)
        ave_acc /= n_fold
        ave_prec /= n_fold
        ave_sens /= n_fold
        ave_f1_score /= n_fold
        ave_mcc /= n_fold
        ave_auc /= n_fold
        ave_auprc /= n_fold
        ave_spec /= n_fold
        print('Final: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(ave_acc)+'\t  prec = \t'+ str(ave_prec)+ '\t  sens = \t'+str(ave_sens)+'\t  f1_score = \t'+str(ave_f1_score)+ '\t  MCC = \t'+str(ave_mcc)+'\t  AUC = \t'+ str(ave_auc)+'\t  AUPRC = \t'+ str(ave_auprc)+'\n'+'\t  spec = \t'+ str(ave_spec)+'\n')
        f.write('Final: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(ave_acc)+'\t  prec = \t'+ str(ave_prec)+ '\t  sens = \t'+str(ave_sens)+'\t  f1_score = \t'+str(ave_f1_score)+ '\t  MCC = \t'+str(ave_mcc)+'\t  AUC = \t'+ str(ave_auc)+'\t  AUPRC = \t'+ str(ave_auprc)+'\n'+'\t  spec = \t'+ str(ave_spec)+'\n')


if __name__ == "__main__":
    args = parameter_parser()

    k_fold = args.fold ## default is 5
    rounds = args.round ## default is 10

    for i in range(rounds):
        circRDRP(k_fold)