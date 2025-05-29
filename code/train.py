import wandb
import pandas as pd
import numpy as np
import random
import torch
import math
import json
from transformers import BertModel,RobertaModel
import time
import os
from sklearn import metrics
from datetime import timedelta
import torch.nn.functional as F
import argparse
from models import GLAN
from util import dataset


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='GLAN', type=str)
    parser.add_argument('--sub_model', default='Model', type=str)
    parser.add_argument('--target', default='Biden', type=str, help='Biden, Bitcoin， SpaceX, Trump, Tesla')
    parser.add_argument('--dataset_path', default='./data/', type=str)
    parser.add_argument('--tokenizer_dataset_path', default='./data/tokenizer/', type=str)
    parser.add_argument('--lr', default=1e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--save_path', default='./save_model', type=str)
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--pretrained_bert_name', default='bert-base-chinese', type=str)
    parser.add_argument('--hop', default=3, type=int, help='')
    parser.add_argument('--lambdaa', default=0.2, type=float, help='')
    parser.add_argument('--optimizer', default='adam', type =str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--blr', default=2e-6, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--l2reg', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=384 * 2, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=9, type=int)
    parser.add_argument('--seed', default=0, type=int, help='set seed for reproducibility')
    parser.add_argument('--filter_sizes', default= (2, 3, 4))
    parser.add_argument('--num_filters', default=32)
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    torch.backends.cudnn.deterministic = True  
        
     # load dataset
    start_time = time.time()
    print("Loading data...")

    train_iter,dev_iter,test_iter = dataset(args=args)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # load model
    model_classes = {
        'GLAN': GLAN,
    }
    
    # model = attention_bert.BERT_Model(args).to(args.device)
    model = model_classes[args.model_name].Model(args).to(args.device)



    reset_params(args,model)

    train(args, model, 
          train_iter=train_iter, 
          dev_iter=dev_iter, 
          test_iter=test_iter)

def get_time_dif(start_time):
        """get used time"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

def reset_params(args,model):
        # print(1)
        for child in model.children():
            if type(child) != BertModel and type(child) != RobertaModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            # args.initializer(p)
                            torch.nn.init.xavier_uniform_(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)



def train(args, model, train_iter, dev_iter, test_iter):
    
    wandb.init(
        project=f"name",
        dir="./",
        name=f"{args.target}",
        config={
        "learning_rate": args.lr,
        "architecture": args.model_name,
        "target": args.target,
        "epochs": args.num_epoch,
        "seed":args.seed,
        "dropout":args.dropout
        }
    )

    train_logger = wandb.run.log
    val_logger = wandb.log
    test_logger = wandb.log

    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)
    total_batch = 0  
    dev_best_macro_f1 = 0
    test_best_macro_f1 = 0
    max_val_epoch = 0

    for epoch in range(args.num_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, args.num_epoch))
        for index,batch in enumerate(train_iter):
            inputs = [batch[col].to(args.device) for col in batch]
            outputs = model(inputs)
            model.zero_grad()
            labels = inputs[3]
            loss = F.cross_entropy(outputs, labels)
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()

            # if epoch > 0:

            if total_batch % 20 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()

                train_acc = metrics.accuracy_score(true, predic)
                train_macro_f1 = metrics.f1_score(true, predic,average='macro',labels=[1,2])
                train_favor_f1 = metrics.f1_score(true, predic,average='macro',labels=[1])
                train_against_f1 = metrics.f1_score(true, predic,average='macro',labels=[2])
                train_micro_f1 = metrics.f1_score(true, predic,average='micro',labels=[1,2])
                dev_acc, dev_loss, macro_f1,favor_micro_f1,against_micro_f1,micro_f1,labels_all,predict_all,macro_f1_3,micro_f1_3 = evaluate(args, model, dev_iter)


                if macro_f1 > dev_best_macro_f1:
                    dev_best_macro_f1 = macro_f1
                    max_val_epoch = epoch
                    if not os.path.isdir(os.path.join(args.save_path, args.target)):
                        os.makedirs(os.path.join(args.save_path, args.target))  
                    torch.save(model.state_dict(), os.path.join(args.save_path, args.target, args.model_name+'.ckpt'))
                    test_best_macro_f1 = test(args,model,test_iter,test_best_macro_f1,test_logger)
                    improve = '*'
                else:
                    improve = ''


                time_dif = get_time_dif(start_time)
                msg1 = 'epoch: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, train macro_f1: {3:>6.2%}, train favor macro_f1: {4:>6.2%},train against macro_f1: {5:>6.2%}, train micro_f1: {6:>6.2%},Time: {7}'
                msg2 = 'epoch: {0:>6},  val   Loss: {1:>5.2},  val   Acc: {2:>6.2%}, val   macro_f1: {3:>6.2%}, val   favor macro_f1: {4:>6.2%},val   against macro_f1: {5:>6.2%}, val   micro_f1: {6:>6.2%},Time: {7} {8}'


                print(msg1.format(epoch, loss.item(), train_acc, train_macro_f1, train_favor_f1,train_against_f1,train_micro_f1, time_dif))
                train_logger({"epoch": epoch, "Train Loss": loss.item(), "Train Acc": train_acc, "train macro_f1": train_macro_f1, "train_favor_f1": train_favor_f1,"train_against_f1":train_against_f1,"train micro_f1": train_micro_f1})

                print(msg2.format(epoch, dev_loss, dev_acc,macro_f1,favor_micro_f1,against_micro_f1,micro_f1, time_dif, improve))
                val_logger({"epoch": epoch, "val Loss": dev_loss, "val Acc": dev_acc, "val macro_f1": macro_f1,"favor_micro_f1":favor_micro_f1,"against_micro_f1":against_micro_f1, "val micro_f1": micro_f1})

                model.train()
            total_batch += 1

        if epoch - max_val_epoch >= 3:
            print('>> early stop.')
            break

    test_best_macro_f1 = test(args,model,test_iter,test_best_macro_f1,test_logger)




# eval

def evaluate(args, model, data_iter, test=False):
    if test:
        checkpoint = torch.load(os.path.join(args.save_path, args.target, args.model_name+'.ckpt'))
        model.load_state_dict(checkpoint)
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for batch in data_iter:
            inputs = [batch[col].to(args.device) for col in batch]
            outputs = model(inputs)
            labels = inputs[3]
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    acc = metrics.accuracy_score(labels_all, predict_all)
    macro_f1 = metrics.f1_score(labels_all, predict_all,average='macro',labels=[1,2])
    micro_f1 = metrics.f1_score(labels_all, predict_all,average='micro',labels=[1,2])

    favor_micro_f1 = metrics.f1_score(labels_all, predict_all,average='macro',labels=[1])
    against_micro_f1 = metrics.f1_score(labels_all, predict_all,average='macro',labels=[2])
    macro_f1_3 = metrics.f1_score(labels_all, predict_all,average='macro')
    micro_f1_3 = metrics.f1_score(labels_all, predict_all,average='micro')

    if test:
        return acc,macro_f1,favor_micro_f1,against_micro_f1,micro_f1,macro_f1_3,micro_f1_3,predict_all
    return acc, loss_total/len(data_iter),macro_f1,favor_micro_f1,against_micro_f1,micro_f1,labels_all,predict_all,macro_f1_3,micro_f1_3




# test
def test(args, model, test_iter,best_macro_f1,test_logger):
    acc,macro_f1,favor_micro_f1,against_micro_f1,micro_f1,macro_f1_3,micro_f1_3,predict_all = evaluate(args, model, test_iter, test=True)

    temp = '>>test>> acc: {0:>6.2}, macro_f1: {1:>6.2%}, favor_micro_f1: {2:>6.2%}, against_micro_f1: {3:>6.2%}, micro_f1: {4:>6.2%}, macro_f1_3: {5:>6.2%}, micro_f1_3: {6:>6.2%}'
    print()
    print(temp.format(acc,macro_f1,favor_micro_f1,against_micro_f1,micro_f1,macro_f1_3,micro_f1_3))
    print()
    test_logger({"Test Acc": acc, "Test macro_f1": macro_f1, "Test favor_micro_f1": favor_micro_f1, "Test against_micro_f1": against_micro_f1, "Test micro_f1": micro_f1, "Test macro_f1_3": macro_f1_3, "Test micro_f1_3": micro_f1_3})

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1

        label_map = {"none": 0,"favor": 1,"against": 2}
        re_label_map = { 0:"none",1:"favor",2:"against"}
        test_data = json.load(open(args.dataset_path+args.target+'/'+'test.json'))
        index_list = [value['index'][-1] for value in test_data]
        label_list = [label_map[value['stance']] for value in test_data]

        test_label = []
        for batch in test_iter:
            inputs = [batch[col] for col in batch]
            test_label.extend(x for x in inputs[3].data.cpu().numpy())

        assert label_list == list(test_label)

        data = {
            'index': index_list,
            'true_label': [re_label_map[x] for x in label_list],
            'predict': [re_label_map[x] for x in list(predict_all)]
        }
        data = pd.DataFrame(data)

        if not os.path.isdir(f'./result/{args.target}'):
            os.makedirs(f'./result/{args.target}')
        data.to_csv(f'./result/{args.target}/{args.model_name}_{args.sub_model}_{args.seed}.csv', index=False)

    return best_macro_f1
     
if __name__ == '__main__':
    main()

