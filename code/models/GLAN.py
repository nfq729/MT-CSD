import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertModel,BertForSequenceClassification
from torch_geometric.nn import RGCNConv

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(
            in_features, out_features))  # (hd,hd)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
    def forward(self, text, adj):
        all_weights = []
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=1, keepdim=True)  
        denom = denom.squeeze().unsqueeze(-1)
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.pretrained_bert_name,ignore_mismatched_sizes=True)

        # for param in self.bert.parameters():
        #     param.requires_grad = True
        self.hid_dim = args.bert_dim


        self.Rconv1_graph1 = RGCNConv(args.bert_dim, args.bert_dim, num_relations=9)
        self.Rconv2_graph1 = RGCNConv(args.bert_dim, args.bert_dim, num_relations=9)

        self.Rconv1_graph2 = RGCNConv(args.bert_dim, args.bert_dim, num_relations=4)
        self.Rconv2_graph2 = RGCNConv(args.bert_dim, args.bert_dim, num_relations=4)

        self.gc1 = GraphConvolution(args.bert_dim, args.bert_dim)
        self.gc2 = GraphConvolution(args.bert_dim, args.bert_dim)
        self.gc3 = GraphConvolution(args.bert_dim, args.bert_dim)
  

        self.conv1 = nn.Conv1d(args.bert_dim, args.bert_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(args.bert_dim, args.bert_dim, 3, padding=1)


        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm1 = torch.nn.LayerNorm(args.bert_dim, eps=1e-5)
        self.layer_norm2 = torch.nn.LayerNorm(args.bert_dim, eps=1e-5)
        self.layer_norm3 = torch.nn.LayerNorm(args.bert_dim, eps=1e-5)

        self.fc2 = nn.Linear(args.bert_dim*3, args.polarities_dim)



    def mask(self,GCN_output,nums):
        mask = []
        for i in range(nums):
            if i < nums-1:
                mask.append(0)
            elif i == nums-1:
                mask.append(1)
        mask = torch.tensor(mask).unsqueeze(1).float().to(self.args.device)

        return mask * GCN_output

    def find_change_indices(self,token_type_ids):
        indices = []
        batched_edge_index = []
        for i in range(token_type_ids.size()[0]):
            changes = []
            for j in range(1, len(token_type_ids[i])):
                if token_type_ids[i][j] != token_type_ids[i][j-1]:
                    changes.append(j)
            if token_type_ids[i][-1] == 1:
                changes.append(len(token_type_ids[i]))
            start = []
            end = []
            for k in range(len(changes)):
                if k == 0:
                    start.append(1)
                    end.append(changes[k])
                else:
                    start.append(changes[k-1])
                    end.append(changes[k])
            indices.append(torch.stack([torch.tensor(start), torch.tensor(end)], dim=0))
            # start_indices.append(torch.tensor(start))
            # end_indices.append(torch.tensor(end))
            row = torch.arange(0, 7)
            col = torch.arange(1, 7+1)
            edge_index = torch.stack([row, col], dim=0)
            batched_edge_index.append(edge_index)
        return indices, torch.stack(batched_edge_index).to(self.args.device)



    def forward(self, inputs):
        context = inputs[0]
        mask = inputs[1]
        token_type_ids = inputs[2]
        bert_output = self.bert(context,mask,token_type_ids)
        target = self.bert(inputs[4],inputs[5],inputs[6]).pooler_output
        bert_output = bert_output.last_hidden_state
        indices,batched_edge_index = self.find_change_indices(token_type_ids)
        adj = (torch.diag(torch.ones(8)) + torch.diag(torch.ones(8-1), 1)).to(self.args.device)

        all_out = []
        for i in range(bert_output.size()[0]):
            sentence = torch.stack([bert_output[i][indices[i][0, j]:indices[i][1, j]].mean(dim=-2) for j in range(indices[i].size()[1])])
            # GCN
            GCN_adj_1 = F.tanh(self.gc1(sentence, adj[:sentence.size()[0], :sentence.size()[0]]))
            GCN_adj_2 = F.tanh(self.gc2(GCN_adj_1, adj[:sentence.size()[0], :sentence.size()[0]]))

            GCN_out = GCN_adj_2
            for mask_i in range(self.args.hop):
                alpha_mat_text = torch.matmul(torch.tensor([0] * (sentence.size()[0] - 1) + [1], device=self.args.device).unsqueeze(1).float() * GCN_out, sentence.transpose(0, 1))
                if mask_i == self.args.hop - 1:
                    # alpha_text = F.softmax(alpha_mat_text.sum(0, keepdim=True), dim=1)
                    # a3 = torch.matmul(alpha_text, sentence).squeeze(0)  
                    a3 = GCN_out[-1,:]
                else:
                    alpha_text = alpha_mat_text
                    a3 = alpha_text.transpose(0, 1)[:,sentence.size()[0]-1:sentence.size()[0]]*sentence
                    GCN_out = self.args.lambdaa * self.layer_norm1(F.sigmoid(a3)) + sentence
            
            # local CNN
            conv_out = F.tanh(self.conv1(sentence.transpose(0, 1)))
            conv_out = F.tanh(self.conv2(conv_out)).transpose(0, 1)
            for mask_i in range(self.args.hop):
                alpha_mat_text = torch.matmul(torch.tensor([0] * (sentence.size()[0] - 1) + [1], device=self.args.device).unsqueeze(1).float() * conv_out, sentence.transpose(0, 1))
                if mask_i == self.args.hop - 1:
                    # alpha_text = F.softmax(alpha_mat_text.sum(0, keepdim=True), dim=1)
                    # a3 = torch.matmul(alpha_text, sentence).squeeze(0)  
                    a4 = conv_out[-1,:]
                else:
                    alpha_text = alpha_mat_text
                    a4 = alpha_text.transpose(0, 1)[:,sentence.size()[0]-1:sentence.size()[0]]*sentence
                    conv_out = self.args.lambdaa * self.layer_norm1(F.sigmoid(a4)) + sentence
            
            # global
            text_out_mask = torch.matmul(torch.tensor([0] * (sentence.size()[0] - 1) + [1], device=self.args.device).unsqueeze(1).float() * sentence, sentence.transpose(0, 1))
            text_out_mask = F.sigmoid(text_out_mask.transpose(0, 1)[:,sentence.size()[0]-1:sentence.size()[0]]*sentence)
            for mask_i in range(self.args.hop):
                alpha_mat_text = torch.matmul(torch.tensor([0] * (sentence.size()[0] - 1) + [1], device=self.args.device).unsqueeze(1).float() * text_out_mask, sentence.transpose(0, 1))
                if mask_i == self.args.hop - 1:
                    # alpha_text = F.softmax(alpha_mat_text.sum(0, keepdim=True), dim=1)
                    # a3 = torch.matmul(alpha_text, sentence).squeeze(0)  
                    a5 = text_out_mask[-1,:]
                else:
                    alpha_text = alpha_mat_text
                    a5 = alpha_text.transpose(0, 1)[:,sentence.size()[0]-1:sentence.size()[0]]*sentence
                    text_out_mask = self.args.lambdaa * self.layer_norm1(F.sigmoid(a5)) + sentence


            fnout = torch.cat((a3,a4,a5),0)
            fnout = torch.cat((a3,a4,a5),0).view(3,a3.size()[0])
            weights = F.softmax(torch.matmul(fnout,target[0]),dim=0).view(fnout.size()[0],1)
            fnout = fnout * weights
            fnout = torch.cat((fnout[0],fnout[1],fnout[2]),0)
            out = self.fc2(fnout)
            all_out.append(out)
        all_out = torch.stack(all_out).to(self.args.device)
        return all_out
