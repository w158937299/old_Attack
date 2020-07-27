import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import Parser
import Bulidata
import time
from tqdm import tqdm
import getData
import networkx
import datetime
import dateutil
import math
import scipy.stats
from dateutil.relativedelta import relativedelta
import sys

from NewAttention import NewAttention

class HetAD(nn.Module):

    def __init__(self, embed_dim, uu_neigh, ui_neigh, iu_neigh):
        super(HetAD, self).__init__()

        self.embed_dim = embed_dim
        self.uu_neigh = uu_neigh
        self.ui_neigh = ui_neigh
        self.iu_neigh = iu_neigh

        #                             embed_dim ,hidden_size, num_layers
        self.u_content_rnn = nn.LSTM(embed_dim, int(embed_dim / 2), 1, bidirectional=True)
        self.i_content_rnn = nn.LSTM(embed_dim, int(embed_dim / 2), 1, bidirectional=True)

        # self.u_neigh_rnn = nn.LSTM(embed_dim, int(embed_dim / 2), 1, bidirectional=True)
        # self.i_neigh_rnn = nn.LSTM(embed_dim, int(embed_dim / 2), 1, bidirectional=True)
        #
        # self.u_neigh_att = nn.Parameter(torch.ones(embed_dim * 2, 1), requires_grad=True)
        # self.i_neigh_att = nn.Parameter(torch.ones(embed_dim * 2, 1), requires_grad=True)

        args = Parser.Define_Params()
        User_num = len(ui_neigh)
        Item_num = len(iu_neigh)
        start_node_embedding = args.start_node_embedding
        self.u_self_embed = np.zeros((User_num + 1, embed_dim))
        self.i_self_embed = np.zeros((Item_num + 1, embed_dim))

        self.u_i_embed = np.zeros((args.u_l + 1, embed_dim))
        self.u_u_embed = np.zeros((args.u_l + 1, embed_dim))
        self.i_u_embed = np.zeros((args.i_l + 1, embed_dim))

        self.att = NewAttention(self.embed_dim)

        with open(start_node_embedding, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip('\n').split('  ')
                embed = line[1].split(', ')
                embeds = np.asarray(embed, dtype='float32')
                if line[0][0] == 'u':
                    self.u_self_embed[int(line[0][1:])] = embeds
                elif line[0][0] == 'i':
                    self.i_self_embed[int(line[0][1:])] = embeds

        # 用户评价项目的embedding聚合这个是本身的权重，然后它的属性例如他评论的项目，的权重相加取均值把然后在聚合
        for user in ui_neigh:
            normal_user = user[1:]
            items = ui_neigh[user][1: -1].split(', ')
            for item in items:
                item = item[1: -1]
                self.u_i_embed[int(normal_user)] = np.add(self.u_i_embed[int(normal_user)], self.i_self_embed[int(item[1:])])
            if len(items) != 0:
                self.u_i_embed[int(normal_user)] = self.u_i_embed[int(normal_user)] / len(items)
            else:
                self.u_i_embed[int(normal_user)] = self.u_self_embed[int(normal_user)]



        for user in uu_neigh:
            normal_user = user[1:]
            users = uu_neigh[user][1: -1].split(', ')
            for neigh_user in users:
                neigh_user = neigh_user[1:-1]
                # 考虑没有用户关联的权重怎么办
                self.u_u_embed[int(normal_user)] = np.add(self.u_u_embed[int(normal_user)], self.u_self_embed[int(neigh_user[1:])])
            if len(users) != 0:
                self.u_u_embed[int(normal_user)] = self.u_u_embed[int(normal_user)] / len(users)
            else:
                self.u_u_embed[int(normal_user)] = self.u_self_embed[int(normal_user)]

        for item in iu_neigh:
            normal_item = item[1:]
            for i_user in iu_neigh[item]:
                i_user = i_user[1:]
                self.i_u_embed[int(normal_item)] = np.add(self.i_u_embed[int(normal_item)], self.u_self_embed[int(i_user)])
            if len(iu_neigh[item]) != 0:
                self.i_u_embed[int(normal_item)] = self.i_u_embed[int(normal_item)] / len(iu_neigh[item])
            else:
                self.i_u_embed[int(normal_item)] = self.i_self_embed[int(normal_item)]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def u_content_agg(self, id_batch):


        u_self_embed = self.u_self_embed[id_batch]
        u_i_embed = self.u_i_embed[id_batch]
        u_u_embed = self.u_u_embed[id_batch]

        u_self_embed = torch.from_numpy(u_self_embed)
        u_i_embed = torch.from_numpy(u_i_embed)
        u_u_embed = torch.from_numpy(u_u_embed)

        # concat_embed = torch.cat((u_self_embed, u_i_embed, u_u_embed), axis = 1)
        concat_embed = torch.cat((u_self_embed, u_i_embed, u_u_embed), axis = 1).view(len(u_self_embed), 3, self.embed_dim)
        att_w = self.att(concat_embed)


        concat_embed = concat_embed * att_w

        concat_embed = torch.transpose(concat_embed, 0, 1)

        # 3 615 128
        concat_embed = torch.tensor(concat_embed, dtype= torch.float32)


        # 这里的h0和c0经测试可以省略，默认设置为0
        #                                第一维是长短数据 batch_size默认第二维度  行数据的个数也就是embedding_size
        all_state, last_state = self.u_content_rnn(concat_embed)

        return torch.mean(all_state, 0)

    def i_content_agg(self, id_batch):

        i_self_embed = self.i_self_embed[id_batch]
        i_u_embed = self.i_u_embed[id_batch]
        # 可以考虑在这加时间评分
        i_self_embed = torch.from_numpy(i_self_embed)
        i_u_embed = torch.from_numpy(i_u_embed)

        concat_embed = torch.cat((i_self_embed, i_u_embed), axis = 1).view(len(i_self_embed), 2, self.embed_dim)
        concat_embed = torch.transpose(concat_embed, 0, 1)
        concat_embed = torch.tensor(concat_embed, dtype=torch.float32)
        all_state, last_state = self.u_content_rnn(concat_embed)

        return torch.mean(all_state, 0)


    # 我这个代码在产生训练集的时候已经考虑了邻居节点所以不用聚合邻居节点,事后可以尝试一下在考虑邻居节点
    def types_agg(self, id_batch, type):



        if type == 1:
            c_agg_deputy = self.u_content_agg(id_batch)
        elif type == 2:
            c_agg_deputy = self.i_content_agg(id_batch)

        # c_agg_batch_2 = torch.cat((c_agg_deputy, c_agg_deputy), 1).view(len(c_agg_batch), self.embed_d * 2)
        # a_agg_batch_2 = torch.cat((c_agg_deputy, a_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        # 求注意力也就不考虑邻居了
        # concat_embed = torch.cat((u_content_embed, i_content_embed), axis = 1).view(len(u_content_embed), 2, self.embed_dim)

        # print(u_content_embed.shape)
        # print(i_content_embed.shape)
        # att = self.att(u_content_embed, i_content_embed)
        # print(att)
        # print(att.shape)
        # exit(0)
        # 不加注意力

        return c_agg_deputy

    def whole_agg(self,c_id_batch, pos_id_batch, neg_id_batch, type):

        if type == 1:
            c_agg = self.types_agg(c_id_batch, 1)
            p_agg = self.types_agg(pos_id_batch, 1)
            n_agg = self.types_agg(neg_id_batch, 1)
        elif type == 2:
            c_agg = self.types_agg(c_id_batch, 1)
            p_agg = self.types_agg(pos_id_batch, 2)
            n_agg = self.types_agg(neg_id_batch, 2)
        else:
            c_agg = self.types_agg(c_id_batch, 2)
            p_agg = self.types_agg(pos_id_batch, 1)
            n_agg = self.types_agg(neg_id_batch, 1)

        return c_agg, p_agg, n_agg

    def forward(self, type_data, type, fighting):

        c_id_batch = [int(x[0]) for x in type_data[0]]
        pos_id_batch = [int(x[1]) for x in type_data[0]]
        neg_id_batch = [int(x[2]) for x in type_data[0]]

        c_pre, p_pre, n_pre = self.whole_agg(c_id_batch, pos_id_batch, neg_id_batch, type)




        return c_pre, p_pre, n_pre


def cross_entropy_loss(uu_embed_batch, ui_embed_batch, iu_embed_batch, embed_d):


    c_uu_embed_batch = uu_embed_batch[0]
    p_uu_embed_batch = uu_embed_batch[1]
    n_uu_embed_batch = uu_embed_batch[2]
    c_ui_embed_batch = ui_embed_batch[0]
    p_ui_embed_batch = ui_embed_batch[1]
    n_ui_embed_batch = ui_embed_batch[2]
    c_iu_embed_batch = iu_embed_batch[0]
    p_iu_embed_batch = iu_embed_batch[1]
    n_iu_embed_batch = iu_embed_batch[2]
    batch_size = c_uu_embed_batch.shape[0] + c_ui_embed_batch.shape[0] + c_iu_embed_batch.shape[0]

    c_embed_batch = torch.cat([c_uu_embed_batch, c_ui_embed_batch, c_iu_embed_batch], axis = 0).view(batch_size, 1, embed_d)
    p_embed_batch = torch.cat([p_uu_embed_batch, p_ui_embed_batch, p_iu_embed_batch], axis = 0).view(batch_size, embed_d, 1)
    n_embed_batch = torch.cat([n_uu_embed_batch, n_ui_embed_batch, n_iu_embed_batch], axis = 0).view(batch_size, embed_d, 1)
    out_p = torch.bmm(c_embed_batch, p_embed_batch)
    out_n = - torch.bmm(c_embed_batch, n_embed_batch)

    sum_p = F.logsigmoid(out_p)
    sum_n = F.logsigmoid(out_n)
    loss_sum = - (sum_p + sum_n)

    # loss_sum = loss_sum.sum() / batch_size

    return loss_sum.mean()

    # batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]
    #
    # c_embed = c_embed_batch.view(batch_size, 1, embed_d)
    # pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
    # neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)
    # # 可以看出论文中公式8是想要根据softmax确保中心节点和正样本节点的概率更大，所以公式等同于让
    # # sum_p更大，sum_n更小
    # out_p = torch.bmm(c_embed, pos_embed)
    # out_n = - torch.bmm(c_embed, neg_embed)
    #
    # sum_p = F.logsigmoid(out_p)
    # sum_n = F.logsigmoid(out_n)
    # loss_sum = - (sum_p + sum_n)
    #
    # # loss_sum = loss_sum.sum() / batch_size
    #
    # return loss_sum.mean()













