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
from getData import get_last_data
from model import HetAD
from model import cross_entropy_loss
from data_input import get_train_data

if __name__ == '__main__':
    args = Parser.Define_Params()
    embed_dim = args.embed_dim
    uu_neigh, ui_neigh, iu_neigh, ii_neigh, _, _ = get_last_data()
    # User_num = len(ui_neigh)
    # Item_num = len(iu_neigh)
    # start_node_embedding = args.start_node_embedding
    # u_self_embed = np.zeros((User_num + 1, embed_dim))
    # i_self_embed = np.zeros((Item_num + 1, embed_dim))
    # with open(start_node_embedding, 'r') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         line = line.strip('\n')
    #         line = line.split('  ')
    #         embed = line[1].split(', ')
    #         embeds = np.asarray(embed, dtype='float32')
    #         if line[0][0] == 'u':
    #             u_self_embed[int(line[0][1:])] = embeds
    #         elif line[0][0] == 'i':
    #             i_self_embed[int(line[0][1:])] = embeds
    #
    # # 这个是本身的权重，然后它的属性例如他评论的项目，的权重相加取均值把然后在聚合
    #
    # print(u_self_embed)

    model = HetAD(embed_dim, uu_neigh, ui_neigh, iu_neigh, ii_neigh)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    optim = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0)
    model.init_weights()


    # triple_list = get_train_data()
    # uu_train_set = torch.utils.data.TensorDataset(torch.LongTensor(triple_list[0]))
    # uu_train_loader = torch.utils.data.DataLoader(uu_train_set, batch_size=args.batch_size, shuffle=True)

    # for type_data in uu_train_loader:
    #
    #     c_id_batch = [int(x[0]) for x in type_data[0]]
    #     pos_id_batch = [int(x[1]) for x in type_data[0]]
    #     neg_id_batch = [int(x[2]) for x in type_data[0]]
    #     #
    #     # model.u_content_agg(c_id_batch)
    #     # model.i_content_agg(c_id_batch)
    #     print(c_id_batch)
    #     embed = model.types_agg(c_id_batch, 1)
    #     print(embed.shape)
    #     for single_embed in embed:
    #         print(single_embed)
    #         print(single_embed.detach().numpy().tolist())
    #         exit(0)
    #
    #     exit(0)
    # c_out = torch.zeros([3, 500, embed_dim])
    # a_out = []
    # a_out.append([0] * 500)
    # a_out.append([0] * 230)
    # a_out.append([0] * 210)
    # print(c_out)
    # for i in range(len(a_out)):
    #     for j in range(len(a_out[i])):
    #         a_out[i][j] = np.zeros(128)
    # a_out = torch.tensor(a_out)
    # print(a_out)
    # exit(0)
    # x = 0
    # y = 0
    # z = 0
    # z1 = 0
    # for i in range(100):
    #     triple_list = get_train_data()
    #     x = x + len(triple_list[0])
    #     y = y + len(triple_list[1])
    #     z = z + len(triple_list[2])
    #     z1 = z1 + len(triple_list[3])
    #     print(len(triple_list[0]))
    #     print(len(triple_list[1]))
    #     print(len(triple_list[2]))
    #     print(len(triple_list[3]))
    # print(x/100, y/100, z/100, z1/100)
    # exit(0)
    start_i = 1
    RESUME = True
    if RESUME:
        path_checkpoint = "./models/checkpoint/amazon_ckpt_best_newChangeModeladdII40.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optim.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_i = checkpoint['i']  # 设置开始的epoch

    for i in range(start_i, args.iter_n):
        triple_list = get_train_data()

        while len(triple_list[0]) <= 2450 or len(triple_list[1]) <= 2320 or len(triple_list[2]) <= 2070 or len(triple_list[3]) <= 2210:

            triple_list = get_train_data()

        # 各种数据的类型数目不一样所以不能用trainloader去统一，最后可以（统一一下数据数量）看一下效果
        uu_train_set = torch.utils.data.TensorDataset(torch.LongTensor(triple_list[0]))
        ui_train_set = torch.utils.data.TensorDataset(torch.LongTensor(triple_list[1]))
        iu_train_set = torch.utils.data.TensorDataset(torch.LongTensor(triple_list[2]))
        ii_train_set = torch.utils.data.TensorDataset(torch.LongTensor(triple_list[3]))

        uu_train_loader = torch.utils.data.DataLoader(uu_train_set, batch_size= 245, shuffle=True)
        ui_train_loader = torch.utils.data.DataLoader(ui_train_set, batch_size= 232, shuffle=True)
        iu_train_loader = torch.utils.data.DataLoader(iu_train_set, batch_size= 207, shuffle=True)
        ii_train_loader = torch.utils.data.DataLoader(ii_train_set, batch_size= 221, shuffle=True)

        uu_out = torch.zeros([len(triple_list), 245, embed_dim])
        ui_out = torch.zeros([len(triple_list), 232, embed_dim])
        iu_out = torch.zeros([len(triple_list), 207, embed_dim])
        ii_out = torch.zeros([len(triple_list), 221, embed_dim])
        epochs = 0
        all_loss = 0
        for uu_data, ui_data, iu_data, ii_data in zip(uu_train_loader, ui_train_loader, iu_train_loader, ii_train_loader):

            if len(uu_data[0]) == 245 and len(ui_data[0]) == 232 and len(iu_data[0]) == 207 and len(ii_data[0]) == 221:
                epochs = epochs + 1
                c_pre, p_pre, n_pre = model(uu_data, 1, i)
                #
                uu_out[0] = c_pre
                uu_out[1] = p_pre
                uu_out[2] = n_pre

                c_pre, p_pre, n_pre = model(ui_data, 2, i)
                ui_out[0] = c_pre
                ui_out[1] = p_pre
                ui_out[2] = n_pre
                c_pre, p_pre, n_pre = model(iu_data, 3, i)
                iu_out[0] = c_pre
                iu_out[1] = p_pre
                iu_out[2] = n_pre

                c_pre, p_pre, n_pre = model(ii_data, 4, i)
                ii_out[0] = c_pre
                ii_out[1] = p_pre
                ii_out[2] = n_pre

                loss = cross_entropy_loss(uu_out, ui_out, iu_out, ii_out, embed_dim)

                all_loss = all_loss + loss
                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()

                # print('after %dth iter, %d epochs, the loss is %.4f' %(i, epochs, loss))

        if i % 10 == 0:
            all_loss = all_loss/epochs
            print('after %dth iter, the whole loss is %.4f'%(i, all_loss))


        if i % 20 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optim.state_dict(),
                "i": i
            }
            if not os.path.isdir("./models/checkpoint"):
                os.makedirs("./models/checkpoint")
            torch.save(checkpoint, './models/checkpoint/amazon_ckpt_best_newChangeModeladdII%s.pth' % (str(i)))

            u_all_batch = []
            for i in range(1, args.u_l+1):
                u_all_batch.append(i)
            all_u_node = model.types_agg(u_all_batch, 1)
            i_all_batch = []
            for j in range(1, args.i_l+1):
                i_all_batch.append(j)
            all_i_node = model.types_agg(i_all_batch, 2)

            f = open('./edges/new_node_embedding5.txt', 'r+')
            f.truncate()
            f.close()
            # 然后再在这里面写
            with open('./edges/new_node_embedding5.txt', 'a') as file:
                # for i in range(len(self.u_self_embed)):
                #     file.write('u' + str(i) + '  ' + str(self.u_self_embed[i])[1: -1] + '\n')
                # for j in range(len(self.i_self_embed)):
                #     file.write('i' + str(j) + '  ' + str(self.i_self_embed[j])[1: -1] + '\n')
                u_num = 1
                i_num = 1
                for u_node in all_u_node:
                    node_embedding = u_node.detach().numpy().tolist()
                    file.write('u' + str(u_num) + '  ' + str(node_embedding)[1: -1] + '\n')
                    u_num = u_num + 1
                for i_node in all_i_node:
                    node_embedding = i_node.detach().numpy().tolist()
                    file.write('i' + str(i_num) + '  ' + str(node_embedding)[1: -1] + '\n')
                    i_num = i_num + 1
                print('更新成功')



