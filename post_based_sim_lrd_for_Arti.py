# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import shutil

import torch


from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.index import build_index_helper, feature_loader
from pyretri.evaluate import build_evaluate_helper
import pandas as pd
from pyretri.index.metric import KNN
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--config_file', '-cfg', default='/home/data1/changhao/iBioHash/Codes/pytorch-image-models-main/post_config/market_w_tricks.yaml', metavar='FILE', type=str, help='path to config file')
    args = parser.parse_args()
    return args

feat_dir_dict = {
    "/home/data1/changhao/iBioHash/Results/similarity/same_domin/eva_ckp_75_dba1_qe12": 1,
    }

def main():

    # init args
    args = parse_args()
    assert args.config_file is not None, 'a config file must be provided!'

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.config_file, args.opts)

    similarity=0

    for feat_dir, weight in feat_dir_dict.items():
        # load features
        query_fea, gallery_fea, query_names, gallery_names = feature_loader.load(feat_dir)
        #---------------
        # query_fea = query_fea[:10][:]
        # gallery_fea = gallery_fea[:200][:]
        query_fea, gallery_fea = torch.Tensor(query_fea), torch.Tensor(gallery_fea)

        if torch.cuda.is_available():
            query_fea = query_fea.cuda()
            gallery_fea = gallery_fea.cuda()
            print("loading to the GPU")
        print(query_fea.shape)

        metric = KNN()
        dis, sorted_index = metric(query_fea, query_fea) # + 2

        # ======== query 分组 ==============
        mode_split = 'method2_10'
        if mode_split == 'method1':  # 分组后，组内top必须严格相同
            skip_i = []
            # group_4 = []
            # for i in range(len(sorted_index)):
            #     if i in skip_i:
            #         continue
            #     # top3-group
            #     top4_index = sorted_index[i][:4]
            #     temp1_index = sorted_index[top4_index[1]][:4].sort().values
            #     temp2_index = sorted_index[top4_index[2]][:4].sort().values
            #     temp3_index = sorted_index[top4_index[3]][:4].sort().values

            #     # 判断是否互相包含
            #     if top4_index.equal(temp1_index) and top4_index.equal(temp2_index) and top4_index.equal(temp3_index):
            #         skip_i.extend([i, int(top4_index[1]), int(top4_index[2]), int(top4_index[3])])  # 包含则之后跳过处琄1�7
            #         group_4.append([i, int(top4_index[1]), int(top4_index[2]), int(top4_index[3])])
            group_3 = []
            for i in range(len(sorted_index)):
                if i in skip_i:
                    continue
                # top3-group
                top3_index = sorted_index[i][:3]
                temp1_index = sorted_index[top3_index[1]][:3].sort().values
                temp2_index = sorted_index[top3_index[2]][:3].sort().values

                # 判断是否互相包含
                if top3_index.equal(temp1_index) and top3_index.equal(temp2_index):
                    skip_i.extend([i, int(top3_index[1]), int(top3_index[2])])  # 包含则之后跳过处琄1�7
                    group_3.append([i, int(top3_index[1]), int(top3_index[2])])

            group_2 = []
            for i in range(len(sorted_index)):
                if i in skip_i:
                    continue
                # top3-group
                top2_index = sorted_index[i][:2]
                temp1_index = sorted_index[top2_index[1]][:2].sort().values

                # 判断是否互相包含
                if top2_index.equal(temp1_index):
                    skip_i.extend([i, int(top2_index[1])])  # 包含则之后跳过处琄1�7
                    group_2.append([i, int(top2_index[1])])
            # print(group_3) # 有近300组，900个query, 6000个gallery
            # print(group_2)  # 有近1837组，3674个query＄1�736740个gallery
            # 可解冄1�74574个query，剩佄1�75426个各戄1�71组，只能使用剩余的1�747260个gallery，明显不够��1�7
            # 扢�以需要放宽条件，改为＄1�75丄1�71组，只要互相之间朄1�74个相同即可；4个一组，只要互相朄1�73个相同即可；3个一组，只要互相朄1�72个相同即可；2个一组，彼此互为top1
        elif mode_split == 'method2':
            # 放宽条件，不必完全相同，可有1个不各1�7
            skip_i = []
            group_select = []
            for topk_select in range(5, 2, -1):
                group_topk_select = []
                # topk_select = 5
                print('topk: ' +str(topk_select))
                for i in range(len(sorted_index)):
                    if i in skip_i:
                        continue
                    # top3-group
                    topk_index = sorted_index[i][:topk_select]
                    
                    temp_index = []

                    for top_i in range(len(topk_index) - 1):
                        temp_index.append(sorted_index[topk_index[top_i + 1]][:topk_select].sort().values)

                    # 判断是否互相至少包含topk-1个相同的query 
                    elements = dict()  # 统计该组query出现次数
                    temp_index.append(topk_index.sort().values)
                    for ti in range(len(temp_index)):
                        for q_temp in temp_index[ti]:
                            q_temp = int(q_temp)
                            elements[q_temp] = elements.get(q_temp, 0) + 1
                    cnt = 0  # 统计出现topk_select次的元素个数

                    flag_re = False
                    for el in elements:
                        if elements[el] >= topk_select - 1:
                            cnt += 1
                    if cnt >= topk_select - 1:  # 满足互相至少包含topk-1个相同的query
                        # 排除两组之间有重复元素的情况
                        flag_re = False  # 重复标志佄1�7
                        for s_index in topk_index:
                            if s_index in skip_i:
                                flag_re = True
                                break
                        if flag_re:
                            print("re condition")
                            continue  # 跳过该query，继续寻扄1�7
                        group_topk_select.append([int(s_index) for s_index in topk_index])
                        skip_i.extend([int(s_index) for s_index in topk_index])  # 之后跳过处理
                group_select.append(group_topk_select)

            group_2 = []
            for i in range(len(sorted_index)):
                if i in skip_i:
                    continue
                # top3-group
                top2_index = sorted_index[i][:2]
                temp1_index = sorted_index[top2_index[1]][:2].sort().values

                # 判断是否互相包含
                if top2_index.equal(temp1_index):
                    # 排除两组之间有重复元素的情况
                    flag_re = False  # 重复标志佄1�7
                    if int(top2_index[1]) in skip_i:
                        flag_re = True
                        continue  # 跳过该query，继续寻扄1�7
                    skip_i.extend([i, int(top2_index[1])])  # 包含则之后跳过处琄1�7
                    group_2.append([i, int(top2_index[1])])
            group_select.append(group_2)
            # 5:929组，4645个query，需18580个gallery＄1�7
            # 4＄1�7193组，772个query，需3860个gallery＄1�7
            # 3＄1�7413组，1239个query，需8260个gallery＄1�7
            # 2＄1�7203组，406个query，需4060个gallery＄1�7
            # 1＄1�72938组，2938个query，需58760个gallery＄1�7 共需93520个gallery，有1w的缺口，但大致ok
            
            # 棢�查group_select中是否有相同元素
            # skip_i.sort()
            # for kk in range(len(skip_i)-1):
            #     if skip_i[kk] == skip_i[kk + 1]:
            #         print('error')
            # 已检查，前面正确
        elif mode_split == 'method2_10':
            # 放宽条件，不必完全相同，可有1个不同
            skip_i = []
            group_select = [[] for i in range(11)]
            # 寻找公共子集
            for topk_select in range(10, 5, -1):   # 从10个一组开始寻找，找组内的公共子集，
                group_topk_select = []
                # topk_select = 5
                print('topk: ' +str(topk_select))
                for i in range(len(sorted_index)):
                    if i in skip_i:
                        continue
        
                    topk_index = sorted_index[i][:topk_select]
                    
                    temp_index = []
                    for top_i in range(len(topk_index) - 1):
                        temp_index.append(sorted_index[topk_index[top_i + 1]][:topk_select].sort().values)

                    elements = dict()  # 统计该组query出现次数
                    temp_index.append(topk_index.sort().values)
                    for ti in range(len(temp_index)):
                        for q_temp in temp_index[ti]:
                            q_temp = int(q_temp)
                            elements[q_temp] = elements.get(q_temp, 0) + 1

                    # cnt = 0  # 统计出现topk_select次的元素个数
                    # for el in elements: 
                    #     if elements[el] >= topk_select - 1:
                    #         cnt += 1
                    # if cnt >= topk_select - 1:  # 满足互相至少包含topk-1个相同的query
                    #     # 排除两组之间有重复元素的情况
                    #     flag_re = False  # 重复标志佄1�7
                    #     for s_index in topk_index:
                    #         if s_index in skip_i:
                    #             flag_re = True
                    #             break
                    #     if flag_re:
                    #         print("re condition")
                    #         continue  # 跳过该query，继续寻扄1�7
                    #     group_topk_select.append([int(s_index) for s_index in topk_index])
                    #     skip_i.extend([int(s_index) for s_index in topk_index])  # 之后跳过处理

                    # 不再判断该组是否满足条件，而是寻找公共子集
                    public_subset = []
                    for el in elements: 
                        if elements[el] >= topk_select - 2: # 如topk_select=10，若某个query出现8次，则认为属于公共子集
                            public_subset.append(el)
                    
                    if len(public_subset) < topk_select - 2: # 如topk_select=10，若公共子集长度小于8，则认为质量不太高
                        continue
                    
                    flag_re = False
                    for s_index in public_subset:
                        if s_index in skip_i:
                            flag_re = True
                            break
                    if flag_re:
                        # print("re condition")
                        continue  # 跳过该query，继续寻找
                    group_topk_select.append([int(s_index) for s_index in public_subset])
                    skip_i.extend([int(s_index) for s_index in public_subset])  # 之后跳过处理

                # 因公共子集长度不确定，因此遍历加入group_select
                for gts in range(len(group_topk_select)):
                    length_gts = len(group_topk_select[gts])
                    group_select[length_gts].append(group_topk_select[gts])

            # 3组为严格互为top3
            group_3 = []
            for i in range(len(sorted_index)):
                if i in skip_i:
                    continue
                # top3-group
                top3_index = sorted_index[i][:3]
                temp1_index = sorted_index[top3_index[1]][:3].sort().values
                temp2_index = sorted_index[top3_index[2]][:3].sort().values

                # 判断是否互相包含
                if top3_index.equal(temp1_index) and top3_index.equal(temp2_index):
                    flag_re = False  # 重复标志位
                    if int(top3_index[1]) in skip_i or int(top3_index[2]) in skip_i:
                        flag_re = True
                        continue  # 跳过该query，继续寻找
                    skip_i.extend([i, int(top3_index[1]), int(top3_index[2])])  # 包含则之后跳过处理
                    group_3.append([i, int(top3_index[1]), int(top3_index[2])])
            group_select[3] = group_3

            # 2组必须严格互为top1
            group_2 = []
            for i in range(len(sorted_index)):
                if i in skip_i:
                    continue
                # top3-group
                top2_index = sorted_index[i][:2]
                temp1_index = sorted_index[top2_index[1]][:2].sort().values

                # 判断是否互相包含
                if top2_index.equal(temp1_index):
                    # 排除两组之间有重复元素的情况
                    flag_re = False   # 重复标志位
                    if int(top2_index[1]) in skip_i:
                        flag_re = True
                        continue  # 跳过该query，继续寻找
                    skip_i.extend([i, int(top2_index[1])]) # 包含则之后跳过处理
                    group_2.append([i, int(top2_index[1])])
            group_select[2] = group_2

            stat_q_num = 0
            stat_g_num = 0
            for i in range(2, len(group_select)):
                stat_q_num += (i * len(group_select[i]))
                stat_g_num += (20 * len(group_select[i]))
                print("{}: {}组, {}个query, 需{}个gallery".format(i, len(group_select[i]), i * len(group_select[i]), 20 * len(group_select[i])))
            print("1: 1组, {}个query, 需{}个gallery".format(10000 - stat_q_num, 20*(10000 - stat_q_num)))

            # 先不处理1组，先交一版
            group_select.pop(0)
            group_select.pop(0)

            
            # re_assign_single_q = {}
            # for i in range(10000):
            #     if i in skip_i:
            #         continue
                
            #     sim_among_group = [[] for i in range(9)]  # 记录单个query和每组的平均相似度
            #     for group_topk_select_i in range(len(group_select)): # 先从2组开始，一直遍历到10组
            #         group_topk_select = group_select[group_topk_select_i]

            #         for group_i in range(len(group_topk_select)):
            #             avg_sim = 0
            #             for q_index in group_topk_select[group_i]:
            #                 avg_sim += dis[i][q_index]
            #             avg_sim = avg_sim / float(len(group_topk_select[group_i]))
            #             sim_among_group[group_topk_select_i].append(avg_sim)


            #     # 遍历 sim_among_group，寻找相似度最大的组所在的位置
            #     max_sim, max_i, max_j = -float('inf'), 0, 0
            #     for sim_i in range(len(sim_among_group)):
            #         for sim_j in range(len(sim_among_group[sim_i])):
            #             if sim_among_group[sim_i][sim_j] > max_sim:
            #                 max_sim = sim_among_group[sim_i][sim_j]
            #                 max_i = sim_i
            #                 max_j = sim_j
            #     # 记录要加入的组
            #     re_assign_single_q[i] = (max_i, max_j)
            
            # # 将单个query合并到2及以上组 
            # for single_q in re_assign_single_q:
            #     re_assign_i, re_assign_j = re_assign_single_q[single_q]
            #     group_select[re_assign_i][re_assign_j].append(single_q)
            #     skip_i.append(single_q)

        elif mode_split == 'method3':  # 使用聚类对query分组，有个问题是聚类很难考虑到对称��，即q1的最相似是q2，但q2的最相似不一定是q1
            
            # 层次聚类
            # model = AgglomerativeClustering(affinity="precomputed", linkage='complete', n_clusters = 1000)
            # dis = -(dis-1) # 将相似度转换为距禄1�7
            # model = model.fit(dis.cpu().numpy())
            # # 0 的最相似应为 0, 1228, 2216, 3822, 3855
            # # for i in range(10000):
            # #     if model.labels_[i] == model.labels_[0]: print(i)
            # # 经检查，效果很不稳定，簇大的大，小的射1�7

            # 谱聚籄1�7
            model = SpectralClustering(affinity = 'precomputed', n_clusters=1000,)
            dis = dis + 2
            model = model.fit(dis.cpu().numpy())
            # 0 的结果是＄1�70, 468, 1228, 1605, 2174, 2216, 3822, 3855, 4759, 5522。效果还可以, 但人工对比后，也不是特别靠谱感觉，后期可以尝试，10min内出结果〄1�7
            print("-")

        # ======= gallery 分配 =======
        qg_dis, qg_sorted_index = metric(query_fea, gallery_fea) # + 2
        qg_sorted_index_assign = torch.zeros((10000, 30), dtype=int)
        used_gallery_index = []
        
        # group_topk_select = group_select[0]
        for group_idx in range(9): # topk 5-2, # 调换分配顺序，先分配10组，最后分配2组
            group_topk_select = group_select[group_idx]
            # 逐组分配gallery
            for group_i in range(len(group_topk_select)):
                used_gallery_index_pergroup = []  # 该组共享gallery index
                current_gallery_point = [0] * len(group_topk_select[group_i])

                # 思路有问预1�7
                # 逐top分配gallery
                # for topk_i in range(20):
                #     for q_index_i in range(len(group_3[group_i])):
                #         cur_q_index = group_3[group_i][q_index_i]  # 当前query id
                #         cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]])  # 当前gallery id
                #         # while cur_g_index in used_gallery_index:  # 该gallery id若已分配，��路有问题，同一组应该共享gallery
                #         #     current_gallery_point[q_index_i] += 1  # query对应gallery的指钄1�7 + 1
                #         #     cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]])  # 更新cur_g_index
                #         # 分配未使用的 gallery
                #         qg_sorted_index_assign[cur_q_index][topk_i] = cur_g_index
                #         used_gallery_index.append(cur_g_index)

                # 方法一： 取一组内不重复的最相似的gallery；方法二：取出现次数最多的gallery
                # 方法一
                while len(used_gallery_index_pergroup) != 30:
                    for q_index_i in range(len(group_topk_select[group_i])):
                        cur_q_index = group_topk_select[group_i][q_index_i]
                        cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]])  # 当前gallery id
                        
                        while cur_g_index in used_gallery_index:  # gallery已分配过，从总的里面看
                            current_gallery_point[q_index_i] += 1
                            cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]])
                        
                        # 分配未使用的 gallery
                        # qg_sorted_index_assign[cur_q_index][topk_i] = cur_g_index
                        used_gallery_index_pergroup.append(cur_g_index)
                        used_gallery_index.append(cur_g_index)

                        if len(used_gallery_index_pergroup) == 30:
                            break

                g_sorted_index_pergroup = torch.tensor(used_gallery_index_pergroup)
                for q_index_i in range(len(group_topk_select[group_i])):
                    cur_q_index = group_topk_select[group_i][q_index_i]
                    qg_sorted_index_assign[cur_q_index] = g_sorted_index_pergroup

        # # ----------------5/7_lrd----------------------#
        re_assign_single_q = {}
        sim_cnt_thr = 10
        for i in range(10000):
            if i in skip_i:
                continue
            
            sim_among_group = [[] for i in range(9)]  # 记录单个query的top40和每组已分配的gallery交集数量
            for group_topk_select_i in range(len(group_select)): # 先从2组开始，一直遍历到10组
                group_topk_select = group_select[group_topk_select_i]

                for group_i in range(len(group_topk_select)):
                    # sim_cnt = 0
                    q_idex = group_topk_select[group_i][0]
                    gallery_index_set = set(qg_sorted_index_assign[q_idex].numpy())
                    cur_q_sim_results_set = set(qg_sorted_index[i][:40].numpy())
                    sim_cnt = len(gallery_index_set.intersection(cur_q_sim_results_set))
                    
                    sim_among_group[group_topk_select_i].append(sim_cnt)

                    # print('debug')
                    # for q_index in group_topk_select[group_i]:
                    #     avg_sim += dis[i][q_index]
                    # avg_sim = avg_sim / float(len(group_topk_select[group_i]))
                    # sim_among_group[group_topk_select_i].append(avg_sim)


            # 遍历 sim_among_group，寻找交集数量最大的组所在的位置
            max_sim, max_i, max_j = -float('inf'), 0, 0
            for sim_i in range(len(sim_among_group)):
                for sim_j in range(len(sim_among_group[sim_i])):
                    if sim_among_group[sim_i][sim_j] > max_sim:
                        max_sim = sim_among_group[sim_i][sim_j]
                        max_i = sim_i
                        max_j = sim_j
            # 如果max_sim>=交集数量阈值, 记录要加入的组
            if max_sim >= sim_cnt_thr:
                re_assign_single_q[i] = (max_i, max_j)
        
        print('单个query在交集数量阈值={}下，成功合并的数量={}'.format(sim_cnt_thr, len(re_assign_single_q)))
        # 将单个query合并到2及以上组 
        for single_q in re_assign_single_q:
            re_assign_i, re_assign_j = re_assign_single_q[single_q]
            group_select[re_assign_i][re_assign_j].append(single_q)
            # 对单个query赋gallery
            qg_sorted_index_assign[single_q] = qg_sorted_index_assign[group_select[re_assign_i][re_assign_j][0]]
            skip_i.append(single_q)

        # 整理分类训练文件夹
        print('debug')
        dir_root = '/home/data1/changhao/iBioHash/Datasets/iBioHash_arti_dataset'
        src_root = '/home/data1/changhao/iBioHash/Datasets'
        for i in range(2, len(group_select) + 2):
            for j in range(len(group_select[i - 2])):
                cur_dir = os.path.join(dir_root, str(i) + '_' + str(j))
                os.makedirs(cur_dir, exist_ok=True)
                add_g_flag = False
                for img_idx in group_select[i - 2][j]:
                    
                    img_name_old = query_names[img_idx]
                    img_name_new = 'query_' + query_names[img_idx]
                    src_path = os.path.join(src_root, 'iBioHash_Query/Query', img_name_old)
                    shutil.copy(src_path, os.path.join(cur_dir, img_name_new))

                    if not add_g_flag:
                        tmg_g_idx_list = qg_sorted_index_assign[img_idx]
                        for g_idx in tmg_g_idx_list:
                            img_name_old = gallery_names[g_idx] + '.jpg'
                            img_name_new = 'gallery_' + gallery_names[g_idx] + '.jpg'
                            src_path = os.path.join(src_root, 'iBioHash_Gallery/Gallery', img_name_old)
                            shutil.copy(src_path, os.path.join(cur_dir, img_name_new))
                        add_g_flag = True
        ## 处理单个query
        single_q_cnt = 0
        for i in range(10000):
            if i in skip_i:
                continue
            single_q_cnt += 1
            cur_query_name = query_names[i]
            cur_dir = os.path.join(dir_root, cur_query_name.split('.')[0])

            os.makedirs(cur_dir, exist_ok=True)
            shutil.copy(os.path.join(src_root, 'iBioHash_Query/Query', cur_query_name), cur_dir)

            q_tmp_idx = sorted_index[i][:20]
            g_tmp_idx = qg_sorted_index[i][:40]

            for q_idx in q_tmp_idx:
                q_name_old = query_names[q_idx]
                q_name_new = 'query' + query_names[q_idx]

                shutil.copy(os.path.join(src_root, 'iBioHash_Query/Query', q_name_old), os.path.join(cur_dir, q_name_new))
            
            for g_idx in g_tmp_idx:
                g_name_old = gallery_names[g_idx] + '.jpg'
                g_name_new = 'gallery' + gallery_names[g_idx] + '.jpg'

                shutil.copy(os.path.join(src_root, 'iBioHash_Gallery/Gallery', g_name_old), os.path.join(cur_dir, g_name_new))
        print('单query有{}个'.format(single_q_cnt))



                                



        # 05/06 ------------- 开始处理1组 ---------------
        # 将单个的query分配给 已分好的组


        # # 分配单个的query, 应当成一个大组
        # group_single_select = []
        # for i in range(10000):
        #     if i in skip_i:
        #         continue
        #     group_single_select.append(i)
        
        # used_gallery_index_pergroup = []  # 该组共享gallery index
        # current_gallery_point = [0] * len(group_single_select)

        # # 逐top分配gallery, 因为此刻并非共享
        # for topk_i in range(20):
        #     for q_index_i in range(len(group_single_select)):
        #         cur_q_index = group_single_select[q_index_i]  # 当前query id

        #         if cur_q_index in skip_i:
        #             continue

        #         cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]])  # 当前gallery id
        #         while cur_g_index in used_gallery_index:  # 该gallery id若已分配
        #             current_gallery_point[q_index_i] += 1  # query对应gallery的指钄1�7 + 1
        #             # if current_gallery_point[q_index_i] >= 82424: # current_gallery_point[q_index_i]可能越界
        #             #     current_gallery_point[q_index_i] = 82423
        #             #     break 
        #             if current_gallery_point[q_index_i] >= 270: # 1000个类别中，起码跨越了3个类, 再往外找应该也是错的亄1�7
        #                 skip_i.append(cur_q_index)
        #                 break
        #             cur_g_index = int(qg_sorted_index[cur_q_index][current_gallery_point[q_index_i]])  # 更新cur_g_index
                
        #         if cur_q_index in skip_i:
        #             continue
                
        #         # 分配未使用的 gallery
        #         qg_sorted_index_assign[cur_q_index][topk_i] = cur_g_index
        #         used_gallery_index.append(cur_g_index)

        # print("assign gallery done")

        # retreival_results = []
        # # 输出 qg_sorted_index_assign 
        # for i in range(qg_sorted_index_assign.size(0)):
        #     temp_idx = qg_sorted_index_assign[i].tolist()
        #     temp_name_list = [gallery_names[j] for j in temp_idx]
        #     retreival_results.append(' '.join(temp_name_list))
        
        # for 1697 
        # for i in range(qg_sorted_index_assign.size(0)):
        #     if i not in re_assign_single_q:
        #         retreival_results.append('')
        #     else:
        #         temp_idx = qg_sorted_index_assign[i].tolist()
        #         temp_name_list = [gallery_names[j] for j in temp_idx]
        #         retreival_results.append(' '.join(temp_name_list))

        # result_dict = pd.DataFrame({'Id':query_names,'Predicted':retreival_results})
        # # result_dict.to_csv('denseFull.csv', index=False)
        # # result_dict[['Id','Predicted']].to_csv('./submited/{}.csv'.format(args.file_name), index=False)
        # result_dict[['Id','Predicted']].to_csv(os.path.join('/home/data1/changhao/iBioHash/Codes/pytorch-image-models-main/lrd_post_sim_submit', 'submit_post_based_sim_method21_eva82.7_thr5_top40.csv'), index=False)
    
            
        # # build helper and index features
        # index_helper = build_index_helper(cfg.index)
        # index_result_info, query_fea, gallery_fea, dis = index_helper.do_index(query_fea, query_names, gallery_fea)

        # # with open(os.path.join(feat_dir,'gallery_dba.pkl'),'wb') as f:
        # #     pickle.dump(gallery_fea.cpu().numpy(), f)
        
        # # with open(os.path.join(feat_dir,'query_qe.pkl'),'wb') as f:
        # #     pickle.dump(query_fea.cpu().numpy(), f)

        # with open(os.path.join('/home/data1/changhao/iBioHash/Results/similarity','similarity_eva_dba15_qe15.pkl'),'wb') as f:
        #     pickle.dump(dis, f,protocol=4)
    
        # # similarity+=(dis*weight)

        # # print(index_result_info[0])

        # del index_result_info, query_fea, gallery_fea, dis

    # retreival_results = []
    # similarity_index = dis.topk(k=20, dim=1)[1]  # indices
    # for i in range(similarity_index.size(0)):
    #     temp_idx = similarity_index[i].tolist()
    #     temp_name_list = [gallery_names[j] for j in temp_idx]
    #     retreival_results.append(' '.join(temp_name_list))
    # result_dict = pd.DataFrame({'Id':query_names,'Predicted':retreival_results})
    # # result_dict.to_csv('denseFull.csv', index=False)
    # # result_dict[['Id','Predicted']].to_csv('./submited/{}.csv'.format(args.file_name), index=False)
    # result_dict[['Id','Predicted']].to_csv(os.path.join('/home/data1/changhao/iBioHash/Results/similarity', 'submit.csv'), index=False)


    # del result_dict
    # del similarity_index
    
        # # build helper and evaluate results
        # evaluate_helper = build_evaluate_helper(cfg.evaluate)
        # mAP, recall_at_k = evaluate_helper.do_eval(index_result_info, gallery_names)

        # # show results
        # evaluate_helper.show_results(mAP, recall_at_k)


if __name__ == '__main__':
    main()

