import pandas as pd
from tqdm import tqdm
import random

sim_result = pd.read_csv('/home/data1/changhao/iBioHash/Codes/pytorch-image-models-main/lrd_post_sim_submit/submit_post_based_sim_method21_eva82.7_thr5_top40.csv')
clust_result = pd.read_csv('/home/data1/zgp/hash_post/spectral_kmeans_2250_eva75_post4_80.32/submit.csv')

sim_group = dict()
clust_group = dict()
for i in tqdm(range(len(sim_result))):
    sim_group[sim_result.iloc[i,1]] = sim_group.get(sim_result.iloc[i,1], [])
    sim_group[sim_result.iloc[i,1]].append(sim_result.iloc[i,0])
print('共{}组'.format(len(sim_group)))  # 1195
for i in tqdm(range(len(clust_result))):
    clust_group[clust_result.iloc[i,1]] = clust_group.get(clust_result.iloc[i,1], [])
    clust_group[clust_result.iloc[i,1]].append(clust_result.iloc[i,0])
print('共{}组'.format(len(clust_group)))  # 1175


merged_group = []
ge_10 = 0
ge_9 = 0
ge_8 = 0
ge_7 = 0

for sg in tqdm(sim_group):
    sim_img = sim_group[sg]

    select_cg = ''
    select_iou = 0.  # 找最大的交并比
    for cg in clust_group:
        cg_img = clust_group[cg]

        # 计算交并比
        cur_i = len(list(set(sim_img).intersection(set(cg_img))))
        cur_u = len(list(set(sim_img).union(set(cg_img))))
        cur_iou = cur_i / float(cur_u)
        if cur_iou > select_iou:
            select_iou = cur_iou
            select_cg = cg
    # print(select_iou)

    if select_iou == 1.0:  # 符合阈值，可以合并
        merge_query = list(set(sim_img).intersection(set(clust_group[select_cg])))
        merged_group.append((merge_query, ' '.join([sg, select_cg])))
        ge_10 += 1
    if select_iou >= 0.9:  # 符合阈值，可以合并
        ge_9 += 1
    if select_iou >= 0.8:
        ge_8 += 1
    if select_iou >= 0.7:
        ge_7 += 1

print('{}组交并比等于1.0'.format(ge_10))
print('{}组交并比大于0.9'.format(ge_9))
print('{}组交并比大于0.8'.format(ge_8))
print('{}组交并比大于0.7'.format(ge_7))

new_merged_gallery = dict()

for i in range(len(merged_group)):
    gallery = merged_group[i][1].split(' ')
    gallery_1 = gallery[:20]
    gallery_2 = gallery[20:]

    # 方法一：次数从高到低，先选出现2次，再选出现1次
    new_gallery = []
    i_gallery = list(set(gallery_1).intersection(set(gallery_2)))
    u_gallery = list(set(gallery_1).union(set(gallery_2)))
    print(len(i_gallery))
    new_gallery.extend(i_gallery)  # 先加重复出现的交集
    rest_gallery = list(set(u_gallery) - set(i_gallery))

    rest_gallery_select = random.sample(rest_gallery, 20-len(i_gallery))  # 从出现1次的gallery中补够20个
    new_gallery.extend(rest_gallery_select)

    for q_id in merged_group[i][0]:
        new_merged_gallery[q_id] = ' '.join(new_gallery)

    print('-')

for i in range(len(sim_result)):
    if sim_result.iloc[i, 0] in new_merged_gallery:
        sim_result.iloc[i, 1] = new_merged_gallery[sim_result.iloc[i, 0]]



sim_result.to_csv('/home/data1/changhao/iBioHash/Codes/pytorch-image-models-main/post_merge_sim_clust.csv', index=False)



print('-')