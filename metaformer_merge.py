import json
import pandas as pd
import copy

with open('/home/data1/changhao/iBioHash/Datasets/iNaturelist/train_mini.json', encoding='utf-8') as a:
    result = json.load(a)

    categories = result['categories']

metaformer_result = pd.read_csv('/home/data/lrd/MetaFormer/result/query_gallery.csv')
metaformer_merge_result = copy.deepcopy(metaformer_result)

classes_images = dict()
species_images = dict()
genus_images = dict()
family_images = dict()
order_images = dict()

species_genus = dict()
genus_family = dict()
family_order = dict()

# exist_list = []
# for item in categories:
#     if item['name'] in exist_list:
#         print('re!')
#     else:
#         exist_list.append(item['name'])
# print(len(exist_list))

for i in range(len(metaformer_result)):

    img_id = metaformer_result.iloc()[i][0]
    pre_label = metaformer_result.iloc()[i][1]
    info = categories[pre_label]
    
    # 添加路径
    classes_images[info['name']] = classes_images.get(info['name'], [])
    classes_images[info['name']].append(img_id)
    # species_images[info['name']] = species_images.get(info['name'], [])
    # species_images[info['name']].append(img_id)
    # genus_images[info['genus']] = genus_images.get(info['genus'], [])
    # genus_images[info['genus']].append(img_id)
    # family_images[info['family']] = family_images.get(info['family'], [])
    # family_images[info['family']].append(img_id)
    # order_images[info['order']] = order_images.get(info['order'], [])
    # order_images[info['order']].append(img_id)

    # 构建映射关系
    species_genus[info['specific_epithet']] = species_genus.get(info['specific_epithet'], info['genus'])
    genus_family[info['genus']] = genus_family.get(info['genus'], info['family'])
    family_order[info['family']] = family_order.get(info['family'], info['order'])

    metaformer_merge_result.iloc[i,1] = info['name']

# 合并类别
print('-')
cnt_s = 0
for c_i in classes_images:
    if len(classes_images[c_i]) <= 5:  # "种"图片小于5张
        for img_id in classes_images[c_i]:
            cur_species = metaformer_merge_result.loc[metaformer_merge_result['image_id'] == img_id, 'class_id'].iloc[0].split(' ')[-1]
            metaformer_merge_result.loc[metaformer_merge_result['image_id'] == img_id, 'class_id'] = cur_species
            species_images[cur_species] = species_images.get(cur_species, [])
            species_images[cur_species].append(img_id)
            # species_images[s_i].remove(img_id)
        
        cnt_s += 1

cnt_g = 0
for s_i in species_images:
    if len(species_images[s_i]) <= 5:  # "属"图片小于5张
        for img_id in species_images[s_i]:
            metaformer_merge_result.loc[metaformer_merge_result['image_id'] == img_id, 'class_id'] = species_genus[s_i]
            genus_images[species_genus[s_i]] = genus_images.get(species_genus[s_i], [])
            genus_images[species_genus[s_i]].append(img_id)
            # genus_images[g_i].remove(img_id)
        cnt_g += 1

print('被合并的类数:{}, 被合并的种数:{}'.format(cnt_s, cnt_g))
metaformer_merge_result.to_csv('/home/data1/changhao/iBioHash/Datasets/metaformer_merge_cz5_result.csv', index=False)