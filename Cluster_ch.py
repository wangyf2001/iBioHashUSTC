from sklearn.cluster import SpectralClustering, kmeans_plusplus, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import shutil
import os
import pickle

num_clusters=2000

# 加载eva 跨域特征
with open('/home/data1/changhao/iBioHash/Results/features/eva_best_cross_domain_90000_e1/gallery.pkl', 'rb') as f:
    # Load object from file
    gallery = pickle.load(f)
    
with open('/home/data1/changhao/iBioHash/Results/features/eva_best_cross_domain_90000_e1/query.pkl', 'rb') as f:
    # Load object from file
    query = pickle.load(f)

with open('/home/data1/changhao/iBioHash/Results/features/eva_best_cross_domain_90000_e1/names.pkl', 'rb') as f:
    # Load object from file
    names = pickle.load(f)
    
output_folder='/home/data1/zgp/spectral_kmeans_4250_qg_beit_24ep'
query_samples = names["query"].samples
gallery_samples = names["gallery"].samples 
query_image_names = [x[0] for x in query_samples]
gallery_image_names = [x[0] for x in gallery_samples]
query_image_names.extend(gallery_image_names)
# 将所有特征向量合并到一个大数组中
all_features = np.concatenate((query, gallery), axis=0)
# similarity_matrix = cosine_similarity(all_features)
# similarity_matrix = np.nan_to_num(similarity_matrix,nan=0.0, posinf=1.0, neginf=0.0)

# 聚类
# SpectralClustering
# clustering = SpectralClustering(n_clusters=num_clusters, n_neighbors=20, assign_labels='discretize')
# clustering = kmeans_plusplus(X=all_features, n_clusters=num_clusters)  # 确实从来没设过种子，kmeans的影响很大。

# 
kmeans = KMeans(n_clusters=num_clusters, n_init="auto").fit(all_features)


# 将特征传递给算法进行聚类
clustering.fit(all_features)
# 获取聚类标签
labels = clustering.labels_
print(labels)

# 将每个图片文件分配到它所属的聚类，并保存到相应的文件夹中
# for i, file_name in enumerate(query_image_names):
#     cluster_label = labels[i]
#     label_folder = os.path.join(output_folder, f"cluster_{cluster_label}")
#     if not os.path.exists(label_folder):
#         os.makedirs(label_folder)
#     file_name2 = os.path.basename(file_name)
#     label_name=label_folder+'/'+file_name2
#     shutil.copyfile(file_name,label_name)