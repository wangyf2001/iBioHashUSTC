# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import torch


from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.index import build_index_helper, feature_loader
from pyretri.evaluate import build_evaluate_helper
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--config_file', '-cfg', default='/home/data1/changhao/iBioHash/Codes/pytorch-image-models-main/post_config/market_w_tricks.yaml', metavar='FILE', type=str, help='path to config file')
    parser.add_argument('--save_path', '-sve', default=None, metavar='FILE', type=str, help='path to save')
    parser.add_argument('--feat_dir', '-fd', default='/home/data1/changhao/iBioHash/Results/features/beit_large_full_fr12_ep3', metavar='FILE', type=str, help='path to feature')
    args = parser.parse_args()
    return args

# feat_dir_dict = {
#     "/home/data1/changhao/iBioHash/Results/features/eva_large_336_fr6_e1_full": 1,

#     }

def main():

    # init args
    args = parse_args()
    assert args.config_file is not None, 'a config file must be provided!'

    feat_dir_dict = {
        # "/home/data1/changhao/iBioHash/Results/features/eva_large_336_fr6_e1_full": 1,
        args.feat_dir: 1,

    }
    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.config_file, args.opts)

    similarity=0

    # load features
    for feat_dir, weight in feat_dir_dict.items():
        query_fea, gallery_fea, query_names, gallery_names = feature_loader.load(feat_dir)
        # gallery_fea = gallery_fea[:400]


        #---------------
        # query_fea = query_fea[:10][:]
        # gallery_fea = gallery_fea[:200][:]

        # query_fea, gallery_fea = torch.Tensor(query_fea).cuda(), torch.Tensor(gallery_fea).cuda()
        print(query_fea.shape)

        # build helper and index features
        index_helper = build_index_helper(cfg.index)
        # index_result_info, query_fea, gallery_fea, dis = index_helper.do_index(query_fea, query_names, gallery_fea)
        index_result_info, query_fea, gallery_fea, dis = index_helper.do_index(query_fea, query_names, gallery_fea)
        

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        with open(os.path.join(args.save_path,'gallery_dba.pkl'),'wb') as f:
            pickle.dump(gallery_fea.cpu().numpy(), f)
        
        with open(os.path.join(args.save_path,'query_qe.pkl'),'wb') as f:
            pickle.dump(query_fea.cpu().numpy(), f)

        with open(os.path.join(args.save_path,'similarity.pkl'),'wb') as f:
            pickle.dump(dis, f,protocol=4)
    
        # similarity+=(dis*weight)

        # print(index_result_info[0])

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
    # result_dict[['Id','Predicted']].to_csv(os.path.join(args.save_path, 'submit.csv'), index=False)


    # del result_dict
    # del similarity_index
    
        # # build helper and evaluate results
        # evaluate_helper = build_evaluate_helper(cfg.evaluate)
        # mAP, recall_at_k = evaluate_helper.do_eval(index_result_info, gallery_names)

        # # show results
        # evaluate_helper.show_results(mAP, recall_at_k)


if __name__ == '__main__':
    main()

