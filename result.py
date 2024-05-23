import pandas as pd

df = pd.read_csv('/home/data1/changhao/iBioHash/Codes/pytorch-image-models-main/lrd_post_sim_submit/submit_post_based_sim_method21_thr1.csv')

df.loc[4000:10000, 'Predicted'] = float('nan')

df.to_csv('/home/data1/changhao/iBioHash/Codes/pytorch-image-models-main/lrd_post_sim_submit/submit_post_based_sim_method21_thr1_0_4000.csv', index=False)
