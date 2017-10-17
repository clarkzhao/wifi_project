from __future__ import division, print_function # Imports from __future__ since we're running Python 2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

if __name__ == '__main__':
    shop_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ccf_first_round_shop_info.csv')
    user_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'ccf_first_round_user_shop_behavior.csv')
    eval_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'evaluation_public.csv')

    shop_train = pd.read_csv(shop_path, delimiter=',')
    user_train = pd.read_csv(user_path, delimiter=',')
    evaluation_data = pd.read_csv(eval_path, delimiter=',')
    # landsat_test = pd.read_csv(test_path, delimiter = ',')
    print(shop_train.head())