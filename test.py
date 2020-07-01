from  modelling.interface import spliter_ui,transformer_ui,filter_ui,trainer_ui
import json
#from modelling import chaintest
label = 'mob3_30'
test_path = 'E:/modelling_test/data/test.csv' #数据源位置
dst = 'E:/modelling_test/demo_modelling/' #运行路径
time_col = 'apply_date'
index_col = 'flowid'
import warnings
warnings.filterwarnings('ignore')
import modelling

params =  {
    'spliter': { "ds": {
        "table": test_path,
        "train": None,
        "test": None,
        "label": {"name": label, "type": "number"}},
        "out": {"dst": dst + "spliter"},
        "st": {
        "method": "oot",
        "time_col": time_col,
        "index_col": index_col,
        "test_size": 0.382,
        "random_state": 7}},
    'transformer': {"ds":
         {"train": dst + "spliter/train/spliter_result.pkl",
         "test": dst + "spliter/test/spliter_result.pkl",
         "label": {"name": label, "type": "number"}},
          "out": {"dst": dst + "transformer"},
          "st": {"cate": [
            {
                "encoders": [
                    {
                        "method": "BaseEncoder",
                        "params": {
                            "cate_thr": 0.5,
                            "missing_thr": 0.8,
                            "same_thr": 0.9
                        }
                    },
                    #{'method': 'WOEEncoder',
                     #'params': {'diff_thr': 20,
                                #'nan_thr': 0.01,
                                #'woe_max': 20,
                                #'woe_min': -20}},
                    {
                        "method": "CountEncoder",
                        "params": {
                            "log_transform": True,
                            "unseen_value": 1,
                            "smoothing": 1}
                    }
                ],
                "cols": []
            }],
        "cont": [
            {
                "encoders": [
                    {
                        "method": "BaseEncoder",
                        "params": {
                            "cate_thr": 0.5,
                            "missing_thr": 0.8,
                            "same_thr": 0.9
                        }
                    }
                ],
                "cols": []
            }],
        "method": "auto",
        "params": {"thr": 5},
        "verbose": True}},
    'filter':{"ds": {
        "train": dst + "transformer/train/transformer_result.pkl",
        "test": dst + "transformer/test/transformer_result.pkl",
        "label": {"name": label, "type": "number"}},
         "st":  [{'method': 'StableFilter',
         'params': {'indice_name': 'psi', 'indice_thr': 0.2}},
           {"method": "SelectFromModelFilter",
             "params":  {"estimator": {
                "method": "XGBClassifier",
                "params": {}},
                 "threshold": "0.05*mean"}
             }],
        "out": {"dst": dst + "filter"}} ,
     'trainer': {"ds": {
        "train": dst + "filter/train/filter_result.pkl",
        "test": dst + "filter/test/filter_result.pkl",
        "label": {"name": label, "type": "number"}},
        "out": {"dst": dst + "trainer"},
        "st": {
        "test_size": 0,
        "oversample": False,
        "n_folds": 5,
        "random_state": 7,
        "estimator":{
        "method": "XGBClassifier",
            "params": {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "max_depth": 5,
            "reg_lambda": 10,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "scale_pos_weight": 1,
            "silent": 1,
            "learning_rate": 0.1,
            "seed": 0,
            "eval_metric": "auc",
            "early_stopping_rounds": 30,
            "n_estimators": 300}},
        "reweight": False,
        "verbose": True}}
     }
modelling.chaintest(app_f=False, optimizer_f=False, custom_params=params, sampler_f=False)













