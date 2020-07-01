# filter模块说明
## 算法层
### StableFilter

计算特征稳定性指标，根据稳定性进行特征选择
```
class StableFilter(BaseEstimator, TransformerMixin):
    """
    特征稳定性判断，基于稳定性筛选
    """

    def __init__(self, indice_name='psi', indice_thr=0.2):
        self.indice_name = indice_name
        self.indice_thr = indice_thr
        
```

### example

```
{'method': 'StableFilter',
 'params': {'indice_name': 'psi', 'indice_thr': 0.2}}

```

### SelectFromModelFilter

sklearn.feature_selection.SelectFromModel的封装

[具体使用链接](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)

**Note**: if `n_features_to_select` in params ,则取`top n features`

### example

```
exampe 01

{"method": "SelectFromModelFilter",
 "params":  {"estimator": {
    "method": "XGBClassifier",
    "params": {}},
     "threshold": "0.05*mean"}
 }
 
example 02 (top n)

{"method": "SelectFromModelFilter",
 "params": {"n_features_to_select": 30,
            "estimator": {"method": "XGBClassifier", "params": {}}
            }
 }

```

### SelectKBestFilter

sklearn.feature_selection.SelectKBest的封装

[具体使用链接](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)

### example

```
{"method": "SelectKBestFilter",
 "params": {'k': 20,
        'score_func':'f_classif'
        }
}

```

### RFEFilter
sklearn.feature_selection.RFE的封装

[具体使用链接](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)

```
{"method": "RFEFilter",
 "params": {"n_features_to_select": 30,
            "estimator": {"method": "XGBClassifier", "params": {}}
            }
 }
```


# trainer模块说明
## 算法层
目前通过接口层对应的服务层主函数提供服务，对接sklearn中的estimator。可以将sklearn中estimator视为encoder层级。换言之，sklearn体系下分类的
api同样适用于modelling
## example
```
xgboost
===========================================================
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
    "n_estimators": 300}
           }

lr 
===========================================================
"estimator":{
"method": "LogisticRegression",
"params": {
    "solver":"saga",
    "penalty":"l2",
    "C":0.1,
    "random_state":42,
    "max_iter":1000}
          }

...         
          
```