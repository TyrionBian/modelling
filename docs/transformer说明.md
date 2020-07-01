transformer模块为特征工程模块，encoder为其基本单元，encoder的作用为进行特征变换，transformer支持多种特征变换方法（encoder）。

按照特征类型可以分为三类：连续型变量encoder，离散型变量encoder，通用型encoder。另外有特征衍生encoder暂时放入transformer中。

此文档用于规范encoder类的编写，同时列出常用的各个encoder的参数列表。

# 编写规范

1. 类取名应该清晰易懂， 符合一般规范（各单词首字母大写），以Encoder结尾， 例如`BaseEncoder`, `WOEEncoder`。对于只适用于cont的Encoder类应该以Cont作为类名开头，对于只适用于cate的Encoder类应该以Cate作为类名开头。

2. 类必须包含`__init__`、`fit`和`transformer`方法，transformer模块也只会调用这三个方法

3. fit方法必须形如：

   ```Python
   # 形式1
   def fit(self, df, y=None):
       ···
   
   # 形式2
   def fit(self, df, y):
       ···
   
   # 不需要return
   ```

   

4. transformer方法必须形如：

   ```python
   def transform(self, df):
       # 一系列处理过程
       df_final = df.copy()
       ...
       df_final.columns = df.columns
       return df.final
   
   # return必须是pandas.DataFrame,不能乱序，顺序必须和传入的df一致(最好保持index和传入的df一致，目前会在外部再做一次index对齐处理防止index不一致,最后列名也需要复原出来，因为某些方法会删掉列名)
   ```

   

5. 异常处理和错误抛出规范：不要使用`try...except...`进行异常跳出，目前只需要正常让程序自行报错即可

6. 必须有类注释，形如：

   ```
   """
   用于剔除缺失值严重列，同值严重列，不同值严重cate列（字符串列如果取值太过于分散，则信息量过低）。
   
   适用于cont和cate，支持缺失值, 建议放置在encoder序列第一位次
   
   Parameters
   ----------
   missing_thr: 0.8, 缺失率高于该值的列会被剔除
   
   same_thr: 0.8, 同值率高于该值的列会被剔除
   
   cate_thr: 0.9， 取值分散率高于该值的字符串列会被剔除
   
   Attributes
   ----------
   missing_cols: list, 被剔除的缺失值列
   
   same_cols: list, 被剔除的同值列
   
   cate_cols: list, 被剔除的取值分散字符串列
   
   exclude_cols: list, 被剔除的列名
   """
   ```

7. 完成后必须进行单元测试，自测通过

# 通用型Encoder
不限制特征类型，均适用
## BaseEncoder

```
class BaseEncoder(BaseEstimator, TransformerMixin):
    """
    用于剔除缺失值严重列，同值严重列，不同值严重cate列（字符串列如果取值太过于分散，则信息量过低）。

    适用于cont和cate，支持缺失值, 建议放置在encoder序列第一位次

    Parameters
    ----------
    missing_thr: 0.8, 缺失率高于该值的列会被剔除

    same_thr: 0.8, 同值率高于该值的列会被剔除

    cate_thr: 0.9， 取值分散率高于该值的字符串列会被剔除

    Attributes
    ----------
    missing_cols: list, 被剔除的缺失值列

    same_cols: list, 被剔除的同值列

    cate_cols: list, 被剔除的取值分散字符串列

    exclude_cols: list, 被剔除的列名
    """
```
## example

```
    {
        "method": "BaseEncoder",
        "params": {
            "cate_thr": 0.5,
            "missing_thr": 0.8,
            "same_thr": 0.9
        }
    }

```


## BinningEncoder

特征分箱，如果是离散值则是归并处理


```
class BinningEncoder(BaseEstimator, TransformerMixin):
    """
    特征分箱，如果是离散值则是归并处理
   
    支持缺失值

    Parameters
    ----------
    diff_thr : int, default: 20
        不同取值数高于该值才进行离散化处理，不然原样返回

    binning_method : str, default: 'dt', {'dt', 'qcut', 'cut'}
        分箱方法, 'dt' which uses decision tree, 'cut' which cuts data by the equal intervals,
        'qcut' which cuts data by the equal quantity. default is 'dt'. if y is None, default auto changes to 'qcut'.

    bins : int, default: 10
        分箱数目， 当binning_method='dt'时，该参数失效

    **kwargs :
        决策树分箱方法使用的决策树参数

    """

```


## example
```
{'method': 'BinningEncoder',
 'params': {'diff_thr':20,
            'bins':10,
            'binning_method':'dt'}
}
```

## BinningWOEEncoder

特征先分箱在做woe变换

```
class BinningWOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, bin_diff_thr=20, bins=10, binning_method='dt', bin_cate_f=True, inplace=True, bin_suffix='_bin',
                 woe_diff_thr=20, woe_min=-20, woe_max=20, woe_nan_thr=0.01, woe_suffix='_woe', woe_limit=True,
                 **kwargs):

```


## example

```
{'method': 'BinningWOEEncoder',
 'params': {'woe_diff_thr': 20,
            'woe_nan_thr': 0.01,
            'woe_max': 20,
            'woe_min': -20}
 }
```

## ImputeEncoder

缺失值填充为统一的数字
```
class ImputeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, fillna_value=-999):
        self.fillna_value = fillna_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        return df.replace(['nan', np.nan], self.fillna_value)

```

# 连续型变量encoder
## ContImputerEncoder
**缺失值填充**
```
class ContImputerEncoder(Imputer):
    """
    此类继承自sklearn.preprocessing.Imputer，fit与基类完全一致，transform方法返回变为pandas.DataFrame。

    仅适用于cont， 支持缺失值

    Parameters
    ----------
    missing_values : integer or "NaN", optional (default="NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as np.nan,
        use the string value "NaN".

    strategy : string, optional (default="mean")
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          the axis.
        - If "median", then replace missing values using the median along
          the axis.
        - If "most_frequent", then replace missing using the most frequent
          value along the axis.

    axis : integer, optional (default=0)
        The axis along which to impute.

        - If `axis=0`, then impute along columns.
        - If `axis=1`, then impute along rows.

    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.

    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If X is not an array of floating values;
        - If X is sparse and `missing_values=0`;
        - If `axis=0` and X is encoded as a CSR matrix;
        - If `axis=1` and X is encoded as a CSC matrix.

    Attributes
    ----------
    enc : 实例化的Imputer对象

    enc.statistics_ : array of shape (n_features,)
        The imputation fill value for each feature if axis == 0.

    Notes
    -----
    - When ``axis=0``, columns which only contained missing values at `fit`
      are discarded upon `transform`.
    - When ``axis=1``, an exception is raised if there are rows for which it is
      not possible to fill in the missing values (e.g., because they only
      contain missing values).
    """
```

## example
```
{'method': 'ContImputerEncoder',
           'params': {'axis': 0,
                    'missing_values': 'NaN',
                    'strategy': 'mean',
                    'verbose': 0}}
```

## ContBinningEncoder

**连续型变量分箱**
```
class ContBinningEncoder(BaseEstimator, TransformerMixin):
    """"
    将连续型变量转化为离散型

    仅适用于cont， 支持缺失值

    Parameters
    ----------
    diff_thr : int, default: 20
        不同取值数高于该值才进行离散化处理，不然原样返回

    binning_method : str, default: 'dt', {'dt', 'qcut', 'cut'}
        分箱方法, 'dt' which uses decision tree, 'cut' which cuts data by the equal intervals,
        'qcut' which cuts data by the equal quantity. default is 'dt'. if y is None, default auto changes to 'qcut'.

    bins : int, default: 10
        分箱数目， 当binning_method='dt'时，该参数失效

    **kwargs :
        决策树分箱方法使用的决策树参数

    Attributes
    ----------
    map: dict 每个特征每个取值所属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'cut_points': [1,2,3,4,5], 'labels': [1.0,2.0,3.0,4.0]}}
    kmap: dict 每个特征每一类取值组成的list及其归属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'(1, 2]': 1, '(2,3]': 2, '(3,4]': 3, '(4,5]': 4}}
    
    """"

```
## example

```
{'method': 'ContBinningEncoder',
         'params': {'binning_method': 'dt', 
         'bins': 10, 
         'diff_thr': 20}}
```



# 离散型变量encoder

## CountEncoder

将离散型变量转成成对应词频

```
class CountEncoder(BaseEstimator, TransformerMixin):
    """
    Parameters
    param unseen_value: 在训练集中没有出现的值给予unseen_value的出现频次，然后参与smoothing
    param log_transform: 是否取log
    param smoothing: 光滑处理，在出现频次上+smoothing
    param inplace: 是否删除原始字段

    Attributes
    ----------
    map: a collections.Counter(which like dict) map variable's values to its frequency.
    
    """
```
## example
```
{'method': 'CountEncoder',
           'params': {'log_transform': True,
                    'smoothing': 1,
                    'unseen_value': 1}}
```


## CateLabelEncoder

将离散型变量按照id编码

```
class CateLabelEncoder(BaseEstimator, TransformerMixin):
    """
    sklearn.preprocess.LabelEncoder can't process values which don't appear in fit label encoder.
    this method can process this problem. Replace all unknown values to a certain value, and encode this
    value to 0.

    Attributes
    ----------
    like sklearn.preprocess.LabelEncoder

    """
```
## example

```
{'method': 'CateLabelEncoder',
           'params': {}}

```

## CateOneHotEncoder

对离散型变量做onehot变换

```
class CateOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. The features are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’) encoding scheme. This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse parameter)
    
    By default, the encoder derives the categories based on the unique values in each feature. Alternatively, you can also specify the categories manually.
    
    This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.
    
    Note: a one-hot encoding of y labels should use a LabelBinarizer instead.


    Attributes
    ----------
    like sklearn.preprocessing.OneHotEncoder

    """
```
## example
```
{'method': 'CateOneHotEncoder',
           'params': {}}
```

## WOEEncoder

对离散型变量做woe变换

```
class WOEEncoder(BaseEstimator, TransformerMixin):
    """
    woe变换

    适用于cont和cate，但对多取值cont无效，支持缺失值

    Parameters
    ----------
    diff_thr : int, default: 20
        不同取值数小于等于该值的才进行woe变换，不然原样返回

    woe_min : int, default: -20
        woe的截断最小值

    woe_max : int, default: 20
        woe的截断最大值

    nan_thr : float, default: 0.01
        对缺失值采用平滑方法计算woe值，nan_thr为平滑参数

    """
```
## example

```
{'method': 'WOEEncoder',
           'params': {'diff_thr': 20,
                    'nan_thr': 0.01,
                    'woe_max': 20,
                    'woe_min': -20}}
```

## CateBinningEncoder
对离散型变量按照woe值进行归并

```

class CateBinningEncoder(BaseEstimator, TransformerMixin):
    # TODO: 强依赖ylabel，应该支持不依赖y的归并方式
    """
    对离散值变量做归并

    仅适用于cate， 支持缺失值

    Parameters
    ----------
    diff_thr : int, default: 20
        不同取值数高于该值才进行离散化处理，不然原样返回

    binning_method : str, default: 'dt', {'dt', 'qcut', 'cut'}
        分箱方法, 'dt' which uses decision tree, 'cut' which cuts data by the equal intervals,
        'qcut' which cuts data by the equal quantity. default is 'dt'. if y is None, default auto changes to 'qcut'.

    bins : int, default: 10
        分箱数目， 当binning_method='dt'时，该参数失效

    **kwargs :
        决策树分箱方法使用的决策树参数

    Attributes
    ----------
    map: dict 每个特征每个取值所属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'a': 1, 'b': 1, 'c': 2}}
    kmap: dict 每个特征每一类取值组成的list及其归属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'[a, b]': 1, '[c]': 2}}

    """
```


## example

```
{'method': 'CateBinningEncoder',
           'params': {'diff_thr':20, 
                      'bins':10, 
                      'binning_method':'dt'}
    
}
```

# 特征衍生encoder
## ReduceGen

通过聚类或者降维方法进行特征维度规约，生成新特征
```
class ReduceGen(BaseEstimator, TransformerMixin):
    """
    支持cluster， decomposition模块相关方法，已测试方法有kmeans，pca
    """

    def __init__(self, method='KMeans', method_params=None, prefix=None):
        self.method = method
        if method_params is None:
            self.method_params = {'random_state': 7}
        else:
            self.method_params = method_params
            self.method_params['random_state'] = 7
        self.prefix = method if prefix is None else prefix
```
## example
```
{'method': 'ReduceGen',
         'params': {'method': 'KMeans', 
                    'method_params':{'n_clusters': 5}
                   }
}

```







