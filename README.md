# imperio

Imperio is a python sci-kit learn inspired package for feature engineering. It contains a some feature transformers to make your data more easy to learn from for Machine Learning Algorithms.

This version of imperio has the next methods of feature selection:
1. Box-Cox (BoxCoxTransformer).
2. Clusterize (ClusterizeTransformer).
3. Combinator (CombinatorTransformer).
4. Frequency Imputation Transformer (FrequencyImputationTransformer).
5. log Transformer (LogTransformer).
6. Smoothing (SmoothingTransformer).
7. Spatial-Sign Transformer (SpatialSignTransformer).
8. Target Imputation Transformer (TargetImputationTransformer).
9. Whitening (WhiteningTransformer).
10. Yeo-Johnson Transformer (YeoJohnsonTransformer).
11. ZCA (ZCATransformer).

All these methods work like normal sklearn transformers. They have fit, transform and fit_transform functions implemented.

Additionally every imperio transformer has an apply function which allows to apply an transformation on a pandas Data Frame.

# How to use imperio

To use a transformer from imperio you should just import the transformer from imperio in the following framework:
```python
from imperio import BoxCoxTransformer
```

class names are written above in parantheses.

Next create a object of this algorithm (Box-Cox is used as an example).

`method = BoxCoxTransformer()`

Firstly you should fit the transformer, passing to it a feature matrix (X) and the target array (y).
NOTE: y argument is really used only by the Target-Imputation.

`method.fit(X, y)`

After you fit the model, you can use it for transforming new data, using the transform function. To transform function you should pass only the feature matrix (X).

`X_transformed = method.transform(X)`

Also you can fit and transform the data at the same time using the `fit_transform` function.

`X_transformed = method.fit_transform(X)`

Also you can apply a transformation directly on a pandas DataFrame, choosing the columns that you want to change.

`new_df = method.apply(df, 'target', ['col1', 'col2']`

Some advices:\
1. Use `FrequencyImputationTransformer` or `TargetImputationTransformer` for categorical features.
2. Use `BoxCoxTransformer` or `YeoJohnsonTransformer` for numerical features to normalize a feature distribution.
3. Use `SpatialSignTransformer` on normalized data to bring outliers to normal samples.
4. Use `CombinatorTransformer` on tombine different transformers on categorical and numerical columns separately.

With <3 from Sigmoid!

We are open for feedback. Please send your impressions to papaluta.vasile@isa.utm.md
