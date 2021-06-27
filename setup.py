from distutils.core import setup
long_description = '''
Imperio is a python sci-kit learn inspired package for feature engineering. It contains a some feature transformers to make your data more easy to learn from for Machine Learning Algorithms.\n
This version of imperio has the next methods of feature selection:\n
1) Box-Cox (BoxCoxTransformer).\n
2) Clusterize (ClusterizeTransformer).\n
3) Combinator (CombinatorTransformer).\n
4) Frequency Imputation Transformer (FrequencyImputationTransformer).\n
5) log Transformer (LogTransformer).\n
6) Smoothing (SmoothingTransformer).\n
7) Spatial-Sign Transformer (SpatialSignTransformer).\n
8) Target Imputation Transformer (TargetImputationTransformer).\n
9) Whitening (WhiteningTransformer).\n
10) Yeo-Johnson Transformer (YeoJohnsonTransformer).\n
11) ZCA (ZCATransformer).\n
All these methods work like normal sklearn transformers. They have fit, transform and fit_transform functions implemented.\n
Additionally every imperio transformer has an apply function which allows to apply an transformation on a pandas Data Frame.\n
How to use imperio\n
To use a transformer from imperio you should just import the transformer from imperio in the following framework:\n
```from imperio import <class name>```\n
class names are written above in parantheses.\n
Next create a object of this algorithm (I will use Box-Cox as an example).\n
```method = BoxCoxTransformer()```\n
Firstly you should fit the transformer, passing to it a feature matrix (X) and the target array (y). y argument is really used only by Target-Imputation\n
```mathod.fit(X, y)```\n
After you fit the model, you can use it for transforming new data, using the transform function. To transform function you should pass only the feature matrix (X).\n
```X_transformed = method.transform(X)```\n
Also you can fit and transform the data at the same time using the fit_transform function.\n
```X_transformed = method.fit_transform(X)```\n
Also you can apply a transformation directly on a pandas DataFrame, choosing the columns that you want to change.\n
```new_df = method.apply(df, 'target', ['col1', 'col2'])```\n
Some advices.\n
1) Use ```FrequencyImputationTransformer``` and ```TargetImputationTransformer``` for categorical features.\n
2) Use ```BoxCoxTransformer``` and ```YeoJohnsonTransformer``` for numerical features to normalize a feature distribution.\n
3) Use ```SpatialSignTransformer``` on normalized data to bring outlayers to normal features..\n
4) Use ```CombinatorTransformer``` to combine different transformers on categorical and numerical columns separately.
With love from Sigmoid.\n
We are open for feedback. Please send your impression to papaluta.vasile@isa.utm.md\n
'''
setup(
  name = 'imperio',
  packages = ['imperio'],
  version = '0.1.3',
  license='MIT',
  description = 'Imperio is a python sci-kit learn inspired package for feature engineering.',
  long_description=long_description,
  author = 'SigmoidAI - Păpăluță Vasile',
  author_email = 'vpapaluta06@gmail.com',
  url = 'https://github.com/user/reponame',
  download_url = 'https://github.com/ScienceKot/kydavra/archive/v1.0.tar.gz',    # I explain this later on
  keywords = ['ml', 'machine learning', 'feature engineering', 'python', 'data science'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'scikit-learn'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Framework :: Jupyter',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)