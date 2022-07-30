# Author:Zhang Yuan

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# ------------------------------------------------------------
# Jupyter Notebook 控制台显示必须加上：%matplotlib inline ，弹出窗显示必须加上：%matplotlib auto
# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

# %%

''' 注意 Pycharm 的 ipython中显示颜色失效。Jupyter Notebook 有效。'''

# Seeding random data from numpy
np.random.seed(24)

# Making the DatFrame
df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
df = pd.concat([df, pd.DataFrame(np.random.randn(10, 4), columns=list('BCDE'))], axis=1)

# DataFrame without any styling
print("Original DataFrame:\n")
print(df)
print("\nModified Stlying DataFrame:")
df.style.set_properties(**{'background-color': 'black',
                           'color': 'green'})


df.iloc[0, 3] = np.nan
df.iloc[2, 3] = np.nan
df.iloc[4, 2] = np.nan
df.iloc[7, 4] = np.nan

# Highlight the NaN values in DataFrame
print("\nModified Stlying DataFrame:")
df.style.highlight_null(null_color='red')

