# %% データの読込
import seaborn as sns
iris = sns.load_dataset("iris")
sns.scatterplot(x='petal_width', y='petal_length', data=iris, hue='species')  # 説明変数と目的変数のデータ点の散布図をプロット
# %% チューニング前のモデル
from custom_scatter_plot import classplot
from sklearn.svm import SVC
from sklearn.model_selection import KFold
model = SVC()  # チューニング前のモデル(パラメータ指定しない)
cv = KFold(n_splits=3, shuffle=True, random_state=42)  # KFoldでクロスバリデーション分割指定
classplot.class_separator_plot(model, ['petal_width', 'petal_length'], 'species', iris,
                               cv=cv, display_cv_indices=[0, 1, 2])
# %% 5-1) チューニング前の評価指標算出
from sklearn.model_selection import cross_val_score
import numpy as np
X = iris[['petal_width', 'petal_length']].values  # 説明変数をndarray化
y = iris['species']  # 目的変数をndarray化
scoring = 'f1_micro'  # 評価指標をf1_microに指定
# クロスバリデーションで評価指標算出
scores = cross_val_score(model, X, y, cv=cv,
                         scoring=scoring, n_jobs=-1)
print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')
# %%
