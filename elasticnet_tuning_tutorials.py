# %% データの読込
import pandas as pd
import time
df_osaka = pd.read_csv(f'./sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIALBLE = 'approval_rate'  # 目的変数
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # 説明変数
y = df_osaka[OBJECTIVE_VARIALBLE].values  # 目的変数をndarray化
X = df_osaka[USE_EXPLANATORY].values  # 説明変数をndarray化
# データを表示
df_osaka[USE_EXPLANATORY + [OBJECTIVE_VARIALBLE]]

# %% チューニング前のモデル（線形回帰）
from custom_scatter_plot import regplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
# 乱数シード
seed = 42
# モデル作成
model_linear = LinearRegression()  # チューニング前のモデル
# クロスバリデーションして決定境界を可視化
cv = KFold(n_splits=3, shuffle=True, random_state=seed)  # KFoldでクロスバリデーション分割指定
regplot.regression_heat_plot(model_linear, USE_EXPLANATORY, OBJECTIVE_VARIALBLE, df_osaka,
                             pair_sigmarange = 0.5, rounddigit_x1=3, rounddigit_x2=3,
                             cv=cv, display_cv_indices=0)

# %% 手順1) チューニング前の評価指標算出（線形回帰）
from sklearn.model_selection import cross_val_score
import numpy as np
X = df_osaka[USE_EXPLANATORY].values  
y = df_osaka[OBJECTIVE_VARIALBLE]  # 目的変数をndarray化
scoring = 'neg_mean_squared_error'  # 評価指標をRMSEに指定
# クロスバリデーションで評価指標算出
scores = cross_val_score(model_linear, X, y, cv=cv,
                         scoring=scoring, n_jobs=-1)
print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')

# %% チューニング前のモデル（ElasiticNet）
from custom_scatter_plot import regplot
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
# 乱数シード
seed = 42
# モデル作成
model = ElasticNet()  # チューニング前のモデル
# クロスバリデーションして決定境界を可視化
cv = KFold(n_splits=3, shuffle=True, random_state=seed)  # KFoldでクロスバリデーション分割指定
regplot.regression_heat_plot(model, USE_EXPLANATORY, OBJECTIVE_VARIALBLE, df_osaka,
                             pair_sigmarange = 0.5, rounddigit_x1=3, rounddigit_x2=3,
                             cv=cv, display_cv_indices=0)

# %% 手順1) チューニング前の評価指標算出（ElasticNet）
from sklearn.model_selection import cross_val_score
import numpy as np
X = df_osaka[USE_EXPLANATORY].values  
y = df_osaka[OBJECTIVE_VARIALBLE]  # 目的変数をndarray化
scoring = 'neg_mean_squared_error'  # 評価指標をRMSEに指定
# クロスバリデーションで評価指標算出
scores = cross_val_score(model, X, y, cv=cv,
                         scoring=scoring, n_jobs=-1)
print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')

# %% 手順2) パラメータ種類と範囲の選択
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
cv_params = {'alpha': [0, 0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 10, 100],
             'l1_ratio': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1]
             }
param_scales = {'alpha': 'log',
                'l1_ratio': 'log'
                }
# 検証曲線のプロット（パラメータ毎にプロット）
for i, (k, v) in enumerate(cv_params.items()):
    train_scores, valid_scores = validation_curve(estimator=model,
                                                  X=X, y=y,
                                                  param_name=k,
                                                  param_range=v,
                                                  cv=cv, scoring=scoring,
                                                  n_jobs=-1)
    # 学習データに対するスコアの平均±標準偏差を算出
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    train_center = train_mean
    train_high = train_mean + train_std
    train_low = train_mean - train_std
    # テストデータに対するスコアの平均±標準偏差を算出
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std  = np.std(valid_scores, axis=1)
    valid_center = valid_mean
    valid_high = valid_mean + valid_std
    valid_low = valid_mean - valid_std
    # training_scoresをプロット
    plt.plot(v, train_center, color='blue', marker='o', markersize=5, label='training score')
    plt.fill_between(v, train_high, train_low, alpha=0.15, color='blue')
    # validation_scoresをプロット
    plt.plot(v, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
    plt.fill_between(v, valid_high, valid_low, alpha=0.15, color='green')
    # スケールを'log'に（線形なパラメータは'linear'にするので注意）
    plt.xscale(param_scales[k])
    # 軸ラベルおよび凡例の指定
    plt.xlabel(k)  # パラメータ名を横軸ラベルに
    plt.ylabel(scoring)  # スコア名を縦軸ラベルに
    plt.legend(loc='lower right')  # 凡例
    # グラフを描画
    plt.show()
# %%
