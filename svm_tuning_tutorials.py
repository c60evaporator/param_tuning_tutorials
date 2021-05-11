# %% データの読込
import seaborn as sns
iris = sns.load_dataset("iris")
sns.scatterplot(x='petal_width', y='petal_length', data=iris, hue='species')  # 説明変数と目的変数のデータ点の散布図をプロット
# %% チューニング前のモデル
from custom_scatter_plot import classplot
from sklearn.svm import SVC
from sklearn.model_selection import KFold
model = SVC()  # チューニング前のモデル(パラメータ指定しない)
seed = 42  # 乱数シード
cv = KFold(n_splits=3, shuffle=True, random_state=seed)  # KFoldでクロスバリデーション分割指定
classplot.class_separator_plot(model, ['petal_width', 'petal_length'], 'species', iris,
                               cv=cv, display_cv_indices=[0, 1, 2])
# %% 1) チューニング前の評価指標算出
from sklearn.model_selection import cross_val_score
import numpy as np
X = iris[['petal_width', 'petal_length']].values  # 説明変数をndarray化
y = iris['species']  # 目的変数をndarray化
scoring = 'f1_micro'  # 評価指標をf1_microに指定
print(f'gamma = {1 /(X.shape[1] * X.var())}') # gammaを表示
# クロスバリデーションで評価指標算出
scores = cross_val_score(model, X, y, cv=cv,
                         scoring=scoring, n_jobs=-1)
print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')
# %% 2) パラメータ種類と範囲の選択
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
cv_params = {'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
             'C': [0.1, 0.3, 1, 3, 10]}
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
    plt.xscale('log')
    # 軸ラベルおよび凡例の指定
    plt.xlabel(k)  # パラメータ名を横軸ラベルに
    plt.ylabel(scoring)  # スコア名を縦軸ラベルに
    plt.legend(loc='lower right')  # 凡例
    # グラフを描画
    plt.show()
# %% 2) パラメータ種類と範囲の選択（範囲を広げる）
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
cv_params = {'gamma': [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100, 1000],
             'C': [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100, 1000]}
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
    plt.xscale('log')
    # 軸ラベルおよび凡例の指定
    plt.xlabel(k)  # パラメータ名を横軸ラベルに
    plt.ylabel(scoring)  # スコア名を縦軸ラベルに
    plt.legend(loc='lower right')  # 凡例
    # グラフを描画
    plt.show()

# %% 3 & 4) パラメータ選択＆クロスバリデーション（グリッドサーチ）
from sklearn.model_selection import GridSearchCV
# 最終的なパラメータ範囲
cv_params = {'gamma': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
             'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}
# グリッドサーチのインスタンス作成
gridcv = GridSearchCV(model, cv_params, cv=cv,
                      scoring=scoring, n_jobs=-1)
# グリッドサーチ実行（学習実行）
gridcv.fit(X, y)
# 最適パラメータの表示と保持
best_params = gridcv.best_params_
best_score = gridcv.best_score_
print(f'最適パラメータ {best_params}\nスコア {best_score}')

# %% グリッド内の評価指標を可視化（ヒートマップ）
import pandas as pd
param1_array = gridcv.cv_results_['param_gamma'].data.astype(np.float64)
param2_array = gridcv.cv_results_['param_C'].data.astype(np.float64)
mean_scores = gridcv.cv_results_['mean_test_score']
df_heat = pd.DataFrame(np.vstack([param1_array, param2_array, mean_scores]).T,
                       columns=['gamma', 'C', 'test_score'])
# グリッドデータをピボット化
df_pivot = pd.pivot_table(data=df_heat, values='test_score', 
                          columns='gamma', index='C', aggfunc=np.mean)
# ヒートマップをプロット
sns.heatmap(df_pivot, cmap='YlGn')

# %% 3 & 4) パラメータ選択＆クロスバリデーション（グリッドサーチをスクラッチ実装）
import numpy as np
from sklearn.metrics import check_scoring
# パラメータ総当たり配列（グリッド）を作成
param_tuple = tuple(cv_params.values())
param_meshgrid = np.meshgrid(*param_tuple)
param_grid = np.vstack([param_array.ravel() for param_array in param_meshgrid]).T
# パラメータと評価指標格納用list
param_score_list = []
# グリッドを走査（スクラッチ実装）
for param_values in param_grid:
    # パラメータをdict型にしてモデルに格納
    params = {k: v for k, v in zip(cv_params.keys(), param_values)}
    model.set_params(**params)

    # クロスバリデーション（スクラッチ実装）
    scores = []  # 指標格納用リスト
    for train, test in cv.split(X, y):
        # 学習データとテストデータ分割
        X_train = X[train]  # 学習データ目的変数
        y_train = y[train]  # 学習データ説明変数
        X_test = X[test]  # テストデータ目的変数
        y_test = y[test]  # テストデータ説明変数
        # モデルの学習
        model.fit(X_train, y_train)
        # 指標算出
        scorer = check_scoring(model, scoring)
        score = scorer(model, X_test, y_test)
        scores.append(score)
    # 指標の平均値を算出
    mean_score = np.mean(scores)
    # パラメータと指標をlistに格納
    param_score_list.append({'score': mean_score,
                             'params': params
                             })

# 最適パラメータの表示と保持
max_index = np.argmax([a['score'] for a in param_score_list])
best_params = [a['params'] for a in param_score_list][max_index]
best_score = [a['score'] for a in param_score_list][max_index]
print(f'最適パラメータ {best_params}\nスコア {best_score}')

# %% 3 & 4) パラメータ選択＆クロスバリデーション（ランダムサーチ）
from sklearn.model_selection import RandomizedSearchCV
# パラメータの密度をグリッドサーチのときより増やす
cv_params = {'gamma': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
             'C': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]}
# ランダムサーチのインスタンス作成
randcv = RandomizedSearchCV(model, cv_params, cv=cv,
                            scoring=scoring, random_state=seed,
                            n_iter=50, n_jobs=-1)
# ランダムサーチ実行（学習実行）
randcv.fit(X, y)
# 最適パラメータの表示と保持
best_params = randcv.best_params_
best_score = randcv.best_score_
print(f'最適パラメータ {best_params}\nスコア {best_score}')

# %% ランダムサーチの評価指標を可視化（散布図）

# %% 3 & 4) パラメータ選択＆クロスバリデーション（BayesianOptimizationでベイズ最適化）
from bayes_opt import BayesianOptimization
# パラメータ範囲（Tupleで範囲選択）
bayes_params = {'gamma': (0.001, 100),
                'C': (0.01, 100)}
# ベイズ最適化時の評価指標算出メソッド
def bayes_evaluate(gamma, C):
    # 最適化対象のパラメータ
    params = {'gamma': gamma,
                'C': C}        
    # モデルにパラメータ適用
    model.set_params(**params)
    # cross_val_scoreでクロスバリデーション
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring=scoring, n_jobs=-1)
    val = scores.mean()
    return val

# ベイズ最適化を実行
bo = BayesianOptimization(bayes_evaluate, bayes_params, random_state=seed)
bo.maximize(init_points=5, n_iter=30, acq='ei')
# 最適パラメータの表示と保持
best_params = bo.max['params']
print(f'最適パラメータ {best_params}\nスコア {best_score}')

# %% 3 & 4) パラメータ選択＆クロスバリデーション（optunaでベイズ最適化）