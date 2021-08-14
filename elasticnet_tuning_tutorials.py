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
from seaborn_analyzer import regplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# 乱数シード
seed = 42
# モデル作成
model_linear = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
# クロスバリデーションして予測値ヒートマップを可視化
cv = KFold(n_splits=3, shuffle=True, random_state=seed)  # KFoldでクロスバリデーション分割指定
regplot.regression_heat_plot(model_linear, USE_EXPLANATORY, OBJECTIVE_VARIALBLE, df_osaka,
                             pair_sigmarange = 0.5, rounddigit_x1=3, rounddigit_x2=3,
                             cv=cv, display_cv_indices=0)

# %% 手順1) チューニング前の評価指標と回帰式算出（線形回帰）
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
# 回帰式を表示
X_train, y_train = [(X[train], y[train]) for train, test in cv.split(X, y)][0]
trained_model = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
trained_model.fit(X_train, y_train)
coef = trained_model['lr'].coef_
intercept = trained_model['lr'].intercept_
print(f'y = {coef[0]}*x1 + {coef[1]}*x2 + {coef[2]}*x3 + {coef[3]}*x4 + {intercept}')

# %% チューニング前のモデル（ElasiticNet）
from seaborn_analyzer import regplot
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
# 乱数シード
seed = 42
# モデル作成
model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet())])  # チューニング前のモデル
# クロスバリデーションして決定境界を可視化
cv = KFold(n_splits=3, shuffle=True, random_state=seed)  # KFoldでクロスバリデーション分割指定
regplot.regression_heat_plot(model, USE_EXPLANATORY, OBJECTIVE_VARIALBLE, df_osaka,
                             pair_sigmarange = 0.5, rounddigit_x1=3, rounddigit_x2=3,
                             cv=cv, display_cv_indices=0)

# %% 手順1) チューニング前の評価指標と回帰式算出（ElasticNet alpha=1, l1_ratio=0.5）
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
# 回帰式を表示
X_train, y_train = [(X[train], y[train]) for train, test in cv.split(X, y)][0]
trained_model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet())])
trained_model.fit(X_train, y_train)
coef = trained_model['enet'].coef_
intercept = trained_model['enet'].intercept_
print(f'y = {coef[0]}*x1 + {coef[1]}*x2 + {coef[2]}*x3 + {coef[3]}*x4 + {intercept}')

# %% 手順2) パラメータ種類と範囲の選択
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
cv_params = {'enet__alpha': [0, 0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100],
             'enet__l1_ratio': [0, 0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9, 0.97, 0.99, 1]
             }
param_scales = {'enet__alpha': 'log',
                'enet__l1_ratio': 'linear'
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

# %% 手順3＆4) パラメータ選択＆クロスバリデーション（グリッドサーチ）
from sklearn.model_selection import GridSearchCV
start = time.time()
# 最終的なパラメータ範囲(14x13通り)
cv_params = {'enet__alpha': [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
             'enet__l1_ratio': [0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.9, 0.97, 1]
             }
# グリッドサーチのインスタンス作成
gridcv = GridSearchCV(model, cv_params, cv=cv,
                      scoring=scoring, n_jobs=-1)
# グリッドサーチ実行（学習実行）
gridcv.fit(X, y)
# 最適パラメータの表示と保持
best_params = gridcv.best_params_
best_score = gridcv.best_score_
print(f'最適パラメータ {best_params}\nスコア {best_score}')
print(f'所要時間{time.time() - start}秒')
# %% グリッド内の評価指標を可視化（ヒートマップ）
import seaborn as sns
import pandas as pd
# パラメータと評価指標をデータフレームに格納
param1_array = gridcv.cv_results_['param_enet__alpha'].data.astype(np.float64)  # パラメータgamma
param2_array = gridcv.cv_results_['param_enet__l1_ratio'].data.astype(np.float64)  # パラメータC
mean_scores = gridcv.cv_results_['mean_test_score']  # 評価指標
df_heat = pd.DataFrame(np.vstack([param1_array, param2_array, mean_scores]).T,
                       columns=['alpha', 'l1_ratio', 'test_score'])
# グリッドデータをピボット化
df_pivot = pd.pivot_table(data=df_heat, values='test_score', 
                          columns='alpha', index='l1_ratio', aggfunc=np.mean)
# 上下軸を反転（元々は上方向が小となっているため）
df_pivot = df_pivot.iloc[::-1]
# ヒートマップをプロット
hm = sns.heatmap(df_pivot, cmap='YlGn', cbar_kws={'label': 'score'})

# %% 手順3＆4) パラメータ選択＆クロスバリデーション（ランダムサーチ）
from sklearn.model_selection import RandomizedSearchCV
start = time.time()
# パラメータの密度をグリッドサーチのときより増やす
cv_params = {'enet__alpha': [0.0001, 0.0002, 0.0004, 0.0007, 0.001, 0.002, 0.004, 0.007, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4, 0.7, 1],
             'enet__l1_ratio': [0, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9, 0.95, 0.99, 1]
             }
# ランダムサーチのインスタンス作成
randcv = RandomizedSearchCV(model, cv_params, cv=cv,
                            scoring=scoring, random_state=seed,
                            n_iter=180, n_jobs=-1)
# ランダムサーチ実行（学習実行）
randcv.fit(X, y)
# 最適パラメータの表示と保持
best_params = randcv.best_params_
best_score = randcv.best_score_
print(f'最適パラメータ {best_params}\nスコア {best_score}')
print(f'所要時間{time.time() - start}秒')
# %% ランダムサーチの評価指標を可視化（散布図）
# パラメータと評価指標をndarrayに格納
param1_array = randcv.cv_results_['param_enet__alpha'].data.astype(np.float64)  # パラメータgamma
param2_array = randcv.cv_results_['param_enet__l1_ratio'].data.astype(np.float64)  # パラメータC
mean_scores = randcv.cv_results_['mean_test_score']  # 評価指標
# 散布図プロット
sc = plt.scatter(param1_array, param2_array, c=mean_scores,
            cmap='YlGn', edgecolors='lightgrey')
cbar = plt.colorbar(sc)  # カラーバー追加
cbar.set_label('score')  # カラーバーのタイトル
plt.xscale('log')  # 第1軸をlogスケールに
plt.yscale('linear')  # 第2軸をlinearスケールに
plt.xlim(np.amin(cv_params['enet__alpha']), np.amax(cv_params['enet__alpha']))  # X軸表示範囲をデータ最小値～最大値に
plt.ylim(np.amin(cv_params['enet__l1_ratio']), np.amax(cv_params['enet__l1_ratio']))  # Y軸表示範囲をデータ最小値～最大値に
plt.xlabel('alpha')  # X軸ラベル
plt.ylabel('l1_ratio')  # Y軸ラベル

# %% 手順3＆4 パラメータ選択＆クロスバリデーション（BayesianOptimizationでベイズ最適化）
from bayes_opt import BayesianOptimization
start = time.time()
# パラメータ範囲（Tupleで範囲選択）
bayes_params = {'enet__alpha': (0.0001, 1),
                'enet__l1_ratio': (0, 1)
                }
# 対数スケールパラメータを対数化
param_scales = {'enet__alpha': 'log',
                'enet__l1_ratio': 'linear',
                }
bayes_params_log = {k: (np.log10(v[0]), np.log10(v[1])) if param_scales[k] == 'log' else v for k, v in bayes_params.items()}

# ベイズ最適化時の評価指標算出メソッド(引数が多いので**kwargsで一括読込)
def bayes_evaluate(**kwargs):
    params = kwargs
    # 対数スケールパラメータは10のべき乗をとる
    params = {k: np.power(10, v) if param_scales[k] == 'log' else v for k, v in params.items()}
    # モデルにパラメータ適用
    model.set_params(**params)
    # cross_val_scoreでクロスバリデーション
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring=scoring, n_jobs=-1)
    val = scores.mean()
    return val

# ベイズ最適化を実行
bo = BayesianOptimization(bayes_evaluate, bayes_params_log, random_state=seed)
bo.maximize(init_points=5, n_iter=30, acq='ei')
# 最適パラメータとスコアを取得
best_params = bo.max['params']
best_score = bo.max['target']
# 対数スケールパラメータは10のべき乗をとる
best_params = {k: np.power(10, v) if param_scales[k] == 'log' else v for k, v in best_params.items()}
# 最適パラメータを表示
print(f'最適パラメータ {best_params}\nスコア {best_score}')
print(f'所要時間{time.time() - start}秒')

# %% BayesianOptimizationの評価指標を可視化（散布図）
# パラメータと評価指標をDataFrameに格納
df_history = pd.DataFrame(bo.space.params, columns=bo.space.keys)  # パラメータ
df_history['enet__alpha'] = df_history['enet__alpha'].map(lambda x: np.power(10, x))  # alphaをLogスケールから戻す
mean_scores = bo.space.target  # 評価指標
# 散布図プロット
import matplotlib.pyplot as plt
sc = plt.scatter(df_history['enet__alpha'].values, df_history['enet__l1_ratio'].values, c=mean_scores,
            cmap='YlGn', edgecolors='lightgrey')
cbar = plt.colorbar(sc)  # カラーバー追加
cbar.set_label('score')  # カラーバーのタイトル
plt.xscale('log')  # 第1軸をlogスケールに
plt.yscale('linear')  # 第2軸をlinearスケールに
plt.xlim(bayes_params['enet__alpha'][0], bayes_params['enet__alpha'][1])  # X軸表示範囲をデータ最小値～最大値に
plt.ylim(bayes_params['enet__l1_ratio'][0], bayes_params['enet__l1_ratio'][1])  # Y軸表示範囲をデータ最小値～最大値に
plt.xlabel('alpha')  # X軸ラベル
plt.ylabel('l1_ratio')  # Y軸ラベル

# %% 手順3＆4) パラメータ選択＆クロスバリデーション（optunaでベイズ最適化）
import optuna
start = time.time()
# ベイズ最適化時の評価指標算出メソッド
def bayes_objective(trial):
    params = {
        'enet__alpha': trial.suggest_float('enet__alpha', 0.0001, 1, log=True),
        'enet__l1_ratio': trial.suggest_float('enet__l1_ratio', 0, 1)
    }
    # モデルにパラメータ適用
    model.set_params(**params)
    # cross_val_scoreでクロスバリデーション
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring=scoring, n_jobs=-1)
    val = scores.mean()
    return val

# ベイズ最適化を実行
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(bayes_objective, n_trials=45)

# 最適パラメータの表示と保持
best_params = study.best_trial.params
best_score = study.best_trial.value
print(f'最適パラメータ {best_params}\nスコア {best_score}')
print(f'所要時間{time.time() - start}秒')

# %% Optunaの評価指標を可視化（散布図）
# パラメータと評価指標をndarrayに格納
param1_array = [trial.params['enet__alpha'] for trial in study.trials]  # パラメータgamma
param2_array = [trial.params['enet__l1_ratio'] for trial in study.trials]  # パラメータC
mean_scores = [trial.value for trial in study.trials]  # 評価指標
# 散布図プロット
sc = plt.scatter(param1_array, param2_array, c=mean_scores,
            cmap='YlGn', edgecolors='lightgrey')
cbar = plt.colorbar(sc)  # カラーバー追加
cbar.set_label('score')  # カラーバーのタイトル
plt.xscale('log')  # 第1軸をlogスケールに
plt.yscale('linear')  # 第2軸をlinearスケールに
plt.xlim(0.0001, 1)  # X軸表示範囲をデータ最小値～最大値に
plt.ylim(0, 1)  # Y軸表示範囲をデータ最小値～最大値に
plt.xlabel('alpha')  # X軸ラベル
plt.ylabel('l1_ratio')  # Y軸ラベル

# %% 学習曲線のプロット
from sklearn.model_selection import learning_curve
# 最適パラメータを学習器にセット
model.set_params(**best_params)

# 学習曲線の取得
train_sizes, train_scores, valid_scores = learning_curve(estimator=model,
                                                         X=X, y=y,
                                                         train_sizes=np.linspace(0.1, 1.0, 10),
                                                         cv=cv, scoring=scoring, n_jobs=-1)
# 学習データ指標の平均±標準偏差を計算
train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
train_center = train_mean
train_high = train_mean + train_std
train_low = train_mean - train_std
# 検証データ指標の平均±標準偏差を計算
valid_mean = np.mean(valid_scores, axis=1)
valid_std  = np.std(valid_scores, axis=1)
valid_center = valid_mean
valid_high = valid_mean + valid_std
valid_low = valid_mean - valid_std
# training_scoresをプロット
plt.plot(train_sizes, train_center, color='blue', marker='o', markersize=5, label='training score')
plt.fill_between(train_sizes, train_high, train_low, alpha=0.15, color='blue')
# validation_scoresをプロット
plt.plot(train_sizes, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
plt.fill_between(train_sizes, valid_high, valid_low, alpha=0.15, color='green')
# 最高スコアの表示
best_score = valid_center[len(valid_center) - 1]
plt.text(np.amax(train_sizes), valid_low[len(valid_low) - 1], f'best_score={best_score}',
                color='black', verticalalignment='top', horizontalalignment='right')
# 軸ラベルおよび凡例の指定
plt.xlabel('training examples')  # 学習サンプル数を横軸ラベルに
plt.ylabel(scoring)  # スコア名を縦軸ラベルに
plt.legend(loc='lower right')  # 凡例

# %% 検証曲線のプロット（横軸パラメータ以外は最適値に固定）
from sklearn.model_selection import validation_curve
# 検証曲線描画対象パラメータ
valid_curve_params = {'enet__alpha': [0, 0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 10, 100],
                      'enet__l1_ratio': [0, 0.00001, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9, 0.97, 0.99, 1]
                      }
param_scales = {'enet__alpha': 'log',
                'enet__l1_ratio': 'linear'
                }
# 最適パラメータを上記描画対象に追加
for k, v in valid_curve_params.items():
    if best_params[k] not in v:
        v.append(best_params[k])
        v.sort()
for i, (k, v) in enumerate(valid_curve_params.items()):
    # モデルに最適パラメータを適用
    model.set_params(**best_params)
    # 検証曲線を描画
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
    # 最適パラメータを縦線表示
    plt.axvline(x=best_params[k], color='gray')
    # スケールをparam_scalesに合わせて変更
    plt.xscale(param_scales[k])
    # 軸ラベルおよび凡例の指定
    plt.xlabel(k)  # パラメータ名を横軸ラベルに
    plt.ylabel(scoring)  # スコア名を縦軸ラベルに
    plt.legend(loc='lower right')  # 凡例
    # グラフを描画
    plt.show()
    
# %% チューニング後のモデルと回帰式可視化
regplot.regression_heat_plot(model, USE_EXPLANATORY, OBJECTIVE_VARIALBLE, df_osaka,
                             pair_sigmarange = 0.5, rounddigit_x1=3, rounddigit_x2=3,
                             cv=cv, display_cv_indices=0,
                             estimator_params=best_params)
# 回帰式を表示
X_train, y_train = [(X[train], y[train]) for train, test in cv.split(X, y)][0]
trained_model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet())])
trained_model.set_params(**best_params)
trained_model.fit(X_train, y_train)
coef = trained_model['enet'].coef_
intercept = trained_model['enet'].intercept_
print(f'y = {coef[0]}*x1 + {coef[1]}*x2 + {coef[2]}*x3 + {coef[3]}*x4 + {intercept}')

# %%
