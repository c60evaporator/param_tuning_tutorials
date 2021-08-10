# %% データの読込
import pandas as pd
import seaborn as sns
df_athelete = pd.read_csv(f'./sample_data/nba_nfl_2.csv')
sns.scatterplot(x='height', y='weight', data=df_athelete, hue='league')  # 説明変数と目的変数のデータ点の散布図をプロット

# %% チューニング前のモデル
from seaborn_analyzer import classplot
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# 乱数シード
seed = 42
# モデル作成（標準化＋SVMパイプライン、loglossを使うためprobability＆random_state指定が必要）
model = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, random_state=seed))])
# クロスバリデーションして決定境界を可視化
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)  # StratifiedKFoldでクロスバリデーション分割指定
classplot.class_separator_plot(model, ['height', 'weight'], 'league', df_athelete,
                               cv=cv, display_cv_indices=[0, 1, 2])
# %% 手順1) チューニング前の評価指標算出
from sklearn.model_selection import cross_val_score
import numpy as np
X = df_athelete[['height', 'weight']].values  # 説明変数をndarray化
y = df_athelete['league']  # 目的変数をndarray化
scoring = 'neg_log_loss'  # 評価指標をneg_log_lossに指定
# クロスバリデーションで評価指標算出
scores = cross_val_score(model, X, y, cv=cv,
                         scoring=scoring, n_jobs=-1)
print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')
# gammaのデフォルト値を表示
print(f'gamma = {1 /(X.shape[1] * StandardScaler().fit_transform(X).var())}')

# %% 手順2) パラメータ種類と範囲の選択（svm_tuning_tutorials.pyより広めに範囲をとる）
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
cv_params = {'svm__gamma': [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100, 1000],
             'svm__C': [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100, 1000]}
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

# %% 手順3＆4) パラメータ選択＆クロスバリデーション（グリッドサーチ）
from sklearn.model_selection import GridSearchCV
# 最終的なパラメータ範囲
cv_params = {'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
             'svm__C': [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}
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
# パラメータと評価指標をDataFrameに格納
param1_array = gridcv.cv_results_['param_svm__gamma'].data.astype(np.float64)  # パラメータgamma
param2_array = gridcv.cv_results_['param_svm__C'].data.astype(np.float64)  # パラメータC
mean_scores = gridcv.cv_results_['mean_test_score']  # 評価指標
df_heat = pd.DataFrame(np.vstack([param1_array, param2_array, mean_scores]).T,
                       columns=['svm__gamma', 'svm__C', 'test_score'])
# グリッドデータをピボット化
df_pivot = pd.pivot_table(data=df_heat, values='test_score', 
                          columns='svm__gamma', index='svm__C', aggfunc=np.mean)
# 上下軸を反転（元々は上方向が小となっているため）
df_pivot = df_pivot.iloc[::-1]
# ヒートマップをプロット
sns.heatmap(df_pivot, cmap='YlGn', cbar_kws={'label': 'score'})

# %% 手順3＆4) パラメータ選択＆クロスバリデーション（ランダムサーチ）
from sklearn.model_selection import RandomizedSearchCV
# パラメータの密度をグリッドサーチのときより増やす
cv_params = {'svm__gamma': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
             'svm__C': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]}
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
# パラメータと評価指標をndarrayに格納
param1_array = randcv.cv_results_['param_svm__gamma'].data.astype(np.float64)  # パラメータgamma
param2_array = randcv.cv_results_['param_svm__C'].data.astype(np.float64)  # パラメータC
mean_scores = randcv.cv_results_['mean_test_score']  # 評価指標
# 散布図プロット
sc = plt.scatter(param1_array, param2_array, c=mean_scores,
                       cmap='YlGn', edgecolors='lightgrey')
cbar = plt.colorbar(sc)  # カラーバー追加
cbar.set_label('score')  # カラーバーのタイトル
plt.xscale('log')  # 第1軸をlogスケールに
plt.yscale('log')  # 第2軸をlogスケールに
plt.xlim(np.amin(cv_params['svm__gamma']), np.amax(cv_params['svm__gamma']))  # X軸表示範囲をデータ最小値～最大値に
plt.ylim(np.amin(cv_params['svm__C']), np.amax(cv_params['svm__C']))  # Y軸表示範囲をデータ最小値～最大値に
plt.xlabel('svm__gamma')  # X軸ラベル
plt.ylabel('svm__C')  # Y軸ラベル

# %% 手順3＆4 パラメータ選択＆クロスバリデーション（BayesianOptimization対数軸でベイズ最適化）
from bayes_opt import BayesianOptimization
# パラメータ範囲（Tupleで範囲選択）
bayes_params = {'svm__gamma': (0.01, 100),
                'svm__C': (0.1, 1000)}
# パラメータ範囲を対数化
bayes_params_log = {k: (np.log10(v[0]), np.log10(v[1])) for k, v in bayes_params.items()}
# ベイズ最適化時の評価指標算出メソッド
def bayes_evaluate(svm__gamma, svm__C):
    # 最適化対象のパラメータ
    params = {'svm__gamma': np.power(10 ,svm__gamma),
              'svm__C': np.power(10, svm__C)}
    # モデルにパラメータ適用
    model.set_params(**params)
    # cross_val_scoreでクロスバリデーション
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring=scoring, n_jobs=-1)
    val = scores.mean()
    return val

# ベイズ最適化を実行
bo = BayesianOptimization(bayes_evaluate, bayes_params_log, random_state=seed)
bo.maximize(init_points=5, n_iter=40, acq='ei')
# 最適パラメータの表示と保持
best_params = {k: np.power(10, v) for k, v in bo.max['params'].items()}
best_score = bo.max['target']
print(f'最適パラメータ {best_params}\nスコア {best_score}')

# %% BayesianOptimization対数軸の評価指標を可視化（散布図）
# パラメータと評価指標をDataFrameに格納
df_history = pd.DataFrame(np.power(10 ,bo.space.params), columns=bo.space.keys)  # パラメータ
mean_scores = bo.space.target  # 評価指標
# 散布図プロット
sc = plt.scatter(df_history['svm__gamma'].values, df_history['svm__C'].values, c=mean_scores,
                 cmap='YlGn', edgecolors='lightgrey')
cbar = plt.colorbar(sc)  # カラーバー追加
cbar.set_label('score')  # カラーバーのタイトル
plt.xscale('log')  # 第1軸をlogスケールに
plt.yscale('log')  # 第2軸をlogスケールに
plt.xlim(bayes_params['svm__gamma'][0], bayes_params['svm__gamma'][1])  # X軸表示範囲をデータ最小値～最大値に
plt.ylim(bayes_params['svm__C'][0], bayes_params['svm__C'][1])  # Y軸表示範囲をデータ最小値～最大値に
plt.xlabel('svm__gamma')  # X軸ラベル
plt.ylabel('svm__C')  # Y軸ラベル

# %% 手順3＆4) パラメータ選択＆クロスバリデーション（optunaでベイズ最適化）
import optuna
# ベイズ最適化時の評価指標算出メソッド
def bayes_objective(trial):
    params = {
        'svm__gamma': trial.suggest_float('svm__gamma', 0.01, 100, log=True),
        'svm__C': trial.suggest_float('svm__C', 0.1, 1000, log=True)
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
study.optimize(bayes_objective, n_trials=40)

# 最適パラメータの表示と保持
best_params = study.best_trial.params
best_score = study.best_trial.value
print(f'最適パラメータ {best_params}\nスコア {best_score}')

# %% Optunaの評価指標を可視化（散布図）
# パラメータと評価指標をndarrayに格納
param1_array = [trial.params['svm__gamma'] for trial in study.trials]  # パラメータgamma
param2_array = [trial.params['svm__C'] for trial in study.trials]  # パラメータC
mean_scores = [trial.value for trial in study.trials]  # 評価指標
# 散布図プロット
sc = plt.scatter(param1_array, param2_array, c=mean_scores,
                 cmap='YlGn', edgecolors='lightgrey')
cbar = plt.colorbar(sc)  # カラーバー追加
cbar.set_label('score')  # カラーバーのタイトル
plt.xscale('log')  # 第1軸をlogスケールに
plt.yscale('log')  # 第2軸をlogスケールに
plt.xlim(0.01, 100)  # X軸表示範囲をデータ最小値～最大値に
plt.ylim(0.1, 1000)  # Y軸表示範囲をデータ最小値～最大値に
plt.xlabel('svm__gamma')  # X軸ラベル
plt.ylabel('svm__C')  # Y軸ラベル

# %% 学習曲線のプロット
from sklearn.model_selection import learning_curve
# 最適パラメータを学習器にセット
model.set_params(**best_params)
scores = cross_val_score(model, X, y, cv=cv,
                             scoring=scoring, n_jobs=-1)
print(scores.mean())
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
# 検証曲線描画対象パラメータ
valid_curve_params = {'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                      'svm__C': [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}
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
    best_index = np.where(np.array(v)==best_params[k])
    best_score = valid_center[best_index][0]
    plt.text(best_params[k], np.amax(valid_center), f'best_{k}={best_params[k]}\nbest_score={best_score}',
                    color='black', verticalalignment='bottom', horizontalalignment='left')
    # スケールを'log'に（線形なパラメータは'linear'にするので注意）
    plt.xscale('log')
    # 軸ラベルおよび凡例の指定
    plt.xlabel(k)  # パラメータ名を横軸ラベルに
    plt.ylabel(scoring)  # スコア名を縦軸ラベルに
    plt.legend(loc='lower right')  # 凡例
    # グラフを描画
    plt.show()

# %% チューニング後のモデル可視化
classplot.class_separator_plot(model, ['height', 'weight'], 'league', df_athelete,
                               cv=cv, display_cv_indices=[0, 1, 2],
                               clf_params=best_params)
# %%