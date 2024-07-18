#%%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold

#%%
# 定义特征矩阵X和目标变量矩阵y
# 假设环境因子数据为 X，地下生物量数据为 y
data1 = pd.read_excel('BD1.xlsx')
#%%
X = data1.iloc[:, 2:]
y = data1.iloc[:, 1]
#%%
# 分割自变量和目标变量
ID = data1.iloc[:, 0]
X_total = data1.iloc[:, 2:]
y_total = data1.iloc[:, 1]
#%%
# 划分数据为训练集和测试集
X_train, X_test, y_train, y_test, ID_train, ID_test = train_test_split(X_total, y_total, ID, test_size=0.2, random_state=42)

# 将训练集和测试集及对应的ID转为DataFrame
train_data = pd.concat([ID_train, y_train, X_train], axis=1)
test_data = pd.concat([ID_test, y_test, X_test], axis=1)

# 输出训练集和测试集到CSV文件
train_data.to_excel('train_dataBD1.xlsx', index=False)
test_data.to_excel('test_dataBD1.xlsx', index=False)
#%%
#%%
# 从文件中加载训练集和测试集
train_data = pd.read_excel('selected J8 trainBD.xlsx')
test_data = pd.read_excel('selected_J8 testBD1.xlsx')
#%%
# 提取自变量和目标变量
X = train_data.iloc[:, 2:]
y = train_data.iloc[:, 1]
X_test = test_data.iloc[:, 2:]
y_test = test_data.iloc[:, 1]
#%%
X_train = train_data.iloc[:, 2:]
y_train = train_data.iloc[:, 1]

#%%
from sklearn.metrics import make_scorer

# 定义计算RMSE的函数
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 将RMSE函数转换为适合交叉验证的scorer对象
scorer_rmse = make_scorer(rmse, greater_is_better=False)
#%%
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

rf = RandomForestRegressor(n_estimators=130,
                           criterion="absolute_error",
                           max_depth=15,
                           min_samples_split=2,
                           min_samples_leaf=2,
                           bootstrap=True,
                           oob_score=True,
                           n_jobs=-1,
                           random_state=42)
kfoldr = KFold(n_splits=10,shuffle=True,random_state=123)
scoresrf = cross_val_score(rf, X, y, cv=kfoldr, scoring='r2', n_jobs=-1)
for i, score in enumerate(scoresrf):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy rf: {:.4f}".format(scoresrf.mean()))

#%%
rf.fit(X, y)
predictions_rf = rf.predict(X_test)
predictions_rf_af = rf.predict(X)
# 输出模型在测试集上的表现
test_score = rf.score(X_test, y_test)
print("Test Set Accuracy rf: {:.4f}".format(test_score))
print("GBR Model Test Score: ", np.sqrt(np.mean((predictions_rf - y_test) ** 2)))

#%%
from sklearn.linear_model import Lasso
lasso = Lasso(alpha= 41)
kfoldl = KFold(n_splits=10,shuffle=True,random_state=123)
scoresl = cross_val_score(lasso,  X, y, cv=KFold(n_splits=10,shuffle=True,random_state=123),
                          scoring='r2',n_jobs=-1)
for i, score in enumerate(scoresl):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy lasso: {:.4f}".format(scoresl.mean()))

#%%
lasso.fit(X, y)
predictions_lasso = lasso.predict(X_test)
predictions_lasso_af = lasso.predict(X)
# 输出模型在测试集上的表现
test_score_lasso = lasso.score(X_test, y_test)
print("Test Set Accuracy lasso: {:.4f}".format(test_score_lasso))
print("GBR Model Test Score: ", np.sqrt(np.mean((predictions_lasso - y_test) ** 2)))
#%%
from sklearn.ensemble import GradientBoostingRegressor
modelg = GradientBoostingRegressor(n_estimators=180,
                                   learning_rate=0.03,
                                   max_depth=15,
                                   max_features='log2',
                                   min_samples_leaf=2,
                                   min_samples_split=5,
                                   loss='squared_error',
                                   random_state=123)
scoresg = cross_val_score(modelg,   X, y, cv=KFold(n_splits=10,shuffle=True,random_state=123)
                          ,scoring='r2',n_jobs=-1)
for i, score in enumerate(scoresg):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy GBR: {:.4f}".format(scoresg.mean()))
#%%
modelg.fit(X, y)
predictions_gbr = modelg.predict(X_test)
predictions_gbr_af = modelg.predict(X)
# 输出模型在测试集上的表现
test_score_gbr = modelg.score(X_test, y_test)
print("Test Set Accuracy GBR: {:.4f}".format(test_score_gbr))

print("GBR Model Test Score: ", np.sqrt(np.mean((predictions_gbr - y_test) ** 2)))
#%%
from yellowbrick.regressor import PredictionError  # 预测误差
visualizer = PredictionError(modelg)

visualizer.fit(X, y)
visualizer.score(X_test, y_test)
visualizer.show(outpath="误差估计GBRBD1.SVG", dpi=600)
#%%
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(criterion= "squared_error",
                            splitter= "random",
                            max_depth=16,
                            min_samples_split=2,
                            min_samples_leaf=3,
                            min_weight_fraction_leaf=0.02,
                            max_features=0.45,
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.05,
                            ccp_alpha=0.5,
                            random_state=123)
scoresreg = cross_val_score(reg,  X, y, cv=KFold(n_splits=10,shuffle=True,random_state=123)
                            ,scoring='r2',n_jobs=-1)
for i, score in enumerate(scoresreg):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy dt: {:.4f}".format(scoresreg.mean()))

#%%
reg.fit(X, y)
predictions_reg = reg.predict(X_test)
predictions_reg_af = reg.predict(X)
# 输出模型在测试集上的表现
test_score_reg = reg.score(X_test, y_test)
print("Test Set Accuracy reg: {:.4f}".format(test_score_reg))
#%%
from sklearn.linear_model import LinearRegression
model = linear = LinearRegression(fit_intercept=True,
                                  copy_X=False,
                                  n_jobs=-1,
                                  positive=True,)
kfoldli = KFold(n_splits=10,shuffle=True,random_state=123)
scoresli = cross_val_score(model, X, y, cv=KFold(n_splits=10,shuffle=True,random_state=123)
                           ,scoring='r2',n_jobs=-1)
for i, score in enumerate(scoresli):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy linear: {:.4f}".format(scoresli.mean()))

#%%
model.fit(X, y)
predictions_linear = model.predict(X_test)
predictions_linear_af = model.predict(X)
# 输出模型在测试集上的表现
test_score_linear = model.score(X_test, y_test)
print("Test Set Accuracy reg: {:.4f}".format(test_score_linear))
print("GBR Model Test Score: ", np.sqrt(np.mean((predictions_linear - y_test) ** 2)))
#%%
from xgboost import XGBRegressor
xgb = XGBRegressor(booster="gbtree"
                   , eta=0.053
                   , max_depth=11
                   , gamma=0.5
                   , subsample=0.612
                   , colsample_bytree=0.6
                   , alpha=0.4015
                   , eval_metric='rmse'
                   , objective= 'reg:squarederror')
kfoldxgb = KFold(n_splits=10,shuffle=True,random_state=123)
scoresxgb = cross_val_score(xgb,  X, y, cv=KFold(n_splits=10,shuffle=True,random_state=123)
                            ,scoring='r2',n_jobs=-1)
for i, score in enumerate(scoresxgb):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy xgb: {:.4f}".format(scoresxgb.mean()))

#%%
xgb.fit(X, y)
predictions_xgb = xgb.predict(X_test)
predictions_xgb_af = xgb.predict(X)
# 输出模型在测试集上的表现
test_score_xgb = xgb.score(X_test, y_test)
print("Test Set Accuracy xgb: {:.4f}".format(test_score_xgb))
print("GBR Model Test Score: ", np.sqrt(np.mean((predictions_xgb - y_test) ** 2)))
#%%
from yellowbrick.regressor import PredictionError  # 预测误差
visualizer = PredictionError(xgb)

visualizer.fit(X, y)
visualizer.score(X_test, y_test)
visualizer.show(outpath="误差估计xgbBD1.SVG", dpi=600)
#%%
from sklearn.linear_model import Ridge
alpha = 5000.81 # 正则化强度参数
rid = Ridge(alpha=alpha)
kfoldrid = KFold(n_splits=10,shuffle=True,random_state=123)
scoresrid = cross_val_score(rid,  X, y, cv=KFold(n_splits=10,shuffle=True,random_state=123)
                            ,scoring='r2',n_jobs=-1)
for i, score in enumerate(scoresrid):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy ridge: {:.4f}".format(scoresrid.mean()))

#%%
rid.fit(X, y)
predictions_rid = rid.predict(X_test)
predictions_rid_af = rid.predict(X)
# 输出模型在测试集上的表现
test_score_rid = rid.score(X_test, y_test)
print("Test Set Accuracy rid: {:.4f}".format(test_score_rid))
print("GBR Model Test Score: ", np.sqrt(np.mean((predictions_rid - y_test) ** 2)))
#%%
from yellowbrick.regressor import PredictionError  # 预测误差
visualizer = PredictionError(rid)

visualizer.fit(X, y,color='blue')
visualizer.score(X_test, y_test)
visualizer.show(outpath="误差估计RIDBD1.SVG", dpi=600)
#%%
from sklearn.linear_model import ElasticNet
net = ElasticNet(alpha=40.42, l1_ratio=0.85)
kfoldnet = KFold(n_splits=10,shuffle=True,random_state=123)
scoresnet = cross_val_score(net,  X, y, cv=KFold(n_splits=10,shuffle=True,random_state=123)
                            ,scoring='r2',n_jobs=-1)
for i, score in enumerate(scoresnet):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy ElasticNet: {:.4f}".format(scoresnet.mean()))

#%%
net.fit(X, y)
predictions_net = net.predict(X_test)
predictions_net_af = net.predict(X)
# 输出模型在测试集上的表现
test_score_net = net.score(X_test, y_test)
print("Test Set Accuracy ElasticNet: {:.4f}".format(test_score_net))
print("NET Model Test Score: ", np.sqrt(np.mean((predictions_net - y_test) ** 2)))
#%%
from yellowbrick.regressor import PredictionError  # 预测误差
visualizer = PredictionError(net)

visualizer.fit(X, y)
visualizer.score(X_test, y_test)
visualizer.show(outpath="误差估计netBD1.SVG", dpi=600)
#%%
# 将数据放入列表中
data2 = [scoresrf,scoresxgb,scoresg,scoresreg,scoresl,scoresrid,scoresnet,scoresli]
df = pd.DataFrame(data2)
df_T = df.transpose()
# 添加 header
df_T.rename(columns={0:'RandomForest',1:'XGBoost',2:'GradientBoosting',3:'DecisionTree',4:'Lasso',5:'Ridge',6:'ElasticNet',7:'Linear'}, inplace=True)

# 打印输出数据表
print(df_T)

df_T.to_excel('Cross-validation O data testBD1 r2.xlsx', index=False)
#%%
#%%
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(font="TimesNewRoman")
plt.rcParams["font.sans-serif"] = ["TimesNewRomans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("white")
sns.set_palette("coolwarm")

# 绘制箱型图并保存返回的轴对象
ax = sns.boxplot(data=df_T,boxprops={'alpha': 0.6})

# 设置X轴坐标文字为纵向显示
ax.set_xticklabels(ax.get_xticklabels(), rotation=75)

# 添加散点图，将散点的颜色设置为与箱型图相同的颜色
sns.stripplot(data=df_T, palette="muted", alpha=0.75, size=4)
# 设置Y轴范围，包括负数
ax.set_ylim([-0.2, 0.9])

plt.title("10 Cross-validation Performance Comparison of origin data")
plt.xlabel("")
plt.ylabel("Model accuracy score")
plt.savefig("Cross-validation Performance Comparison origin data trainBD1.svg", dpi=600, format="svg")
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6), dpi=800)
sns.set(font="TimesNewRoman")
plt.rcParams["font.sans-serif"] = ["TimesNewRomans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("white")
sns.set_palette("coolwarm")
# 绘制箱型图并保存返回的轴对象
ax = sns.boxplot(data=df_T,boxprops={'alpha': 0.7},showfliers=False,palette="coolwarm",linewidth=1,width=0.4)

# 设置X轴坐标文字为纵向显示
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# 添加散点图，并设置散点的颜色、大小和形状
sns.stripplot(data=df_T, palette="muted", alpha=0.75, size=5,jitter=0)

# 设置Y轴范围，包括负数
ax.set_ylim([-0.1, 0.95])

# 在X轴的0位置添加一条水平线
ax.axhline(0, color='black', linestyle='--')

plt.title("10 Cross-validation Performance Comparison of Selected Data Train Set")
plt.xlabel("")
plt.ylabel("Model accuracy score")
# 调整X轴和Y轴的字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 调整子图之间的间距
plt.subplots_adjust(bottom=0.2)
plt.savefig("Cross-validation Performance Comparison Selected data trainBD1.svg", dpi=800, format="svg")

plt.show()
#%%
