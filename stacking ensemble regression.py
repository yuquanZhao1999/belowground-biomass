import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import  cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
#%%
# 从文件中加载训练集和测试集
train_data = pd.read_excel('selected J8 trainBD.xlsx')
test_data = pd.read_excel('selected_J8 testBD1.xlsx')

# 提取自变量和目标变量

X_test = test_data.iloc[:, 2:]
y_test = test_data.iloc[:, 1]
X_train = train_data.iloc[:, 2:]
y_train = train_data.iloc[:, 1]
X = train_data.iloc[:, 2:]
y = train_data.iloc[:, 1]
#%%
gbm = GradientBoostingRegressor(n_estimators=180,
                                learning_rate=0.04,
                                max_depth=15,
                                max_features='log2',
                                min_samples_leaf=2,
                                min_samples_split=4,
                                loss='squared_error',
                                random_state=123)
elasticnet = ElasticNet(alpha=0.55, l1_ratio=0.67)
xgb = XGBRegressor(booster="gbtree"
                   , eta=0.053
                   , max_depth=11
                   , gamma=0.5
                   , subsample=0.612
                   , colsample_bytree=0.6
                   , alpha=0.4015
                   , eval_metric='rmse'
                   , objective= 'reg:squarederror')
 # 正则化强度参数
rid = Ridge(alpha=50.81)
#%%

class StackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_learners, meta_learner=LinearRegression(), n_folds=10, random_state=2):
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.n_folds = n_folds
        self.random_state = random_state
        self.kf_ = None
        self.models_ = None
        self.meta_model_ = None

    def fit(self, X, y):
        # 初始化KFold交叉验证
        self.kf_ = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # 初始化用于存储每个模型在每个fold上的预测结果
        X_train_stack = np.zeros((X.shape[0], len(self.base_learners)))

        # 对每个基模型进行训练和预测
        for i, model in enumerate(self.base_learners):
            # 存储每个fold的预测结果
            fold_preds = np.zeros(X.shape[0])
            for j, (train_index, test_index) in enumerate(self.kf_.split(X)):
                tr_x = X.iloc[train_index]
                tr_y = y.iloc[train_index]
                model.fit(tr_x, tr_y)
                fold_preds[test_index] = model.predict(X.iloc[test_index])
            # 将预测结果赋值到对应的列
            X_train_stack[:, i] = fold_preds

        # 训练元模型
        self.meta_model_ = self.meta_learner.fit(X_train_stack, y)

        return self

    def predict(self, X):
        # 使用每个基模型对新数据进行预测
        X_test_stack = np.array([model.predict(X) for model in self.base_learners]).T

        # 使用元模型进行最终预测
        return self.meta_model_.predict(X_test_stack)

    def score(self, X, y):
        # 使用预测方法预测测试集
        pred = self.predict(X)
        # 计算R2分数
        return r2_score(y, pred)

# 假设 models 已经定义为包含不同基学习器的列表
base_learners = [rid, gbm, elasticnet, xgb]
meta_learner = LinearRegression(fit_intercept=True,copy_X=False)
# 创建StackingRegressor实例
stacked_model = StackingRegressor(base_learners=base_learners, meta_learner=meta_learner)

# 拟合模型
stacked_model.fit(X_train, y_train)

# 进行预测
predictions = stacked_model.predict(X_test)

# 计算性能指标
r2 = stacked_model.score(X_test, y_test)

r2train = stacked_model.score(X_train, y_train)
print(f"R2: {r2}")
#%%
stacked_pred = stacked_model.predict(X_test)
print("Stacked Model Test Score: ", np.sqrt(np.mean((stacked_pred - y_test) ** 2)))
#%%
stacked_cv_score = cross_val_score(stacked_model, X, y
                                   , cv=KFold(n_splits=10,shuffle=True,random_state=123)
                                   , scoring='neg_root_mean_squared_error')
# 输出所有 R 方值
print("Stacked Model Cross-Validated R^2 Scores: ", stacked_cv_score)

# 计算并输出平均 R 方值
print("Average R^2 Score: ", stacked_cv_score.mean())
#%%
#neg_root_mean_squared_error
#%%
from sklearn.metrics import r2_score

# 使用 stacking_model 对测试集进行预测
y_pred = stacked_model.predict(X_test)

# 计算 R 方值
test_r2_score = r2_score(y_test, y_pred)

# 输出测试集上的 R 方值
print("Test Set R^2 Score: ", test_r2_score)
#%%
from sklearn.metrics import mean_squared_error

# 使用 stacking_model 对测试集进行预测
y_pred = stacked_model.predict(X_test)

# 计算均方误差
test_mse = mean_squared_error(y_test, y_pred)

# 计算均方根误差 (RMSE)
test_rmse = test_mse ** 0.5

# 输出测试集上的 RMSE
print("Test Set RMSE: ", test_rmse)
#%%
from yellowbrick.regressor import PredictionError  # 预测误差
visualizer = PredictionError(stacked_model)
ax = visualizer.ax
ax.grid(False)
visualizer.fit(X_train, y_train,color='#87CEEB')
visualizer.score(X_test, y_test)
visualizer.show(outpath="误差估计stackingBD.svg", dpi=600)
#%%
# 使用训练好的Stacking模型计算基础模型的预测
base_predictions = stacked_model.transform(X_train)
#%%
import shap
# 假设 stacking_model 已经被训练，并且 base_predictions 是基础模型对训练集的预测
# 使用元模型的 predict 方法作为可调用对象传入 shap.Explainer
explainer = shap.Explainer(stacked_model.predict, X_train)

# 计算 SHAP 值
shap_values = explainer(X_train)  # X_train 是用于训练的基础模型的数据
#%%
shap.plots.waterfall(shap_values[22],max_display=10)  # 可视化第一个样本的 SHAP 值
#%%
shap.plots.bar(shap_values[17], show_data=True,show=False)

# 使用 SHAP 图对象的 savefig() 方法来保存图表
plt.savefig("shap_plot17.svg", dpi=800, bbox_inches="tight")
#%%
shap.plots.scatter(shap_values[:,"Total.AGB"], color=shap_values[:,"Soil.Organic.Matter"])
#%%
# 绘制瀑布图，限制显示的SHAP值数量
shap.plots.bar(shap_values[16], show_data=True,show=False)
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.sans-serif": ['Times New Roman'],  # 指定字体
    'axes.unicode_minus': False,  # 用来正常显示负号
})

plt.xticks( fontproperties='Times New Roman', size=12) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=12) #设置y坐标字体和大小
# 关闭网格线
plt.grid(False)
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
# 保存图表为文件，确保数字和符号显示完全
plt.savefig("shap_waterfall_plot16.svg", dpi=800,bbox_inches="tight" )

#%%
# 绘制瀑布图，限制显示的SHAP值数量
shap.plots.waterfall(shap_values[1],show=False)
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xticks( fontproperties='Times New Roman', size=12) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=12) #设置y坐标字体和大小
# 关闭网格线
plt.grid(False)
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
plt.savefig("shap_waterfall_plot1.svg", dpi=800,bbox_inches="tight" )

#%%
shap.plots.waterfall(shap_values[1], max_display=10)
plt.margins(x=0.5, y=0.5)
# 保存图表为文件，确保数字和符号显示完全
plt.savefig("shap_ waterfall_plot001.svg", dpi=800,bbox_inches="tight" )
plt.close()
#%%
shap.summary_plot(shap_values, plot_type='bar',show=False)
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xticks( fontproperties='Times New Roman', size=12) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=12) #设置y坐标字体和大小
plt.xlabel('Mean(average impact on model output magnitude)', fontsize=12)#设置x轴标签和大小
# 关闭网格线
plt.grid(False)
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
plt.savefig("stacking train importanceBD123.svg",dpi=800,bbox_inches="tight") #可以保存图片

#%%
shap.plots.bar(shap_values,max_display=20,)
#%%
shap.summary_plot(shap_values, X_train,max_display=10)
#%%
shap.plots.violin(shap_values,max_display=15)
#%%
shap.plots.heatmap(explainer(X),show=False,max_display=37)
  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xticks( fontproperties='Times New Roman', size=12) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=12) #设置y坐标字体和大小
# 关闭网格线
plt.grid(False)
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
plt.savefig("stacking train heatmapBD1.svg",dpi=800,bbox_inches="tight") #可以保存图片
#%%
force_plot_figure = shap.force_plot(shap_values[22], X.iloc[22])
plt.show(force_plot_figure)
plt.savefig("shap stacking trainforce.svg",dpi=800) #可以保存图片
#%%
shap.summary_plot(shap_values, X, feature_names=X.columns,show=False,max_display=37)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xticks( fontproperties='Times New Roman', size=12) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=12) #设置y坐标字体和大小
plt.xlabel('Mean(average impact on model output magnitude)', fontsize=12)#设置x轴标签和大小
# 关闭网格线
plt.grid(False)
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
plt.title('Stacking Ensemble')
plt.savefig("shap stacking trainBD123.svg",dpi=800) #可以保存图片
plt.close()
#%%
