import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy import stats

# 数据
digital_twin_values = np.array([1.2122679841, 1.2154716254, 1.2181724616, 1.2374678802, 
                                1.2405801694, 1.2410197428, 1.2657081696, 1.2683463342, 
                                1.2706132452])
tscan_values = np.array([1.212641, 1.214701, 1.217159, 1.244498, 1.244014, 1.245495, 
                         1.276710, 1.279198, 1.282128])

# 计算皮尔逊相关系数和平均绝对误差
pearson_corr, _ = pearsonr(digital_twin_values, tscan_values)
differences = np.abs(digital_twin_values - tscan_values)
mae = np.mean(differences)

# 拟合线性回归模型
regressor = LinearRegression()
regressor.fit(digital_twin_values.reshape(-1, 1), tscan_values)

# 预测拟合值
predicted_values = regressor.predict(digital_twin_values.reshape(-1, 1))

# 计算残差
residuals = tscan_values - predicted_values

# 残差的标准误差
s_err = np.sqrt(np.sum(residuals**2) / (len(tscan_values) - 2))

# 获取预测区间 (95% 置信区间)
n = len(digital_twin_values)
t_val = stats.t.ppf(1 - 0.025, df=n-2)  # 95% 置信区间的 t 值
mean_x = np.mean(digital_twin_values)
confidence_interval = t_val * s_err * np.sqrt(1/n + (digital_twin_values - mean_x)**2 / np.sum((digital_twin_values - mean_x)**2))

# 计算上下限
upper_bound = predicted_values + confidence_interval
lower_bound = predicted_values - confidence_interval

# 创建可视化图表
plt.figure(figsize=(8, 6))

# 散点图，黑色圆形点
plt.scatter(digital_twin_values, tscan_values, color='black', label='数据点', s=100)

# 拟合线，红色直线
plt.plot(digital_twin_values, predicted_values, color='red', label='拟合线', linestyle='-', linewidth=2)

# 绘制置信区间范围（使用填充区域表示）
plt.fill_between(digital_twin_values, lower_bound, upper_bound, color='red', alpha=0.2, label='置信区间')

# 图中显示 R 和 MAE 信息
plt.text(min(digital_twin_values), max(tscan_values), f'R = {pearson_corr:.4f}\nMAE = {mae:.4f}', 
         fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.7))

# 中文标签和标题
# plt.title('数字孪生驱动的测量值与T-Scan测量值的相关性', fontsize=14)
plt.xlabel('数字孪生驱动的测量值', fontsize=12)
plt.ylabel('T-Scan测量值', fontsize=12)
plt.legend()

# 显示网格和图表
plt.grid(True)
plt.show()
