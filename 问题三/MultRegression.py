import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 变量列表（2015-2024年年度数据） - 时间序列版本
variables = {
    # 因变量（全球总量）
    'global_stablecoin_mcap': '全球稳定币总市值(百万美元) - 从CoinMarketCap获取',

    # 经济因素（全球指标）
    'global_gdp_growth': '全球GDP增长率(%) - 世界银行/IMF',
    'global_trade_volume': '全球贸易总额(十亿美元) - WTO',
    'cross_border_payment': '跨境支付规模(十亿美元) - BIS',
    'avg_inflation_rate': '主要经济体平均通货膨胀率(%) - IMF',
    'us_interest_rate': '美国基准利率(%) - 美联储',

    # 政策因素（全球性虚拟变量或指数）
    'regulation_index': '全球监管指数(0-10分) - 手动编码',
    'policy_dummy_2022': '2022年日本法案虚拟变量(0/1)',
    'policy_dummy_2025_us': '2025年美国GENIUS法案虚拟变量(0/1)',
    'policy_dummy_2025_hk': '2025年香港条例虚拟变量(0/1)',

    # 市场因素
    'crypto_mcap': '加密货币总市值(百万美元) - CoinMarketCap',
    'btc_volatility': '比特币价格年化波动率(%)',
    'defi_tvl': 'DeFi总锁仓价值(百万美元) - DeFiPulse'
}


def create_time_series_data():
    """
    创建时间序列数据集（2015-2024年）
    返回：包含年份和各个变量的DataFrame
    """
    years = list(range(2015, 2025))
    n_years = len(years)

    # 设置随机种子保证结果可重现
    np.random.seed(123)

    # 创建时间序列数据
    data = {
        'year': years,

        # 因变量：全球稳定币总市值（模拟指数增长，符合实际趋势）
        'global_stablecoin_mcap': [50,
                                   200,
                                   1500,
                                   2800,
                                   4500,
                                   20000,
                                   150000,
                                   130000,
                                   145000,
                                   160000
                                   ],  # 单位：百万美元

        # 经济因素（基于真实经济趋势模拟）
        'global_gdp_growth': [2.8,
                              2.5,
                              3.1,
                              2.9,
                              2.3,
                              -3.3,
                              5.9,
                              3.1,
                              2.7,
                              3.2
                              ],  # 单位：%
        'global_trade_volume': [14660.556924188973,
                                15508.201071087109,
                                16856.805924173255,
                                16347.215668324343,
                                17000.23124239919,
                                17560.60213996852,
                                19436.29837469606,
                                19982.281933375783,
                                19946.471393383883,
                                20193.093199587427
                                ],  # 单位： 十亿美元
        'cross_border_payment': [16450,
                                 15850,
                                 17350,
                                 19450,
                                 18800,
                                 15900,
                                 22300,
                                 24800,
                                 23500,
                                 25200
                                 ],  # 单位： 十亿美元
        'avg_inflation_rate': [2.8,
                               3.4,
                               3.2,
                               3.5,
                               3.1,
                               3.8,
                               4.7,
                               8.1,
                               6.3,
                               4.2
                               ],  # 单位： %
        'us_interest_rate': [0.13,
                             0.66,
                             1.39,
                             2.41,
                             2.38,
                             0.08,
                             0.09,
                             4.33,
                             5.38,
                             4.75
                             ],  # 模拟美联储利率变化 单位： %

        # 政策因素（基于真实事件时间点）
        'regulation_index': [4.2,
                             4.8,
                             5.3,
                             6.1,
                             6.5,
                             7,
                             7.8,
                             8.3,
                             8.9,
                             9.5,
                             ],  # 监管逐渐加强
        'policy_dummy_2022': [0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              1,
                              1,
                              1
                              ],  # 2022年日本法案
        'policy_dummy_2025_us': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 2025年美国法案
        'policy_dummy_2025_hk': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 2025年香港条例

        # 市场因素
        'crypto_mcap': [4500,
                        9200,
                        650000,
                        125000,
                        250000,
                        750000,
                        2500000,
                        850000,
                        1200000,
                        1800000
                        ],  # 单位：百万美元
        'btc_volatility': [78,
                           65,
                           120,
                           82,
                           68,
                           145,
                           95,
                           110,
                           70,
                           85
                           ],  # 单位 %
        'defi_tvl': [0,
                     0,
                     10,
                     50,
                     200,
                     15000,
                     150000,
                     80000,
                     120000,
                     180000
                     ]  # 单位： 百万美元
    }

    df = pd.DataFrame(data)
    df.set_index('year', inplace=True)
    return df


def check_multicollinearity(df, features):
    """
    检查多重共线性 - 计算VIF值
    """
    print("=" * 50)
    print("多重共线性检查 (VIF值)")
    print("=" * 50)

    # 准备特征矩阵
    X = df[features]
    X = sm.add_constant(X)  # 添加常数项

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif_data.round(2))
    print("\nVIF判断标准: VIF > 10 表示严重多重共线性")
    return vif_data


def run_time_series_regression(df):
    """
    执行时间序列多元回归分析
    模型: ln(global_stablecoin_mcap) = β₀ + β₁X₁ + β₂X₂ + ... + ε
    """
    # 对数转换处理指数增长（经济学常用方法）
    df['log_stablecoin_mcap'] = np.log(df['global_stablecoin_mcap'])

    # 选择自变量（根据经济意义和VIF结果调整）
    features = [
        'global_gdp_growth',
        'cross_border_payment',
        'regulation_index',
        'crypto_mcap',
        'policy_dummy_2022',
        'policy_dummy_2025_us'
    ]

    # 检查多重共线性
    vif_results = check_multicollinearity(df, features)

    # === 新增代码：基于VIF过滤特征 ===
    vif_df = vif_results[vif_results['feature'] != 'const']
    high_vif_features = vif_df[vif_df['VIF'] > 10]['feature'].tolist()
    if high_vif_features:
        print(f"移除高VIF特征: {high_vif_features}")
        features = [f for f in features if f not in high_vif_features]
    # ===============================

    # 准备回归数据
    X = df[features]
    X = sm.add_constant(X)  # 添加常数项
    y = df['log_stablecoin_mcap']

    # 执行OLS回归
    model = sm.OLS(y, X)
    results = model.fit()

    # 输出回归结果
    print("=" * 70)
    print("时间序列多元回归结果（全球稳定币总需求）")
    print("=" * 70)
    print(results.summary())

    return results, features


def visualize_regression_results(df, results, features, df_clean=None):
    """
    可视化回归分析结果 - 修正版
    支持处理清洗后的数据（当使用滞后变量时）
    """
    # 确定使用完整数据还是清洗后的数据
    if df_clean is not None:
        data_for_plot = df_clean
        y_pred = results.predict()
        print(f"使用清洗后数据: {len(data_for_plot)} 个观测值")
    else:
        data_for_plot = df
        # 确保预测值与数据匹配
        if hasattr(results, 'fittedvalues'):
            y_pred = results.fittedvalues
        else:
            y_pred = results.predict()
        print(f"使用完整数据: {len(data_for_plot)} 个观测值")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 实际值 vs 预测值（修正：确保数组大小一致）
    actual_values = data_for_plot['global_stablecoin_mcap']

    # 确保预测值和实际值长度一致
    if len(y_pred) != len(actual_values):
        print(f"警告: 预测值数量({len(y_pred)})与实际值数量({len(actual_values)})不匹配")
        # 使用索引对齐
        aligned_actual = actual_values.loc[data_for_plot.index[:len(y_pred)]]
        aligned_pred = y_pred
    else:
        aligned_actual = actual_values
        aligned_pred = y_pred

    axes[0, 0].scatter(np.exp(aligned_pred), aligned_actual, alpha=0.7)
    min_val = min(np.exp(aligned_pred).min(), aligned_actual.min())
    max_val = max(np.exp(aligned_pred).max(), aligned_actual.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val],
                    'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Actual Values')
    axes[0, 0].set_title('Actual vs Predicted Values')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 残差图
    residuals = results.resid
    axes[0, 1].scatter(aligned_pred, residuals, alpha=0.7)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted Values')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 系数重要性（标准化系数）
    if 'const' in results.params:
        coefficients = results.params.drop('const')
    else:
        coefficients = results.params

    # 标准化系数
    y_std = data_for_plot['log_stablecoin_mcap'].std()
    standardized_coefs = coefficients / y_std

    y_pos = np.arange(len(standardized_coefs))
    axes[0, 2].barh(y_pos, standardized_coefs.values)
    axes[0, 2].set_yticks(y_pos)
    axes[0, 2].set_yticklabels(standardized_coefs.index)
    axes[0, 2].set_xlabel('Standardized Coefficients')
    axes[0, 2].set_title('Variable Importance (Standardized Coefficients)')
    axes[0, 2].grid(True, alpha=0.3, axis='x')

    # 4. 时间序列趋势（使用完整数据）
    axes[1, 0].plot(df.index, df['global_stablecoin_mcap'], marker='o', linewidth=2)
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Stablecoin Market Cap (Million USD)')
    axes[1, 0].set_title('Global Stablecoin Market Cap Growth Trend')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='plain', axis='y')

    # 5. 预测 vs 实际（时间序列）- 使用对齐的数据
    axes[1, 1].plot(data_for_plot.index, aligned_actual,
                    label='Actual Values', marker='o')
    axes[1, 1].plot(data_for_plot.index, np.exp(aligned_pred),
                    label='Predicted Values', marker='s')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Market Cap (Million USD)')
    axes[1, 1].set_title('Time Series: Predicted vs Actual Values')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='plain', axis='y')

    # 6. 变量相关性热力图
    try:
        # 只选择数值型列计算相关性
        numeric_columns = ['global_stablecoin_mcap'] + [f for f in features
                                                        if
                                                        f in data_for_plot.select_dtypes(include=[np.number]).columns]
        corr_matrix = data_for_plot[numeric_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    ax=axes[1, 2], fmt=".2f")
        axes[1, 2].set_title('Variable Correlation Heatmap')
    except Exception as e:
        print(f"热力图生成失败: {e}")
        axes[1, 2].text(0.5, 0.5, 'Heatmap unavailable',
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Variable Correlation Heatmap (Unavailable)')

    plt.tight_layout()
    plt.show()


def time_series_decomposition(df):
    """
    时间序列分解：趋势、季节性、残差
    """
    # 由于是年度数据，季节性不明显，主要看趋势
    series = df['global_stablecoin_mcap']

    # 尝试分解（周期设为2年，虽然年度数据季节性弱）
    try:
        decomposition = seasonal_decompose(series, model='multiplicative', period=2)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=axes[0], title='观测值')
        decomposition.trend.plot(ax=axes[1], title='趋势')
        decomposition.seasonal.plot(ax=axes[2], title='季节性')
        decomposition.resid.plot(ax=axes[3], title='残差')
        plt.tight_layout()
        plt.show()
    except:
        print("时间序列分解可能由于数据量不足而失败，跳过此步骤")


def model_diagnostic_tests(results, df, y):
    """
    模型诊断检验
    """
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad

    print("=" * 50)
    print("模型诊断检验")
    print("=" * 50)

    # Durbin-Watson检验（自相关）
    dw_stat = durbin_watson(results.resid)
    print(f"Durbin-Watson统计量: {dw_stat:.4f}")
    print("判断标准: 接近2表示无自相关，<1或>3可能存在问题")

    # 正态性检验 - 使用正确的Jarque-Bera检验
    jb_stat, jb_pvalue, jb_skew, jb_kurtosis = jarque_bera(results.resid)
    print(f"Jarque-Bera正态性检验: 统计量={jb_stat:.4f}, p值={jb_pvalue:.4f}")
    print("p值>0.05表示残差符合正态分布")

    # 额外的正态性检验（Anderson-Darling）
    try:
        ad_result = normal_ad(results.resid)
        print(f"Anderson-Darling正态性检验: 统计量={ad_result[0]:.4f}, p值={ad_result[1]:.4f}")
    except:
        # 如果normal_ad返回值格式不同，使用通用方法
        ad_stat, ad_pvalue = normal_ad(results.resid)
        print(f"Anderson-Darling正态性检验: 统计量={ad_stat:.4f}, p值={ad_pvalue:.4f}")

    # 异方差性检验（Breusch-Pagan）
    try:
        bp_test = het_breuschpagan(results.resid, results.model.exog)
        print(f"Breusch-Pagan异方差检验: LM统计量={bp_test[0]:.4f}, p值={bp_test[1]:.4f}")
        print("p值>0.05表示无异方差问题")
    except Exception as e:
        print(f"异方差检验失败: {e}")

    # 添加模型拟合优度评估
    print(f"R-squared: {results.rsquared:.4f}")
    print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
    print(f"AIC: {results.aic:.4f}")
    print(f"BIC: {results.bic:.4f}")


def improved_regression_analysis(df):
    """
    改进的回归分析 - 解决多重共线性和小样本问题
    """
    # 创建数据副本避免修改原始数据
    df_analysis = df.copy()

    # 对数转换（经济学常用方法）
    df_analysis['log_stablecoin_mcap'] = np.log(df_analysis['global_stablecoin_mcap'])
    df_analysis['log_crypto_mcap'] = np.log(df_analysis['crypto_mcap'])
    df_analysis['log_defi_tvl'] = np.log(df_analysis['defi_tvl'] + 1)  # 避免log(0)

    # 选择更合适的特征组合，避免多重共线性
    features_v2 = [
        'us_interest_rate',  # 利率因素（通常与稳定币需求负相关）
        'log_crypto_mcap',  # 加密货币总市值（对数形式减少量级差异）
        'regulation_index',  # 监管指数
        'policy_dummy_2022'  # 政策虚拟变量
    ]

    # 由于样本量小（n=10），使用更少的变量
    if len(df_analysis) < 15:
        # 基于经济意义选择最重要的2个变量
        features_v2 = ['us_interest_rate', 'log_crypto_mcap']
        print(f"小样本警告: 仅使用{len(features_v2)}个变量以避免过拟合")

    # 检查相关性
    correlation_matrix = df_analysis[['log_stablecoin_mcap'] + features_v2].corr()
    print("变量相关性矩阵:")
    print(correlation_matrix.round(3))

    # 移除缺失值
    df_clean = df_analysis.dropna()

    if len(df_clean) < len(df_analysis):
        print(f"数据清洗: 从{len(df_analysis)}行减少到{len(df_clean)}行")

    # 执行回归
    X = df_clean[features_v2]
    X = sm.add_constant(X)
    y = df_clean['log_stablecoin_mcap']

    model = sm.OLS(y, X)
    results = model.fit()

    return results, features_v2, df_clean


import pickle

# 在主程序开始处添加字体设置
if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建时间序列数据
    print("正在生成时间序列数据...")
    df = create_time_series_data()

    # 显示数据基本信息
    print(f"数据时间范围: {df.index.min()} - {df.index.max()}")
    print(f"变量数量: {len(df.columns)}")
    print(f"观测值数量: {len(df)}")

    # 描述性统计
    print("\n描述性统计:")
    print(df.describe().round(2))


    # 修改时间序列分解函数，避免中文字符
    def time_series_decomposition_safe(df):
        """安全的时间序列分解"""
        series = df['global_stablecoin_mcap']
        try:
            # 使用加法模型，因为年度数据季节性弱
            decomposition = seasonal_decompose(series, model='additive', period=2)

            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            decomposition.observed.plot(ax=axes[0], title='Observed Values')
            decomposition.trend.plot(ax=axes[1], title='Trend Component')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
            decomposition.resid.plot(ax=axes[3], title='Residual Component')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"时间序列分解失败: {e}")
            # 绘制简单的趋势图作为替代
            plt.figure(figsize=(12, 6))
            plt.plot(series.index, series.values, marker='o', linewidth=2)
            plt.title('Stablecoin Market Cap Trend (2015-2024)')
            plt.xlabel('Year')
            plt.ylabel('Market Cap (Million USD)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


    print("\n进行时间序列分解...")
    time_series_decomposition_safe(df)

    # 使用改进的回归分析
    print("\n正在执行改进的时间序列回归分析...")
    results, features, df_clean = improved_regression_analysis(df)

    # 保存关键数据供2.py使用
    data_to_save = {
        'regression_results': results,  # 回归模型对象
        'historical_df': df_clean,  # 清洗后的历史数据
        'features_used': features  # 使用的特征列表
    }

    # 使用pickle保存复杂对象（如模型结果）
    with open('phase1_output.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    # 将历史数据同时保存为CSV（便于查看）
    df_clean.to_csv('historical_data.csv')
    print("阶段1数据已保存至 phase1_output.pkl 和 historical_data.csv")

    # 输出回归结果
    print("=" * 70)
    print("改进的时间序列多元回归结果")
    print("=" * 70)
    print(results.summary())

    # 模型诊断
    model_diagnostic_tests(results, df_clean, df_clean['log_stablecoin_mcap'])

    # 可视化结果（传递清洗后的数据）
    print("\n生成可视化图表...")
    visualize_regression_results(df, results, features, df_clean)

    print("\n时间序列回归分析完成！")

    # 输出模型性能评估
    print("\n" + "=" * 50)
    print("模型性能总结")
    print("=" * 50)
    print(f"样本数量: {len(df_clean)}")
    print(f"R²: {results.rsquared:.4f}")
    print(f"调整R²: {results.rsquared_adj:.4f}")
    print(f"使用的变量: {features}")

    # 经济解释
    print("\n经济解释:")
    for feature, coef in results.params.items():
        if feature != 'const':
            effect = "正相关" if coef > 0 else "负相关"
            print(f"{feature}: 系数 = {coef:.4f} ({effect})")
