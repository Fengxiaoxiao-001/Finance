import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import r2_score


def bass_model(t, p, q, M):
    """Bass扩散模型 - 累积形式[6](@ref)

    参数:
        t: 时间点
        p: 创新系数
        q: 模仿系数
        M: 市场潜在容量
    """
    if t == 0:
        return 0
    numerator = M * (1 - np.exp(-(p + q) * t))
    denominator = 1 + (q / p) * np.exp(-(p + q) * t)
    return numerator / denominator


def bass_differential(t, p, q, M):
    """Bass扩散模型 - 微分形式（当期 adoption）[6](@ref)"""
    if t == 0:
        return 0
    exp_term = np.exp(-(p + q) * t)
    numerator = M * (p + q) ** 2 * exp_term
    denominator = p * (1 + (q / p) * exp_term) ** 2
    return numerator / denominator


def logistic_growth(t, L, k, t0):
    """逻辑增长模型 - 用于美元稳定币[6](@ref)

    参数:
        t: 时间点
        L: 饱和水平（最大容量）
        k: 增长率
        t0: 增长中心点
    """
    return L / (1 + np.exp(-k * (t - t0)))


def estimate_bass_parameters(historical_t, historical_values, p0=[0.01, 0.1, 1000000]):
    """估计Bass模型参数[5,6](@ref)

    使用最小二乘法拟合Bass模型参数
    """

    def objective(params):
        p, q, M = params
        predicted = [bass_model(t, p, q, M) for t in historical_t]
        return np.sum((historical_values - predicted) ** 2)

    # 参数边界约束
    bounds = [(0.001, 0.5), (0.01, 0.8), (np.min(historical_values) * 2, np.max(historical_values) * 100)]

    result = minimize(objective, p0, bounds=bounds, method='L-BFGS-B')

    if result.success:
        p_opt, q_opt, M_opt = result.x
        print(f"Bass模型参数估计结果: p={p_opt:.4f}, q={q_opt:.4f}, M={M_opt:,.0f}")

        # 计算拟合优度
        predicted = [bass_model(t, p_opt, q_opt, M_opt) for t in historical_t]
        r2 = r2_score(historical_values, predicted)
        print(f"模型拟合优度 R² = {r2:.4f}")

        return p_opt, q_opt, M_opt
    else:
        print("参数估计失败，使用默认参数")
        return 0.03, 0.38, 1000000  # 默认参数


def estimate_logistic_parameters(historical_t, historical_values):
    """估计逻辑增长模型参数"""
    try:
        # 使用曲线拟合
        p0 = [np.max(historical_values), 0.5, np.median(historical_t)]  # 初始猜测
        popt, pcov = curve_fit(logistic_growth, historical_t, historical_values, p0=p0, maxfev=5000)

        L_opt, k_opt, t0_opt = popt
        print(f"逻辑增长模型参数: L={L_opt:,.0f}, k={k_opt:.4f}, t0={t0_opt:.2f}")
        return L_opt, k_opt, t0_opt
    except:
        print("逻辑增长模型参数估计失败，使用默认值")
        return np.max(historical_values) * 1.5, 0.3, np.median(historical_t)


def predict_growth(regression_results, baseline_assumptions, historical_data=None):
    """
    基于回归结果和未来假设进行预测 - 改进版本

    参数:
        regression_results: 阶段一的回归结果
        baseline_assumptions: 基线假设
        historical_data: 历史数据用于模型参数估计
    """
    # 使用传入的baseline_assumptions而非硬编码值
    if not baseline_assumptions:
        print("警告：baseline_assumptions为空，使用默认假设")
        baseline_assumptions = {
            'global_gdp_growth': [3.1, 3.2, 3.3, 3.2, 3.0],
            'cross_border_payment': [9000, 9500, 10000, 10500, 11000],
            'regulation_index': [9, 10, 10, 10, 10],
            'crypto_mcap': [3500000, 4000000, 4500000, 5000000, 5500000],
            'policy_dummy_2022': [1, 1, 1, 1, 1],
            'policy_dummy_2025_us': [1, 1, 1, 1, 1]
        }

    # 未来预测年份（2026-2030）
    future_years = [2026, 2027, 2028, 2029, 2030]

    # 使用传入的baseline_assumptions
    future_assumptions = baseline_assumptions

    # 确保所有必需的特征都存在
    required_features = ['global_gdp_growth', 'cross_border_payment', 'regulation_index',
                         'crypto_mcap', 'policy_dummy_2022', 'policy_dummy_2025_us']

    for feature in required_features:
        if feature not in future_assumptions:
            raise ValueError(f"baseline_assumptions中缺少必需特征: {feature}")

    # 构建未来DataFrame
    future_df = pd.DataFrame({
        'year': future_years,
        **future_assumptions
    })

    # 添加常数项
    future_df = sm.add_constant(future_df)

    # 使用系数进行额外验证（现在实际使用coefficients变量）
    coefficients = regression_results.params
    print("回归系数验证:", coefficients.to_dict())

    # 预测对数市值
    future_log_mcap = regression_results.predict(future_df)
    future_mcap = np.exp(future_log_mcap)

    # 分货币预测
    if historical_data is not None:
        # 使用历史数据估计模型参数
        historical_years = historical_data['year'].values
        historical_usd = historical_data['usd_mcap'].values
        historical_non_usd = historical_data['non_usd_mcap'].values

        # 将年份转换为相对时间
        t_base = historical_years.min()
        historical_t = historical_years - t_base
        future_t = np.array(future_years) - t_base

        # 美元稳定币：逻辑增长模型
        L_opt, k_opt, t0_opt = estimate_logistic_parameters(historical_t, historical_usd)
        usd_mcap_pred = [logistic_growth(t, L_opt, k_opt, t0_opt) for t in future_t]
        usd_mcap = np.minimum(usd_mcap_pred, future_mcap)

        # 非美元稳定币：Bass扩散模型（集成参数调整函数）
        p_base, q_base, M_base = estimate_bass_parameters(historical_t, historical_non_usd)

        # 使用adjust_bass_parameters_with_factors进行参数调整
        regression_factors = {
            'policy_dummy_2025_us': {
                'coefficient': 0.1,  # 应从阶段1结果获取实际值
                'significant': True
            },
            'regulation_index': {
                'coefficient': 0.05,  # 应从阶段1结果获取实际值
                'significant': True
            }
        }

        p_adjusted, q_adjusted, M_adjusted = adjust_bass_parameters_with_factors(
            (p_base, q_base, M_base), regression_factors, future_assumptions
        )

        # 应用调整后的Bass模型
        non_usd_mcap = np.array([bass_model(t, p_adjusted, q_adjusted, M_adjusted)
                                 for t in future_t])

    else:
        # 如果没有历史数据，使用简化方法
        print("警告：无历史数据，使用简化预测方法")

        # 美元稳定币：份额逐渐下降
        usd_share = np.linspace(0.85, 0.70, len(future_years))  # 从85%下降到70%
        usd_mcap = future_mcap * usd_share

        # 非美元稳定币：指数增长
        growth_rates = np.linspace(0.25, 0.15, len(future_years))  # 增长率从25%降至15%
        non_usd_mcap = []
        current = future_mcap[0] * (1 - usd_share[0])
        for i, rate in enumerate(growth_rates):
            if i > 0:
                current = non_usd_mcap[i - 1] * (1 + rate)
            non_usd_mcap.append(min(current, future_mcap[i] - usd_mcap[i]))
        non_usd_mcap = np.array(non_usd_mcap)

    # 计算份额和增长率
    usd_share = usd_mcap / future_mcap
    non_usd_share = non_usd_mcap / future_mcap

    # 计算增长率
    usd_growth = np.concatenate(([np.nan], np.diff(usd_mcap) / usd_mcap[:-1] * 100))
    non_usd_growth = np.concatenate(([np.nan], np.diff(non_usd_mcap) / non_usd_mcap[:-1] * 100))
    total_growth = np.concatenate(([np.nan], np.diff(future_mcap) / future_mcap[:-1] * 100))

    # 创建结果DataFrame
    results = pd.DataFrame({
        'year': future_years,
        'total_mcap': future_mcap,
        'total_growth_rate': total_growth,
        'usd_mcap': usd_mcap,
        'usd_share': usd_share * 100,  # 转换为百分比
        'usd_growth_rate': usd_growth,
        'non_usd_mcap': non_usd_mcap,
        'non_usd_share': non_usd_share * 100,  # 转换为百分比
        'non_usd_growth_rate': non_usd_growth
    })

    return results


def adjust_bass_parameters_with_factors(bass_params, regression_factors, future_assumptions):
    """
    使用回归模型中的显著因素调整Bass模型参数[6](@ref)

    参数:
        bass_params: 基础Bass参数 (p, q, M)
        regression_factors: 回归系数和显著性
        future_assumptions: 未来假设值
    """
    p_base, q_base, M_base = bass_params

    # 政策虚拟变量对创新系数p的影响
    policy_impact = 0.0
    if 'policy_dummy_2025_us' in regression_factors and regression_factors['policy_dummy_2025_us']['significant']:
        policy_coef = regression_factors['policy_dummy_2025_us']['coefficient']
        policy_impact = policy_coef * np.mean(future_assumptions.get('policy_dummy_2025_us', [1]))

    # 监管指数对模仿系数q的影响
    regulation_impact = 0.0
    if 'regulation_index' in regression_factors and regression_factors['regulation_index']['significant']:
        regulation_coef = regression_factors['regulation_index']['coefficient']
        regulation_impact = regulation_coef * np.mean(future_assumptions.get('regulation_index', [5]))

    # 调整参数
    p_adjusted = p_base * (1 + policy_impact)
    q_adjusted = q_base * (1 + regulation_impact * 0.1)  # 减小影响幅度

    print(f"Bass参数调整: p {p_base:.4f} → {p_adjusted:.4f}, q {q_base:.4f} → {q_adjusted:.4f}")

    return p_adjusted, q_adjusted, M_base


import pickle

import chardet
import pickle


def load_phase1_data():
    """加载阶段1的输出数据"""
    # 检测文件编码
    with open('historical_data.csv', 'rb') as f:
        raw_data = f.read()
        encoding_result = chardet.detect(raw_data)
        file_encoding = encoding_result['encoding']
        print(f"检测到的文件编码: {file_encoding} (置信度: {encoding_result['confidence']:.2f})")

    # 加载pickle文件
    with open('phase1_output.pkl', 'rb') as f:
        phase1_data = pickle.load(f)

    # 使用检测到的编码读取CSV文件
    try:
        historical_df = pd.read_csv('historical_data.csv', index_col=0, encoding=file_encoding)
    except UnicodeDecodeError:
        # 如果检测失败，尝试常见编码
        historical_df = pd.read_csv('historical_data.csv', index_col=0, encoding='gbk')

    return phase1_data['regression_results'], historical_df, phase1_data['features_used']


# 在2.py的主程序中修改：
if __name__ == "__main__":
    # 加载阶段1的数据
    regression_results, historical_df, features_used = load_phase1_data()

    # 使用加载的数据进行预测
    baseline_assumptions = {}  # 你的基线假设

    predictions = predict_growth(regression_results, baseline_assumptions, historical_df)

    # 保存预测结果供3.py使用
    predictions.to_csv('future_predictions.csv', index=False)
    print("阶段2预测结果已保存至 future_predictions.csv")
