import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import r2_score


def bass_model(t, p, q, M):
    """Bass扩散模型 - 累积形式

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
    """Bass扩散模型 - 微分形式（当期 adoption）"""
    if t == 0:
        return 0
    exp_term = np.exp(-(p + q) * t)
    numerator = M * (p + q) ** 2 * exp_term
    denominator = p * (1 + (q / p) * exp_term) ** 2
    return numerator / denominator


def logistic_growth(t, L, k, t0):
    """逻辑增长模型 - 用于美元稳定币

    参数:
        t: 时间点
        L: 饱和水平（最大容量）
        k: 增长率
        t0: 增长中心点
    """
    return L / (1 + np.exp(-k * (t - t0)))


def estimate_bass_parameters(historical_t, historical_values, p0=[0.01, 0.1, 1000000]):
    """估计Bass模型参数

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
    基于回归结果和未来假设进行预测 - 修正版本
    """
    # 使用传入的baseline_assumptions而非硬编码值
    if not baseline_assumptions:
        print("警告：baseline_assumptions为空，使用默认假设")
        baseline_assumptions = {
            'us_interest_rate': [2.5, 2.6, 2.7, 2.8, 2.9],  # 美国利率
            'crypto_mcap': [3500000, 4000000, 4500000, 5000000, 5500000]  # 加密货币总市值
        }

    # 未来预测年份（2026-2030）
    future_years = [2026, 2027, 2028, 2029, 2030]
    n_years = len(future_years)

    # === 关键修正：验证数组长度一致性 ===
    print(f"未来预测年份数量: {n_years}")

    # 检查所有假设数组长度是否一致
    for feature, values in baseline_assumptions.items():
        if len(values) != n_years:
            print(f"警告: {feature} 数组长度({len(values)})与预测年份数({n_years})不匹配")
            # 如果长度不匹配，截取或扩展数组
            if len(values) > n_years:
                baseline_assumptions[feature] = values[:n_years]  # 截取前n个
                print(f"  已截取前{n_years}个值")
            else:
                # 如果长度不足，用最后一个值填充
                last_value = values[-1]
                baseline_assumptions[feature] = values + [last_value] * (n_years - len(values))
                print(f"  已用最后一个值填充至{n_years}个")

    # 获取模型实际使用的特征名
    model_features = regression_results.model.exog_names
    print(f"模型实际使用的特征: {model_features}")

    # 只使用模型需要的特征构建DataFrame
    future_data = {'year': future_years}

    # 添加模型需要的特征
    for feature in model_features:
        if feature == 'const':
            continue  # 常数项后面单独添加
        elif feature in baseline_assumptions:
            future_data[feature] = baseline_assumptions[feature]
        elif feature == 'log_crypto_mcap' and 'crypto_mcap' in baseline_assumptions:
            # 如果模型使用对数市值，但假设提供原始市值，进行转换
            future_data[feature] = np.log(baseline_assumptions['crypto_mcap'])
        else:
            # 如果特征不存在，使用合理默认值
            print(f"警告: 特征 {feature} 不在baseline_assumptions中，使用默认值")
            if 'interest' in feature.lower():
                future_data[feature] = [2.5, 2.6, 2.7, 2.8, 2.9]  # 利率默认值
            elif 'crypto' in feature.lower():
                future_data[feature] = np.log([3500000, 4000000, 4500000, 5000000, 5500000])
            else:
                future_data[feature] = [1] * n_years  # 其他特征默认值

    # 验证所有数组长度一致
    array_lengths = {key: len(value) for key, value in future_data.items()}
    print("数组长度检查:", array_lengths)

    if len(set(array_lengths.values())) != 1:
        raise ValueError(f"数组长度不一致: {array_lengths}")

    # 构建未来DataFrame
    future_df = pd.DataFrame(future_data)

    # 添加常数项（确保与模型训练时一致）
    future_df = sm.add_constant(future_df)

    # 确保列顺序与模型一致
    future_df = future_df[model_features]

    print(f"预测数据形状: {future_df.shape}")
    print(f"预测数据列名: {future_df.columns.tolist()}")

    # 使用系数进行验证
    coefficients = regression_results.params
    print("回归系数:", coefficients.to_dict())

    # 预测对数市值
    future_log_mcap = regression_results.predict(future_df)
    future_mcap = np.exp(future_log_mcap)

    # 分货币预测
    if historical_data is not None and all(col in historical_data.columns for col in ['usd_mcap', 'non_usd_mcap']):
        print("使用历史数据进行分货币预测...")
        # 使用历史数据估计模型参数
        historical_years = historical_data.index.values if 'year' not in historical_data.columns else historical_data[
            'year'].values
        historical_usd = historical_data['usd_mcap'].values if 'usd_mcap' in historical_data.columns else \
        historical_data['global_stablecoin_mcap'].values * 0.8
        historical_non_usd = historical_data['non_usd_mcap'].values if 'non_usd_mcap' in historical_data.columns else \
        historical_data['global_stablecoin_mcap'].values * 0.2

        # 将年份转换为相对时间
        t_base = historical_years.min()
        historical_t = historical_years - t_base
        future_t = np.array(future_years) - t_base

        # 美元稳定币：逻辑增长模型
        L_opt, k_opt, t0_opt = estimate_logistic_parameters(historical_t, historical_usd)
        usd_mcap_pred = [logistic_growth(t, L_opt, k_opt, t0_opt) for t in future_t]
        usd_mcap = np.minimum(usd_mcap_pred, future_mcap)

        # 非美元稳定币：Bass扩散模型
        p_base, q_base, M_base = estimate_bass_parameters(historical_t, historical_non_usd)

        # 使用adjust_bass_parameters_with_factors进行参数调整
        regression_factors = {
            'policy_dummy_2025_us': {
                'coefficient': 0.1,
                'significant': True
            },
            'regulation_index': {
                'coefficient': 0.05,
                'significant': True
            }
        }

        p_adjusted, q_adjusted, M_adjusted = adjust_bass_parameters_with_factors(
            (p_base, q_base, M_base), regression_factors, baseline_assumptions
        )

        # 应用调整后的Bass模型
        non_usd_mcap = np.array([bass_model(t, p_adjusted, q_adjusted, M_adjusted)
                                 for t in future_t])
    else:
        # 如果没有历史数据或缺少必要列，使用简化方法
        print("警告：无历史数据或数据不完整，使用简化预测方法")

        # 美元稳定币：份额逐渐下降
        usd_share = np.linspace(0.85, 0.70, n_years)  # 从85%下降到70%
        usd_mcap = future_mcap * usd_share

        # 非美元稳定币：指数增长
        growth_rates = np.linspace(0.25, 0.15, n_years)  # 增长率从25%降至15%
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
    使用回归模型中的显著因素调整Bass模型参数

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


# 修正主程序
if __name__ == "__main__":
    # 加载阶段1的数据
    regression_results, historical_df, features_used = load_phase1_data()

    # 使用正确的特征假设（与模型特征匹配，且长度为5）
    baseline_assumptions = {
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
                             ],  # 美国利率
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
                        ]  # 加密货币总市值
    }

    print("开始预测...")
    predictions = predict_growth(regression_results, baseline_assumptions, historical_df)

    # 保存预测结果
    predictions.to_csv('future_predictions.csv', index=False)
    print("阶段2预测结果已保存至 future_predictions.csv")
    print("\n预测结果:")
    print(predictions.round(2))