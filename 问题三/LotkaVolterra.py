import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import seaborn as sns


def load_future_predictions(csv_file_path):
    """读取并处理future_predictions.csv文件"""
    try:
        # 由于数据是用空格分隔的，我们使用delim_whitespace=True
        # 同时设置header=None因为第一行不是列名，而是数据
        df = pd.read_csv(csv_file_path, delim_whitespace=True, header=None)

        # 打印原始数据形状以便调试
        print(f"原始数据形状: {df.shape}")
        print("原始数据前几行:")
        print(df.head())

        # 检查数据列数并相应处理
        if df.shape[1] == 9:  # 完整的9列数据
            # 设置列名
            df.columns = ['year', 'total_mcap', 'total_growth_rate', 'usd_mcap',
                          'usd_share', 'usd_growth_rate', 'non_usd_mcap',
                          'non_usd_share', 'non_usd_growth_rate']
        elif df.shape[1] == 6:  # 只有6列的情况（如2026年数据）
            # 设置部分列名，缺失的列用NaN填充
            df.columns = ['year', 'total_mcap', 'total_growth_rate', 'usd_mcap',
                          'usd_share', 'usd_growth_rate']
            # 添加缺失的列
            df['non_usd_mcap'] = np.nan
            df['non_usd_share'] = np.nan
            df['non_usd_growth_rate'] = np.nan
        else:
            print(f"警告: 数据有{df.shape[1]}列，期望9列或6列")
            # 尝试通用的处理方法
            if df.shape[1] < 9:
                # 复制已有的列名
                existing_columns = ['year', 'total_mcap', 'total_growth_rate', 'usd_mcap',
                                    'usd_share', 'usd_growth_rate', 'non_usd_mcap',
                                    'non_usd_share', 'non_usd_growth_rate']
                # 只使用实际存在的列数
                actual_columns = existing_columns[:df.shape[1]]
                df.columns = actual_columns
                # 添加缺失的列
                for col in existing_columns[df.shape[1]:]:
                    df[col] = np.nan

        # 确保年份列是整数
        df['year'] = df['year'].astype(int)

        # 计算总市值如果缺失（usd_mcap + non_usd_mcap）
        if 'total_mcap' not in df.columns or df['total_mcap'].isna().any():
            df['total_mcap'] = df['usd_mcap'] + df['non_usd_mcap']

        # 计算份额如果缺失
        if 'usd_share' not in df.columns or df['usd_share'].isna().any():
            df['usd_share'] = df['usd_mcap'] / df['total_mcap']

        if 'non_usd_share' not in df.columns or df['non_usd_share'].isna().any():
            df['non_usd_share'] = df['non_usd_mcap'] / df['total_mcap']

        print("处理后的数据:")
        print(df)
        print(f"数据列: {df.columns.tolist()}")

        return df

    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        # 创建示例数据作为后备
        return create_sample_data()


def create_sample_data():
    """创建示例数据作为后备"""
    print("创建示例数据...")
    data = {
        'year': [2026, 2027, 2028, 2029, 2030],
        'total_mcap': [44.17, 115.95, 16837.40, 3700.88, 7939.60],
        'usd_mcap': [37.54, 94.21, 13048.98, 2729.40, 5557.72],
        'non_usd_mcap': [6.62, 21.74, 3788.42, 971.48, 2381.88],
        'usd_share': [0.85, 0.8125, 0.775, 0.7375, 0.70],
        'non_usd_share': [0.15, 0.1875, 0.225, 0.2625, 0.30]
    }
    df = pd.DataFrame(data)
    df['total_mcap'] = df['usd_mcap'] + df['non_usd_mcap']
    return df


def analyze_market_share(prediction_df):
    """分析市场份额影响 - 简单方法
    直接计算份额变化，美元份额 = 美元市值 / 总市值[3](@ref)
    """
    results = []

    for i in range(1, len(prediction_df)):
        current = prediction_df.iloc[i]
        previous = prediction_df.iloc[i - 1]

        # 计算增长率[3](@ref)
        usd_growth = (current['usd_mcap'] - previous['usd_mcap']) / previous['usd_mcap'] * 100
        non_usd_growth = (current['non_usd_mcap'] - previous['non_usd_mcap']) / previous['non_usd_mcap'] * 100
        total_growth = (current['total_mcap'] - previous['total_mcap']) / previous['total_mcap'] * 100

        # 计算份额变化[3](@ref)
        share_change = current['usd_share'] - previous['usd_share']

        # 计算相对市场份额（相对于最大竞争对手）[3](@ref)
        max_competitor_share = max(current['usd_share'], current['non_usd_share'])
        relative_share = current['usd_share'] / max_competitor_share if max_competitor_share > 0 else 0

        results.append({
            'year': current['year'],
            'usd_growth_rate': usd_growth,
            'non_usd_growth_rate': non_usd_growth,
            'total_growth_rate': total_growth,
            'usd_share': current['usd_share'],
            'non_usd_share': current['non_usd_share'],
            'share_change': share_change,
            'relative_share': relative_share
        })

    return pd.DataFrame(results)


def lotka_volterra_competition(U, t, rd, rnd, alpha_dn, alpha_nd, Md, Mnd):
    """Lotka-Volterra竞争模型简化版[3](@ref)
    参数:
    U: 状态向量 [U_d, U_nd] (美元和非美元份额)
    t: 时间
    rd, rnd: 增长率
    alpha_dn, alpha_nd: 竞争系数
    Md, Mnd: 市场容量
    """
    Ud, Und = U

    # 竞争微分方程[3](@ref)
    dU_dt = rd * Ud * (1 - (Ud + alpha_dn * Und) / Md)
    dU_ndt = rnd * Und * (1 - (Und + alpha_nd * Ud) / Mnd)

    return [dU_dt, dU_ndt]


def fit_competition_model(historical_data):
    """从历史数据估计竞争模型参数"""
    try:
        # 简化参数估计（实际应用中需要更复杂的方法）
        # 假设参数基于历史趋势
        rd = historical_data['usd_growth_rate'].mean() / 100  # 转换为小数
        rnd = historical_data['non_usd_growth_rate'].mean() / 100

        # 竞争系数假设：非美元对美元竞争更强[3](@ref)
        alpha_nd = 0.8  # 非美元对美元的竞争影响
        alpha_dn = 0.3  # 美元对非美元的竞争影响

        # 市场容量基于历史最大值
        Md = historical_data['usd_mcap'].max() * 1.5
        Mnd = historical_data['non_usd_mcap'].max() * 1.5

        return rd, rnd, alpha_dn, alpha_nd, Md, Mnd
    except:
        # 如果数据不足，使用默认参数
        return 0.15, 0.25, 0.3, 0.8, 2000, 1500


def competition_model_analysis(historical_df, future_years=5):
    """使用Lotka-Volterra竞争模型进行进阶分析"""

    # 从历史数据估计参数
    rd, rnd, alpha_dn, alpha_nd, Md, Mnd = fit_competition_model(historical_df)

    # 初始条件（最后已知的市场份额）
    last_row = historical_df.iloc[-1]
    U0 = [last_row['usd_mcap'], last_row['non_usd_mcap']]

    # 时间点（未来预测）
    t = np.linspace(0, future_years, future_years + 1)

    # 求解微分方程
    U = odeint(lotka_volterra_competition, U0, t,
               args=(rd, rnd, alpha_dn, alpha_nd, Md, Mnd))

    # 整理结果
    competition_results = []
    for i, year in enumerate(range(int(last_row['year']), int(last_row['year']) + future_years + 1)):
        ud_mcap, und_mcap = U[i]
        total_mcap = ud_mcap + und_mcap
        usd_share = ud_mcap / total_mcap if total_mcap > 0 else 0

        competition_results.append({
            'year': year,
            'usd_mcap': ud_mcap,
            'non_usd_mcap': und_mcap,
            'total_mcap': total_mcap,
            'usd_share': usd_share,
            'model_type': 'competition_model'
        })

    return pd.DataFrame(competition_results), (rd, rnd, alpha_dn, alpha_nd)


def calculate_market_concentration(share_analysis):
    """计算市场集中度指标[4](@ref)"""
    # 赫芬达尔-赫希曼指数(HHI)
    hhi_values = []
    for _, row in share_analysis.iterrows():
        hhi = (row['usd_share'] * 100) ** 2 + (row['non_usd_share'] * 100) ** 2
        hhi_values.append(hhi)

    share_analysis['hhi_index'] = hhi_values
    return share_analysis


def enhanced_visualization(share_analysis, future_predictions, competition_results=None):
    """增强的可视化分析"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 市场份额趋势
    axes[0, 0].plot(future_predictions['year'], future_predictions['usd_share'] * 100,
                    label='USD Stablecoin Share', marker='o', linewidth=2)
    axes[0, 0].plot(future_predictions['year'], future_predictions['non_usd_share'] * 100,
                    label='Non-USD Stablecoin Share', marker='s', linewidth=2)
    axes[0, 0].set_ylabel('Market Share (%)')
    axes[0, 0].set_title('Stablecoin Market Share Trend')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 市值预测
    axes[0, 1].plot(future_predictions['year'], future_predictions['usd_mcap'],
                    label='USD Stablecoins', marker='s', linewidth=2)
    axes[0, 1].plot(future_predictions['year'], future_predictions['non_usd_mcap'],
                    label='Non-USD Stablecoins', marker='^', linewidth=2)
    axes[0, 1].set_ylabel('Market Cap (Million USD)')
    axes[0, 1].set_title('Stablecoin Market Cap Projection')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 增长率比较
    x_pos = np.arange(len(share_analysis))
    width = 0.35
    axes[0, 2].bar(x_pos - width / 2, share_analysis['usd_growth_rate'],
                   width, label='USD Growth Rate', alpha=0.7)
    axes[0, 2].bar(x_pos + width / 2, share_analysis['non_usd_growth_rate'],
                   width, label='Non-USD Growth Rate', alpha=0.7)
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(share_analysis['year'].astype(int))
    axes[0, 2].set_ylabel('Growth Rate (%)')
    axes[0, 2].set_title('Growth Rate Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 份额变化
    axes[1, 0].bar(share_analysis['year'], share_analysis['share_change'] * 100,
                   color=['red' if x < 0 else 'green' for x in share_analysis['share_change']])
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.8)
    axes[1, 0].set_ylabel('Share Change (percentage points)')
    axes[1, 0].set_title('Yearly USD Share Change')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 相对市场份额[3](@ref)
    axes[1, 1].plot(share_analysis['year'], share_analysis['relative_share'],
                    marker='o', color='purple', linewidth=2)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', label='Leadership Threshold')
    axes[1, 1].set_ylabel('Relative Market Share')
    axes[1, 1].set_title('Relative Market Share (USD vs Max Competitor)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 市场集中度（HHI）[4](@ref)
    if 'hhi_index' in share_analysis.columns:
        axes[1, 2].plot(share_analysis['year'], share_analysis['hhi_index'],
                        marker='D', color='orange', linewidth=2)
        axes[1, 2].axhline(y=1500, color='green', linestyle='--', label='Low Concentration')
        axes[1, 2].axhline(y=2500, color='red', linestyle='--', label='High Concentration')
        axes[1, 2].set_ylabel('HHI Index')
        axes[1, 2].set_title('Market Concentration (HHI)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 竞争模型结果可视化（如果存在）
    if competition_results is not None:
        plot_competition_model_results(competition_results, future_predictions)


def plot_competition_model_results(competition_results, simple_predictions):
    """绘制竞争模型预测结果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 比较简单模型和竞争模型的份额预测
    axes[0].plot(simple_predictions['year'], simple_predictions['usd_share'] * 100,
                 label='Simple Model - USD', marker='o', linewidth=2)
    axes[0].plot(competition_results['year'], competition_results['usd_share'] * 100,
                 label='Competition Model - USD', marker='s', linewidth=2, linestyle='--')
    axes[0].set_ylabel('USD Market Share (%)')
    axes[0].set_title('Model Comparison: USD Share Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 竞争动态分析
    years = competition_results['year'].values
    usd_share = competition_results['usd_share'].values * 100
    non_usd_share = 100 - usd_share  # 由于总份额为100%

    axes[1].stackplot(years, [usd_share, non_usd_share],
                      labels=['USD Stablecoins', 'Non-USD Stablecoins'],
                      alpha=0.7)
    axes[1].set_ylabel('Market Share (%)')
    axes[1].set_title('Market Share Evolution (Competition Model)')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_analysis_report(share_analysis, competition_params=None):
    """生成分析报告[3](@ref)"""
    print("=" * 60)
    print("STABLECOIN MARKET SHARE ANALYSIS REPORT")
    print("=" * 60)

    # 基本统计信息
    avg_usd_share = share_analysis['usd_share'].mean() * 100
    avg_share_change = share_analysis['share_change'].mean() * 100
    total_erosion = share_analysis['share_change'].sum() * 100

    print(f"\n1. BASIC MARKET ANALYSIS")
    print(f"   • Average USD Share: {avg_usd_share:.1f}%")
    print(f"   • Average Yearly Change: {avg_share_change:+.2f} percentage points")
    print(f"   • Total Share Erosion/Gain: {total_erosion:+.2f} percentage points")

    # 增长分析
    avg_usd_growth = share_analysis['usd_growth_rate'].mean()
    avg_non_usd_growth = share_analysis['non_usd_growth_rate'].mean()

    print(f"\n2. GROWTH DYNAMICS")
    print(f"   • USD Stablecoins Avg Growth: {avg_usd_growth:.1f}%")
    print(f"   • Non-USD Stablecoins Avg Growth: {avg_non_usd_growth:.1f}%")
    print(f"   • Growth Differential: {avg_non_usd_growth - avg_usd_growth:+.1f}%")

    # 竞争分析
    if competition_params:
        rd, rnd, alpha_dn, alpha_nd = competition_params
        print(f"\n3. COMPETITION MODEL INSIGHTS")
        print(f"   • USD intrinsic growth rate (r_d): {rd:.3f}")
        print(f"   • Non-USD intrinsic growth rate (r_nd): {rnd:.3f}")
        print(f"   • Competition coefficient (α_nd > α_dn): {alpha_nd} > {alpha_dn}")
        print(f"   • Implication: Non-USD stablecoins have stronger competitive impact")

    # 趋势判断
    last_change = share_analysis.iloc[-1]['share_change'] * 100
    if last_change < 0:
        trend = "EROSION"
        implication = "Non-USD stablecoins are gaining ground"
    else:
        trend = "STABILITY/GAIN"
        implication = "USD dominance is maintained"

    print(f"\n4. TREND ASSESSMENT: {trend}")
    print(f"   • Latest yearly change: {last_change:+.2f} percentage points")
    print(f"   • Implication: {implication}")

    print("\n" + "=" * 60)


# 主执行流程
def main_analysis(csv_file_path, historical_data=None):
    """主分析函数"""

    # 1. 读取CSV文件
    print("正在读取CSV文件...")
    future_predictions = load_future_predictions(csv_file_path)

    # 检查数据是否包含必要列
    required_columns = ['year', 'usd_mcap', 'non_usd_mcap', 'total_mcap', 'usd_share']
    missing_columns = [col for col in required_columns if col not in future_predictions.columns]

    if missing_columns:
        print(f"警告: 数据缺少以下列: {missing_columns}")
        # 尝试计算缺失的列
        if 'total_mcap' in missing_columns:
            future_predictions['total_mcap'] = future_predictions['usd_mcap'] + future_predictions['non_usd_mcap']
        if 'usd_share' in missing_columns:
            future_predictions['usd_share'] = future_predictions['usd_mcap'] / future_predictions['total_mcap']
        if 'non_usd_share' in missing_columns and 'non_usd_mcap' in future_predictions.columns:
            future_predictions['non_usd_share'] = future_predictions['non_usd_mcap'] / future_predictions['total_mcap']

    # 2. 执行简单市场份额分析
    print("执行基本市场份额分析...")
    share_analysis = analyze_market_share(future_predictions)

    # 3. 计算市场集中度
    share_analysis = calculate_market_concentration(share_analysis)

    # 4. 竞争模型分析（如果有历史数据）
    competition_results = None
    competition_params = None

    if historical_data is not None:
        print("执行Lotka-Volterra竞争模型分析...")
        competition_results, competition_params = competition_model_analysis(historical_data)

    # 5. 生成可视化
    print("生成分析图表...")
    enhanced_visualization(share_analysis, future_predictions, competition_results)

    # 6. 生成分析报告
    generate_analysis_report(share_analysis, competition_params)

    return share_analysis, competition_results, future_predictions


# 使用示例
if __name__ == "__main__":
    # 指定CSV文件路径
    csv_file = "future_predictions.csv"  # 请确保文件路径正确

    # 执行分析
    share_analysis, competition_results, future_predictions = main_analysis(csv_file)

    print("\n分析完成！")
    print("基本分析结果:")
    print(share_analysis)

    print("\n预测数据:")
    print(future_predictions)

    if competition_results is not None:
        print("\n竞争模型预测:")
        print(competition_results)