import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data(filepath):
    """加载并预处理绘图数据"""
    df = pd.read_csv(filepath)

    # 数据清洗和类型转换
    df['stablecoin_size'] = pd.to_numeric(df['stablecoin_size'], errors='coerce')
    df['return_to_liquidity_ratio'] = pd.to_numeric(df['return_to_liquidity_ratio'], errors='coerce')
    df['total_return'] = pd.to_numeric(df['total_return'], errors='coerce')
    df['total_risk'] = pd.to_numeric(df['total_risk'], errors='coerce')

    # 删除无效数据
    df = df.dropna(subset=['stablecoin_size', 'return_to_liquidity_ratio'])

    return df


def create_core_analysis_plots(df, stablecoin_type):
    """创建核心分析图表 - 优化版本：年化收益率使用点图"""

    # 筛选特定稳定币类型的数据
    coin_data = df[df['stablecoin_type'] == stablecoin_type]

    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{stablecoin_type}稳定币储备优化核心分析', fontsize=16, fontweight='bold', y=0.95)

    # 设置颜色和标记
    markers = ['o', 's', '^']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    regime_labels = {'normal': '正常市场', 'stress': '压力市场', 'extreme': '极端市场'}

    # 1. 核心指标：收益-流动性风险比率 vs 规模（保持线图）
    ax1 = axes[0, 0]
    for i, regime in enumerate(['normal', 'stress', 'extreme']):
        regime_data = coin_data[coin_data['market_regime'] == regime]
        if len(regime_data) > 0:
            # 对数据进行排序以确保线图正确连接
            sorted_data = regime_data.sort_values('stablecoin_size')
            ax1.plot(sorted_data['stablecoin_size'],
                     sorted_data['return_to_liquidity_ratio'],
                     label=regime_labels[regime],
                     marker=markers[i], markersize=6, linewidth=2, color=colors[i], alpha=0.8)

    ax1.set_title('收益-流动性风险比率 vs 规模', fontsize=14, fontweight='bold')
    ax1.set_xlabel('稳定币规模 (十亿美元)', fontsize=12)
    ax1.set_ylabel('收益-流动性风险比率', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 年化收益率 vs 规模 - 改为点图
    ax2 = axes[0, 1]
    for i, regime in enumerate(['normal', 'stress', 'extreme']):
        regime_data = coin_data[coin_data['market_regime'] == regime]
        if len(regime_data) > 0:
            ax2.scatter(regime_data['stablecoin_size'],
                        regime_data['total_return'] * 100,  # 转换为百分比
                        label=regime_labels[regime],
                        marker=markers[i], s=80, color=colors[i], alpha=0.7)

            # 可选：添加趋势线
            if len(regime_data) > 1:
                # 按规模排序
                sorted_data = regime_data.sort_values('stablecoin_size')
                z = np.polyfit(sorted_data['stablecoin_size'],
                               sorted_data['total_return'] * 100, 1)
                p = np.poly1d(z)
                ax2.plot(sorted_data['stablecoin_size'],
                         p(sorted_data['stablecoin_size']),
                         color=colors[i], linewidth=1.5, alpha=0.5, linestyle='--')

    ax2.set_title('年化收益率 vs 规模', fontsize=14, fontweight='bold')
    ax2.set_xlabel('稳定币规模 (十亿美元)', fontsize=12)
    ax2.set_ylabel('年化收益率 (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 流动性风险 vs 规模（保持线图）
    ax3 = axes[1, 0]
    for i, regime in enumerate(['normal', 'stress', 'extreme']):
        regime_data = coin_data[coin_data['market_regime'] == regime]
        if len(regime_data) > 0:
            sorted_data = regime_data.sort_values('stablecoin_size')
            ax3.plot(sorted_data['stablecoin_size'],
                     sorted_data['total_risk'],
                     label=regime_labels[regime],
                     marker=markers[i], markersize=6, linewidth=2, color=colors[i], alpha=0.8)

    ax3.set_title('流动性风险 vs 规模', fontsize=14, fontweight='bold')
    ax3.set_xlabel('稳定币规模 (十亿美元)', fontsize=12)
    ax3.set_ylabel('流动性风险 (基点)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 风险-收益散点图（按市场状态着色）
    ax4 = axes[1, 1]
    color_map = {'normal': '#2E86AB', 'stress': '#A23B72', 'extreme': '#F18F01'}

    for regime in ['normal', 'stress', 'extreme']:
        regime_data = coin_data[coin_data['market_regime'] == regime]
        if len(regime_data) > 0:
            # 使用不同大小表示规模
            sizes = 50 + (regime_data['stablecoin_size'] / regime_data['stablecoin_size'].max() * 100)
            scatter = ax4.scatter(regime_data['total_risk'],
                                  regime_data['total_return'] * 100,
                                  c=color_map[regime],
                                  label=regime_labels[regime],
                                  alpha=0.7, s=sizes)

    ax4.set_title('风险-收益分布', fontsize=14, fontweight='bold')
    ax4.set_xlabel('流动性风险 (基点)', fontsize=12)
    ax4.set_ylabel('年化收益率 (%)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_comparison_plot_simple(df):
    """创建简化的对比分析图 - 优化版本：年化收益率使用点图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('USDT vs USDC 稳定币优化对比分析', fontsize=16, fontweight='bold', y=0.95)

    # 只分析正常市场情况
    normal_data = df[df['market_regime'] == 'normal']

    # 1. 收益-流动性风险比率对比（保持线图）
    ax1 = axes[0, 0]
    for coin_type in ['USDT', 'USDC']:
        coin_data = normal_data[normal_data['stablecoin_type'] == coin_type]
        if len(coin_data) > 0:
            sorted_data = coin_data.sort_values('stablecoin_size')
            ax1.plot(sorted_data['stablecoin_size'],
                     sorted_data['return_to_liquidity_ratio'],
                     label=coin_type, linewidth=2.5, marker='o', markersize=6)

    ax1.set_title('收益-流动性风险比率对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('稳定币规模 (十亿美元)', fontsize=12)
    ax1.set_ylabel('收益-流动性风险比率', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 年化收益率对比 - 改为点图
    ax2 = axes[0, 1]
    colors = ['#1f77b4', '#ff7f0e']  # USDT蓝色，USDC橙色

    for i, coin_type in enumerate(['USDT', 'USDC']):
        coin_data = normal_data[normal_data['stablecoin_type'] == coin_type]
        if len(coin_data) > 0:
            ax2.scatter(coin_data['stablecoin_size'],
                        coin_data['total_return'] * 100,
                        label=coin_type, s=80, color=colors[i], alpha=0.7)

            # 添加趋势线
            if len(coin_data) > 1:
                sorted_data = coin_data.sort_values('stablecoin_size')
                z = np.polyfit(sorted_data['stablecoin_size'],
                               sorted_data['total_return'] * 100, 1)
                p = np.poly1d(z)
                ax2.plot(sorted_data['stablecoin_size'],
                         p(sorted_data['stablecoin_size']),
                         color=colors[i], linewidth=2, alpha=0.6)

    ax2.set_title('年化收益率对比', fontsize=14, fontweight='bold')
    ax2.set_xlabel('稳定币规模 (十亿美元)', fontsize=12)
    ax2.set_ylabel('年化收益率 (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 风险对比（保持线图）
    ax3 = axes[1, 0]
    for coin_type in ['USDT', 'USDC']:
        coin_data = normal_data[normal_data['stablecoin_type'] == coin_type]
        if len(coin_data) > 0:
            sorted_data = coin_data.sort_values('stablecoin_size')
            ax3.plot(sorted_data['stablecoin_size'],
                     sorted_data['total_risk'],
                     label=coin_type, linewidth=2.5, marker='^', markersize=6)

    ax3.set_title('流动性风险对比', fontsize=14, fontweight='bold')
    ax3.set_xlabel('稳定币规模 (十亿美元)', fontsize=12)
    ax3.set_ylabel('流动性风险 (基点)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 最优规模点分析
    ax4 = axes[1, 1]
    optimal_points = []
    colors = ['#1f77b4', '#ff7f0e']

    for i, coin_type in enumerate(['USDT', 'USDC']):
        coin_data = normal_data[normal_data['stablecoin_type'] == coin_type]
        if len(coin_data) > 0:
            # 找到最优比率对应的点
            optimal_idx = coin_data['return_to_liquidity_ratio'].idxmax()
            optimal_point = coin_data.loc[optimal_idx]
            optimal_points.append(optimal_point)

            # 绘制比率曲线
            sorted_data = coin_data.sort_values('stablecoin_size')
            ax4.plot(sorted_data['stablecoin_size'],
                     sorted_data['return_to_liquidity_ratio'],
                     label=f'{coin_type}比率曲线',
                     linewidth=2, alpha=0.7, color=colors[i])

            # 标记最优点
            ax4.scatter(optimal_point['stablecoin_size'],
                        optimal_point['return_to_liquidity_ratio'],
                        s=150, color=colors[i],
                        label=f'{coin_type}最优点',
                        alpha=0.9, edgecolors='black', linewidth=1.5)

            # 绘制垂直线标记最优规模
            ax4.axvline(x=optimal_point['stablecoin_size'],
                        linestyle='--', alpha=0.5, color=colors[i],
                        label=f'{coin_type}最优规模: {optimal_point["stablecoin_size"]:.1f}B')

    ax4.set_title('最优规模点分析', fontsize=14, fontweight='bold')
    ax4.set_xlabel('稳定币规模 (十亿美元)', fontsize=12)
    ax4.set_ylabel('收益-流动性风险比率', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, optimal_points


def create_market_regime_analysis(df):
    """市场状态敏感性分析"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('市场状态敏感性分析', fontsize=16, fontweight='bold', y=0.95)

    # 1. 不同市场状态下的比率变化
    ax1 = axes[0]
    market_scenarios = ['normal', 'stress', 'extreme']
    regime_labels = ['正常', '压力', '极端']

    for coin_type in ['USDT', 'USDC']:
        ratios_by_regime = []
        for regime in market_scenarios:
            regime_data = df[(df['stablecoin_type'] == coin_type) &
                             (df['market_regime'] == regime)]
            if len(regime_data) > 0:
                avg_ratio = regime_data['return_to_liquidity_ratio'].mean()
                ratios_by_regime.append(avg_ratio)

        if len(ratios_by_regime) == 3:
            x_pos = np.arange(3)
            width = 0.35
            offset = width if coin_type == 'USDC' else 0
            ax1.bar(x_pos + offset, ratios_by_regime, width,
                    label=coin_type, alpha=0.8)

    ax1.set_xlabel('市场状态', fontsize=12)
    ax1.set_ylabel('平均收益-流动性风险比率', fontsize=12)
    ax1.set_title('不同市场状态下的表现', fontsize=14, fontweight='bold')
    ax1.set_xticks(np.arange(3) + width / 2)
    ax1.set_xticklabels(regime_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. 规模弹性分析 - 使用点图显示
    ax2 = axes[1]

    for coin_type in ['USDT', 'USDC']:
        coin_data = df[df['stablecoin_type'] == coin_type]
        normal_data = coin_data[coin_data['market_regime'] == 'normal']

        if len(normal_data) > 0:
            # 计算不同规模区间的平均比率
            size_bins = [0, 1, 5, 10, 15, 20]
            size_labels = ['0-1B', '1-5B', '5-10B', '10-15B', '15-20B']
            normal_data['size_bin'] = pd.cut(normal_data['stablecoin_size'],
                                             bins=size_bins, labels=size_labels)

            bin_means = normal_data.groupby('size_bin')['return_to_liquidity_ratio'].mean()
            # 使用点图而不是线图
            ax2.scatter(range(len(size_labels)), bin_means.values,
                        label=coin_type, s=100, alpha=0.8)

            # 可选：添加连接线
            ax2.plot(range(len(size_labels)), bin_means.values,
                     alpha=0.5, linestyle='-')

    ax2.set_xlabel('稳定币规模区间 (十亿美元)', fontsize=12)
    ax2.set_ylabel('平均收益-流动性风险比率', fontsize=12)
    ax2.set_title('规模弹性分析', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(size_labels)))
    ax2.set_xticklabels(size_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def generate_insightful_summary(df, optimal_points):
    """生成有洞察力的总结"""
    print("=" * 70)
    print("稳定币储备优化关键洞察总结")
    print("=" * 70)

    # 基本统计
    print(f"\n数据概况:")
    print(f"- 总数据点: {len(df)}")
    print(f"- 稳定币类型: {df['stablecoin_type'].unique().tolist()}")
    print(f"- 规模范围: {df['stablecoin_size'].min():.1f} - {df['stablecoin_size'].max():.1f} 十亿美元")
    print(f"- 市场状态: {df['market_regime'].unique().tolist()}")

    # 最优配置分析
    print(f"\n最优配置分析:")
    for i, point in enumerate(optimal_points):
        coin_type = point['stablecoin_type']
        print(f"\n{coin_type}:")
        print(f"  • 最优规模: {point['stablecoin_size']:.1f}十亿美元")
        print(f"  • 最优比率: {point['return_to_liquidity_ratio']:.3f}")
        print(f"  • 对应收益率: {point['total_return'] * 100:.2f}%")
        print(f"  • 对应风险: {point['total_risk']:.1f}基点")

    # 市场状态影响分析
    print(f"\n市场状态敏感性分析:")
    for coin_type in ['USDT', 'USDC']:
        coin_data = df[df['stablecoin_type'] == coin_type]

        normal_data = coin_data[coin_data['market_regime'] == 'normal']
        stress_data = coin_data[coin_data['market_regime'] == 'stress']
        extreme_data = coin_data[coin_data['market_regime'] == 'extreme']

        if len(normal_data) > 0 and len(stress_data) > 0 and len(extreme_data) > 0:
            normal_ratio = normal_data['return_to_liquidity_ratio'].mean()
            stress_ratio = stress_data['return_to_liquidity_ratio'].mean()
            extreme_ratio = extreme_data['return_to_liquidity_ratio'].mean()

            stress_impact = (stress_ratio - normal_ratio) / normal_ratio * 100
            extreme_impact = (extreme_ratio - normal_ratio) / normal_ratio * 100

            print(f"\n{coin_type}市场状态影响:")
            print(f"  • 正常市场平均比率: {normal_ratio:.3f}")
            print(f"  • 压力市场影响: {stress_impact:+.1f}%")
            print(f"  • 极端市场影响: {extreme_impact:+.1f}%")

    # 规模效应分析
    print(f"\n规模效应分析:")
    for coin_type in ['USDT', 'USDC']:
        coin_data = df[df['stablecoin_type'] == coin_type]
        small_scale = coin_data[coin_data['stablecoin_size'] <= 5]
        large_scale = coin_data[coin_data['stablecoin_size'] > 5]

        if len(small_scale) > 0 and len(large_scale) > 0:
            small_ratio = small_scale['return_to_liquidity_ratio'].mean()
            large_ratio = large_scale['return_to_liquidity_ratio'].mean()
            scale_effect = (large_ratio - small_ratio) / small_scale['return_to_liquidity_ratio'].mean() * 100

            print(f"{coin_type}规模效应: {scale_effect:+.1f}% (小型→大型)")


def main():
    # 加载数据
    filepath = r"E:\Preprocessing\Finance\问题二\results\plotting_data_20251106_215146.csv"
    df = load_and_prepare_data(filepath)

    print("数据加载完成!")
    print(f"数据形状: {df.shape}")
    print(f"稳定币类型分布: {df['stablecoin_type'].value_counts().to_dict()}")

    # 创建核心分析图表
    print("\n生成核心分析图表...")

    # USDT分析
    print("生成USDT分析图表...")
    fig_usdt = create_core_analysis_plots(df, 'USDT')
    # fig_usdt.savefig('USDT_核心分析_优化版.png', dpi=300, bbox_inches='tight')

    # USDC分析
    print("生成USDC分析图表...")
    fig_usdc = create_core_analysis_plots(df, 'USDC')
    # fig_usdc.savefig('USDC_核心分析_优化版.png', dpi=300, bbox_inches='tight')

    # 对比分析
    print("生成对比分析图表...")
    fig_compare, optimal_points = create_comparison_plot_simple(df)
    # fig_compare.savefig('USDT_vs_USDC_对比分析_优化版.png', dpi=300, bbox_inches='tight')

    # 市场状态分析
    print("生成市场状态分析图表...")
    fig_market = create_market_regime_analysis(df)
    # fig_market.savefig('市场状态敏感性分析_优化版.png', dpi=300, bbox_inches='tight')

    # 生成洞察总结
    generate_insightful_summary(df, optimal_points)

    plt.show()
    print("\n所有优化版图表已生成完成!")


if __name__ == "__main__":
    main()