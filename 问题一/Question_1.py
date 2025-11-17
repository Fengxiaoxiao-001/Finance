import numpy as np


class StableCoinAHPModel:
    """稳定币多维度分析模型 - 基于AHP方法"""

    def __init__(self):
        # 七个核心维度
        self.dimensions = [
            '监管合规性', '储备透明度', '历史稳定性',
            '流动性风险', '市场占有率', '技术安全', '风险集中度得分'
        ]

        # 初始化判断矩阵（基于文档重要性和专家判断）
        self.judgment_matrix = np.array([
            [1, 3, 2, 2, 1.5, 1.5, 3],  # 监管合规性
            [1 / 3, 1, 1 / 2, 1 / 2, 1 / 3, 1 / 3, 1],  # 储备透明度
            [1 / 2, 2, 1, 1, 2 / 3, 2 / 3, 2],  # 历史稳定性
            [1 / 2, 2, 1, 1, 2 / 3, 2 / 3, 2],  # 流动性风险
            [2 / 3, 3, 1.5, 1.5, 1, 1, 3],  # 市场占有率
            [2 / 3, 3, 1.5, 1.5, 1, 1, 3],  # 技术安全
            [1 / 3, 1, 1 / 2, 1 / 2, 1 / 3, 1 / 3, 1]  # 风险集中度得分
        ])

    def calculate_ahp_weights(self):
        """计算AHP权重[1,3](@ref)"""
        # 1. 计算判断矩阵的列和
        col_sums = self.judgment_matrix.sum(axis=0)

        # 2. 归一化处理
        normalized_matrix = self.judgment_matrix / col_sums

        # 3. 计算行平均值得到权重
        weights = normalized_matrix.mean(axis=1)

        # 4. 一致性检验[2](@ref)
        n = len(self.dimensions)
        lambda_max = (self.judgment_matrix @ weights / weights).mean()
        ci = (lambda_max - n) / (n - 1)

        # 随机一致性指标RI（n=7）
        ri = 1.32
        cr = ci / ri

        return weights, cr

    def calculate_comprehensive_score(self, scores, weights):
        """计算综合得分"""
        return np.dot(scores, weights)

    def calculate_scene_score(self, scores):
        """计算应用场景得分 - 修正权重分配"""
        # 所有维度都应参与计算，权重可调整但不应为0
        scene_weights = np.array([0.15, 0.1, 0.2, 0.25, 0.1, 0.15, 0.05])
        return np.dot(scores, scene_weights)

    def calculate_competitiveness_score(self, scores):
        """计算市场竞争力得分 - 修正权重分配"""
        comp_weights = np.array([0.2, 0.1, 0.15, 0.15, 0.25, 0.1, 0.05])
        return np.dot(scores, comp_weights)

    def calculate_risk_score(self, scores):
        """计算潜在风险得分[4](@ref)"""
        # 风险维度权重：监管0.25, 透明度0.25, 稳定性0.2, 集中度0.3
        risk_weights = np.array([0.25, 0.25, 0.2, 0, 0, 0, 0.3])
        # 风险得分 = Σ(权重 × (10 - 维度得分))
        risk_scores = 10 - scores
        return np.dot(risk_scores, risk_weights)

    def calculate_potential_score(self, scores):
        """计算发展潜力得分"""
        # 权重分配：市场0.4, 合规0.3, 技术0.2, 流动性0.1
        potential_weights = np.array([0.3, 0, 0, 0.1, 0.4, 0.2, 0])
        return np.dot(scores, potential_weights)

    def analyze_stablecoins(self):
        """主分析函数"""
        # 计算AHP权重
        weights, cr = self.calculate_ahp_weights()

        # USDT和USDC的维度得分（基于您的数据）
        usdt_scores = np.array([8,3,9,6,5.8,10,7.7])  # 监管合规性, 储备透明度, 历史稳定性, 流动性风险, 市场占有率, 技术安全, 风险集中度得分
        usdc_scores = np.array([10,5,9,8,2.39,3,2.3])

        # 计算各项得分
        results = {
            '维度权重': dict(zip(self.dimensions, weights)),
            '一致性比率CR': cr,
            'USDT': {
                '综合得分': self.calculate_comprehensive_score(usdt_scores, weights),
                '应用场景得分': self.calculate_scene_score(usdt_scores),
                '市场竞争力得分': self.calculate_competitiveness_score(usdt_scores),
                '潜在风险得分': self.calculate_risk_score(usdt_scores),
                '发展潜力得分': self.calculate_potential_score(usdt_scores)
            },
            'USDC': {
                '综合得分': self.calculate_comprehensive_score(usdc_scores, weights),
                '应用场景得分': self.calculate_scene_score(usdc_scores),
                '市场竞争力得分': self.calculate_competitiveness_score(usdc_scores),
                '潜在风险得分': self.calculate_risk_score(usdc_scores),
                '发展潜力得分': self.calculate_potential_score(usdc_scores)
            }
        }

        return results


def visualize_results(results):
    """可视化分析结果"""
    import matplotlib.pyplot as plt

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建对比图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 综合得分对比
    categories = ['综合得分', '应用场景', '市场竞争力', '发展潜力']
    usdt_scores = [
        results['USDT']['综合得分'],
        results['USDT']['应用场景得分'],
        results['USDT']['市场竞争力得分'],
        results['USDT']['发展潜力得分']
    ]
    usdc_scores = [
        results['USDC']['综合得分'],
        results['USDC']['应用场景得分'],
        results['USDC']['市场竞争力得分'],
        results['USDC']['发展潜力得分']
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, usdt_scores, width, label='USDT', alpha=0.8)
    bars2 = ax1.bar(x + width / 2, usdc_scores, width, label='USDC', alpha=0.8)
    ax1.set_ylabel('得分')
    ax1.set_title('USDT vs USDC 综合对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()

    # 在条形图上方添加数字标签
    for bar, score in zip(bars1, usdt_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{score:.2f}', ha='center', va='bottom', fontsize=9)

    for bar, score in zip(bars2, usdc_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{score:.2f}', ha='center', va='bottom', fontsize=9)

    # 2. 风险对比
    risk_data = [results['USDT']['潜在风险得分'], results['USDC']['潜在风险得分']]
    bars_risk = ax2.bar(['USDT', 'USDC'], risk_data, color=['red', 'green'], alpha=0.7)
    ax2.set_ylabel('风险得分(越高风险越大)')
    ax2.set_title('潜在风险对比')

    # 在风险条形图上方添加数字标签
    for bar, score in zip(bars_risk, risk_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{score:.2f}', ha='center', va='bottom', fontsize=10)

    # 3. 维度权重分布
    weights = list(results['维度权重'].values())
    ax3.pie(weights, labels=results['维度权重'].keys(), autopct='%1.1f%%')
    ax3.set_title('AHP维度权重分布')

    # 4. 雷达图对比
    dimensions = list(results['维度权重'].keys())
    usdt_dim_scores = [8,3,9,6,5.8,10,7.7]
    usdc_dim_scores = [10,5,9,8,2.39,3,2.3]

    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    usdt_dim_scores += usdt_dim_scores[:1]
    usdc_dim_scores += usdc_dim_scores[:1]
    dimensions += dimensions[:1]

    ax4 = plt.subplot(2, 2, 4, polar=True)
    ax4.plot(angles, usdt_dim_scores, 'o-', linewidth=2, label='USDT')
    ax4.fill(angles, usdt_dim_scores, alpha=0.25)
    ax4.plot(angles, usdc_dim_scores, 'o-', linewidth=2, label='USDC')
    ax4.fill(angles, usdc_dim_scores, alpha=0.25)
    ax4.set_thetagrids(np.degrees(angles[:-1]), dimensions[:-1])
    ax4.set_title('七个维度对比雷达图')
    ax4.legend()

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    model = StableCoinAHPModel()

    # 进行分析
    results = model.analyze_stablecoins()

    # 打印结果
    print("=" * 60)
    print("稳定币多维度分析结果")
    print("=" * 60)

    print(f"\nAHP一致性比率(CR): {results['一致性比率CR']:.4f}")
    if results['一致性比率CR'] < 0.1:
        print("✓ 判断矩阵通过一致性检验")
    else:
        print("⚠ 判断矩阵需要调整")

    print("\n各维度权重分布:")
    for dim, weight in results['维度权重'].items():
        print(f"  {dim}: {weight:.3f}")

    print("\n" + "=" * 40)
    print("USDT分析结果:")
    print("=" * 40)
    for key, value in results['USDT'].items():
        print(f"{key}: {value:.2f}")

    print("\n" + "=" * 40)
    print("USDC分析结果:")
    print("=" * 40)
    for key, value in results['USDC'].items():
        print(f"{key}: {value:.2f}")

    # 对比分析
    print("\n" + "=" * 50)
    print("对比分析结论:")
    print("=" * 50)

    usdt_total = results['USDT']['综合得分']
    usdc_total = results['USDC']['综合得分']

    if usdc_total > usdt_total:
        advantage = ((usdc_total - usdt_total) / usdt_total) * 100
        print(f"✓ USDC综合得分领先 {advantage:.1f}%")
    else:
        advantage = ((usdt_total - usdc_total) / usdc_total) * 100
        print(f"✓ USDT综合得分领先 {advantage:.1f}%")

    # 专项优势分析
    if results['USDC']['应用场景得分'] > results['USDT']['应用场景得分']:
        print("✓ USDC在应用场景方面更具优势")
    else:
        print("✓ USDT在应用场景方面更具优势")

    if results['USDT']['市场竞争力得分'] > results['USDC']['市场竞争力得分']:
        print("✓ USDT在市场竞争力方面保持优势")
    else:
        print("✓ USDC在市场竞争力方面表现更好")

    if results['USDT']['潜在风险得分'] > results['USDC']['潜在风险得分']:
        print("⚠ USDT潜在风险较高，需重点关注")
    else:
        print("✓ USDC风险控制较好")

    # 生成可视化图表
    try:
        visualize_results(results)
        print("\n✓ 可视化图表已生成完成")
    except:
        print("\n⚠ 可视化功能需要matplotlib库支持")
        print("请运行: pip install matplotlib")