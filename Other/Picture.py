import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def create_ahp_hierarchy_diagram():
    """创建AHP层次结构图（图3.1）"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 清除坐标轴
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 标题
    ax.text(5, 7.5, '图3.1 稳定币AHP评价指标体系层次结构',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # 第一层：目标层
    target_box = FancyBboxPatch((3.5, 6.2), 3, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='lightblue',
                                edgecolor='black',
                                linewidth=2)
    ax.add_patch(target_box)
    ax.text(5, 6.6, '目标层\n稳定币综合评估',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # 第二层：准则层（七个维度）
    criteria_labels = [
        '监管合规性\n(0.242)', '储备透明度\n(0.066)', '历史稳定性\n0.126)',
        '流动性风险\n(0.126)', '市场占有率\n(0.186)', '技术安全\n(0.186)',
        '风险集中度\n(0.066)'
    ]

    criteria_positions = np.linspace(0.5, 9, 7)

    # 绘制准则层框和连线
    for i, (label, x) in enumerate(zip(criteria_labels, criteria_positions)):
        # 准则框
        criteria_box = FancyBboxPatch((x - 0.8, 4.2), 1.6, 0.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor='lightgreen',
                                      edgecolor='black')
        ax.add_patch(criteria_box)
        ax.text(x, 4.6, label, ha='center', va='center', fontsize=10)

        # 连线（目标层到准则层）
        ax.plot([5, x], [6.2, 5], 'k-', alpha=0.6, linewidth=1)

    # 第三层：方案层
    solution_box1 = FancyBboxPatch((2, 2), 2, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightcoral',
                                   edgecolor='black')
    solution_box2 = FancyBboxPatch((6, 2), 2, 0.8,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightcoral',
                                   edgecolor='black')

    ax.add_patch(solution_box1)
    ax.add_patch(solution_box2)
    ax.text(3, 2.4, '方案层\nUSDT', ha='center', va='center', fontsize=11)
    ax.text(7, 2.4, '方案层\nUSDC', ha='center', va='center', fontsize=11)

    # 连线（准则层到方案层）
    for x in criteria_positions:
        ax.plot([x, 3], [4.2, 2.8], 'k--', alpha=0.4, linewidth=0.8)
        ax.plot([x, 7], [4.2, 2.8], 'k--', alpha=0.4, linewidth=0.8)

    # 添加说明文本
    explanation_text = (
        "层次说明：\n"
        "• 目标层：稳定币综合评估的最终目标\n"
        "• 准则层：7个评价维度及其AHP权重（CR=0.05<0.1）\n"
        "• 方案层：待评估的具体稳定币方案\n"
        "连线表示各层次间的逻辑关联和权重传递关系"
    )

    ax.text(0.5, 1, explanation_text, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))

    # 添加数学公式说明
    formula_text = (
        "核心数学模型：\n"
        "权重计算：$w_i = \\frac{1}{n}\\sum_{j=1}^{n}\\frac{a_{ij}}{\\sum_{i=1}^{n}a_{ij}}$\n"
        "一致性检验：$CI = \\frac{\\lambda_{max}-n}{n-1},\\ CR = \\frac{CI}{RI}$\n"
        "综合得分：$S_{综合} = \\sum_{i=1}^{7}w_i \\times s_i$"
    )

    ax.text(6, 1, formula_text, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lavender'))

    plt.tight_layout()
    return fig, ax


def create_simplified_ahp_diagram():
    """创建简化版AHP层次结构图（适合论文排版）"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 层次标题
    levels = ['目标层', '准则层', '方案层']
    level_y = [5, 3, 1]

    for i, (level, y) in enumerate(zip(levels, level_y)):
        ax.text(0.5, y, level, fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray'))

    # 目标层
    ax.text(5, 5, '稳定币综合评估', ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))

    # 准则层（七个维度）
    criteria = [
        ('监管合规性', 0.242), ('储备透明度', 0.066), ('历史稳定性', 0.126),
        ('流动性风险', 0.126), ('市场占有率', 0.186), ('技术安全', 0.186),
        ('风险集中度', 0.066)
    ]

    criteria_x = np.linspace(1.5, 8.5, 7)
    for i, ((name, weight), x) in enumerate(zip(criteria, criteria_x)):
        # 准则框
        ax.text(x, 3, f'{name}\n({weight})', ha='center', va='center',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'))

        # 连线
        ax.plot([5, x], [4.7, 3.3], 'k-', alpha=0.6, linewidth=1)

    # 方案层
    solutions = ['USDT', 'USDC']
    solution_x = [3, 7]

    for i, (name, x) in enumerate(zip(solutions, solution_x)):
        ax.text(x, 1, name, ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))

        # 连线（从所有准则到方案）
        for crit_x in criteria_x:
            ax.plot([crit_x, x], [2.7, 1.3], 'k--', alpha=0.3, linewidth=0.5)

    # 图标题
    ax.text(5, 5.5, '图3.1 稳定币AHP评价指标体系层次结构',
            ha='center', va='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    return fig, ax


# 生成图表
if __name__ == "__main__":
    # 生成详细版本
    fig1, ax1 = create_ahp_hierarchy_diagram()
    # plt.savefig('AHP层次结构图_详细版.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 生成简化版本（更适合论文）
    fig2, ax2 = create_simplified_ahp_diagram()
    # plt.savefig('AHP层次结构图_简化版.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ 图3.1 AHP层次结构图已生成完成")
    print("✓ 已保存为高分辨率PNG文件，可直接插入论文")
    print("✓ 包含详细版和简化版两个版本，建议在论文中使用简化版")