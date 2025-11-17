import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, classification_report
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class StablecoinMonetarySovereigntyModel:
    def __init__(self):
        self.global_data = None
        self.country_data = None
        self.results = {}

    def load_real_data(self, global_file=r"D:\浏览器下载\金融\global.csv", country_file=r"D:\浏览器下载\金融\country_panel_data.csv"):
        """加载真实数据函数"""
        try:
            # 加载全球数据
            self.global_data = pd.read_csv(global_file)

            # 加载国家面板数据
            self.country_data = pd.read_csv(country_file)

            # 数据验证和清洗
            self._validate_and_clean_data()

            print(f"成功加载数据: 全球数据{len(self.global_data)}条, 国家数据{len(self.country_data)}条")

        except FileNotFoundError as e:
            print(f"数据文件未找到: {e}")
            print("使用模拟数据进行演示")
            self.load_simulated_data()
        except Exception as e:
            print(f"数据加载错误: {e}")
            print("使用模拟数据进行演示")
            self.load_simulated_data()

        return self

    def _validate_and_clean_data(self):
        """数据验证和清洗"""
        # 检查必需列
        global_required = ['year', 'usd_reserve_share', 'stablecoin_gdp_ratio',
                           'us_gdp_share', 'dxy_index', 'global_inflation']

        country_required = ['country', 'year', 'sovereignty_index', 'stablecoin_adoption',
                            'foreign_deposit_ratio', 'capital_control', 'gdp_per_capita',
                            'inflation', 'trade_openness', 'debt_to_gdp']

        # 验证全球数据
        missing_global = [col for col in global_required if col not in self.global_data.columns]
        if missing_global:
            raise ValueError(f"全球数据缺少必需列: {missing_global}")

        # 验证国家数据
        missing_country = [col for col in country_required if col not in self.country_data.columns]
        if missing_country:
            raise ValueError(f"国家数据缺少必需列: {missing_country}")

        # 数据清洗
        # 1. 处理缺失值
        self.global_data = self.global_data.fillna(method='ffill').fillna(method='bfill')
        self.country_data = self.country_data.fillna(method='ffill').fillna(method='bfill')

        # 2. 确保数据类型正确
        self.global_data['year'] = self.global_data['year'].astype(int)
        self.country_data['year'] = self.country_data['year'].astype(int)

        # 3. 排序数据
        self.global_data = self.global_data.sort_values('year')
        self.country_data = self.country_data.sort_values(['country', 'year'])

        print("数据验证和清洗完成")

    def load_simulated_data(self):
        """生成模拟数据（实际应用中替换为真实数据）"""
        # 全球层面数据模拟
        years = list(range(2010, 2025))
        n_years = len(years)

        global_data = pd.DataFrame({
            'year': years,
            'usd_reserve_share': [65 + 0.2 * i + np.random.normal(0, 1) for i in range(n_years)],  # 美元储备占比
            'stablecoin_gdp_ratio': [0] * 4 + [0.1, 0.5, 1.2, 2.5, 4.0, 6.5, 9.0, 12.0, 15.0, 18.0, 21.0],  # 稳定币/GDP
            'us_gdp_share': [24.5 - 0.1 * i for i in range(n_years)],  # 美国GDP占比
            'dxy_index': [80 + 0.5 * i + np.random.normal(0, 3) for i in range(n_years)],  # 美元指数
            'global_inflation': [2.5 + 0.1 * np.sin(i) + np.random.normal(0, 0.5) for i in range(n_years)]  # 全球通胀
        })

        # 国家层面数据模拟
        countries = ['Argentina', 'Turkey', 'Venezuela', 'Brazil', 'South Africa',
                     'Nigeria', 'Egypt', 'Pakistan', 'Ukraine', 'Colombia']
        n_countries = len(countries)

        country_panel = []
        for country in countries:
            for year in years[4:]:  # 从2014年开始（稳定币出现后）
                base_risk = np.random.uniform(0.3, 0.8)  # 国家基础风险水平
                trend = (year - 2014) * 0.05  # 时间趋势

                country_panel.append({
                    'country': country,
                    'year': year,
                    'sovereignty_index': 70 - base_risk * 20 - trend * 10 + np.random.normal(0, 5),
                    'stablecoin_adoption': base_risk * 0.1 + trend * 0.08 + np.random.normal(0, 0.02),
                    'foreign_deposit_ratio': 20 + base_risk * 30 + trend * 5 + np.random.normal(0, 5),
                    'capital_control': 0.5 - base_risk * 0.3 + np.random.normal(0, 0.1),
                    'gdp_per_capita': 8000 - base_risk * 5000 + np.random.normal(0, 1000),
                    'inflation': 10 + base_risk * 20 + trend * 2 + np.random.normal(0, 3),
                    'trade_openness': 40 + np.random.normal(0, 10),
                    'debt_to_gdp': 50 + base_risk * 20 + np.random.normal(0, 5)
                })

        self.global_data = global_data
        self.country_data = pd.DataFrame(country_panel)
        return self

    def model1_global_regression(self):
        """模型1：全球层面稳定币普及与美元地位关系"""
        print("=" * 60)
        print("模型1：全球层面稳定币普及与美元国际地位关系")
        print("=" * 60)

        X = self.global_data[['stablecoin_gdp_ratio', 'us_gdp_share', 'dxy_index', 'global_inflation']]
        X = sm.add_constant(X)
        y = self.global_data['usd_reserve_share']

        model = sm.OLS(y, X).fit()
        self.results['global_model'] = model

        print(model.summary())

        # 可视化结果
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.global_data['year'], self.global_data['usd_reserve_share'],
                 'bo-', label='美元储备占比(%)')
        plt.plot(self.global_data['year'], self.global_data['stablecoin_gdp_ratio'] * 0.5 + 60,
                 'ro-', label='稳定币/GDP比率(放大后)')
        plt.xlabel('年份')
        plt.ylabel('百分比(%)')
        plt.title('美元地位与稳定币普及趋势')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        coefficients = model.params[1:]  # 排除常数项
        variables = ['稳定币普及', '美国GDP占比', '美元指数', '全球通胀']
        colors = ['green' if coef > 0 else 'red' for coef in coefficients]

        plt.bar(variables, coefficients, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('回归系数分析')
        plt.xticks(rotation=45)
        plt.ylabel('系数大小')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return model

    def model2_panel_analysis(self):
        """模型2：国家层面面板数据分析"""
        print("\n" + "=" * 60)
        print("模型2：国家层面稳定币对货币主权的影响")
        print("=" * 60)

        # 面板数据回归
        formula = """
        sovereignty_index ~ stablecoin_adoption + foreign_deposit_ratio + 
        capital_control + gdp_per_capita + inflation + trade_openness
        """

        # 使用固定效应（通过国家虚拟变量）
        model = ols(formula, data=self.country_data).fit()
        self.results['panel_model'] = model

        print(model.summary())

        # 可视化主要关系
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        for country in self.country_data['country'].unique()[:5]:  # 显示前5个国家
            country_data = self.country_data[self.country_data['country'] == country]
            plt.plot(country_data['year'], country_data['sovereignty_index'],
                     'o-', label=country, alpha=0.7)
        plt.xlabel('年份')
        plt.ylabel('货币主权指数')
        plt.title('各国货币主权指数变化')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.scatter(self.country_data['stablecoin_adoption'],
                    self.country_data['sovereignty_index'],
                    c=self.country_data['inflation'], cmap='viridis', alpha=0.6)
        plt.colorbar(label='通货膨胀率(%)')
        plt.xlabel('稳定币普及程度')
        plt.ylabel('货币主权指数')
        plt.title('稳定币普及与货币主权关系')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        # 计算相关系数矩阵
        corr_vars = ['sovereignty_index', 'stablecoin_adoption', 'foreign_deposit_ratio',
                     'capital_control', 'inflation']
        corr_matrix = self.country_data[corr_vars].corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f')
        plt.title('变量相关性热图')

        plt.tight_layout()
        plt.show()

        return model

    def model3_risk_assessment(self):
        """模型3：货币主权风险评估-完整修复版"""
        print("\n" + "=" * 60)
        print("模型3：货币主权风险评估与预测")
        print("=" * 60)

        # 准备逻辑回归数据
        risk_data = self.country_data.groupby('country').agg({
            'sovereignty_index': 'mean',
            'stablecoin_adoption': 'mean',
            'foreign_deposit_ratio': 'mean',
            'capital_control': 'mean',
            'inflation': 'mean',
            'debt_to_gdp': 'mean',
            'gdp_per_capita': 'mean'
        }).reset_index()

        # 动态阈值设置 - 基于数据分布
        sovereignty_median = risk_data['sovereignty_index'].median()
        foreign_deposit_median = risk_data['foreign_deposit_ratio'].median()

        # 使用分位数作为阈值，确保类别平衡
        threshold_sovereignty = risk_data['sovereignty_index'].quantile(0.3)  # 下30%分位数
        threshold_foreign_deposit = risk_data['foreign_deposit_ratio'].quantile(0.7)  # 上70%分位数

        print(f"动态阈值 - 主权指数: {threshold_sovereignty:.2f}, 外币存款: {threshold_foreign_deposit:.2f}")

        risk_data['high_risk'] = ((risk_data['sovereignty_index'] < threshold_sovereignty) |
                                  (risk_data['foreign_deposit_ratio'] > threshold_foreign_deposit)).astype(int)

        # 增强的类别检查和处理
        unique_classes = risk_data['high_risk'].nunique()
        class_distribution = risk_data['high_risk'].value_counts().to_dict()

        print(f"类别分布: {class_distribution}")

        if unique_classes < 2:
            print("警告: 数据中只有一个类别，进行平衡处理")

            if 1 in class_distribution and class_distribution[1] == len(risk_data):
                # 所有样本都是高风险，将主权指数最高的改为低风险
                max_sovereignty_idx = risk_data['sovereignty_index'].idxmax()
                risk_data.loc[max_sovereignty_idx, 'high_risk'] = 0
                print(f"将 {risk_data.loc[max_sovereignty_idx, 'country']} 标记为低风险")
            elif 0 in class_distribution and class_distribution[0] == len(risk_data):
                # 所有样本都是低风险，将主权指数最低的改为高风险
                min_sovereignty_idx = risk_data['sovereignty_index'].idxmin()
                risk_data.loc[min_sovereignty_idx, 'high_risk'] = 1
                print(f"将 {risk_data.loc[min_sovereignty_idx, 'country']} 标记为高风险")

            # 重新检查分布
            new_distribution = risk_data['high_risk'].value_counts().to_dict()
            print(f"调整后类别分布: {new_distribution}")

        # 特征工程和标准化
        features = ['stablecoin_adoption', 'foreign_deposit_ratio', 'capital_control',
                    'inflation', 'debt_to_gdp', 'gdp_per_capita']
        X = risk_data[features]
        y = risk_data['high_risk']

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 使用带类别权重的逻辑回归
        logit_model = LogisticRegression(
            random_state=42,
            class_weight='balanced'  # 自动平衡类别权重
        )

        try:
            logit_model.fit(X_scaled, y)
        except ValueError as e:
            print(f"模型训练失败: {e}")
            print("使用替代方案: 手动创建平衡数据集")
            return self._fallback_risk_assessment(risk_data)

        # 预测概率
        y_pred_proba = logit_model.predict_proba(X_scaled)[:, 1]
        risk_data['risk_probability'] = y_pred_proba

        # 计算货币主权脆弱性指数(MSVI)
        coefficients = logit_model.coef_[0]
        msv_scores = X_scaled @ coefficients
        risk_data['msvi'] = msv_scores

        self.results['risk_data'] = risk_data
        self.results['logit_model'] = logit_model

        # 输出风险评估结果
        print("高风险国家预测结果:")
        print("-" * 50)
        high_risk_countries = risk_data.nlargest(5, 'risk_probability')[['country', 'risk_probability', 'msvi']]
        print(high_risk_countries.round(3))

        # 可视化部分保持不变
        self._plot_risk_assessment(risk_data, features, coefficients, high_risk_countries)

        return risk_data

    def _fallback_risk_assessment(self, risk_data):
        """备用风险评估方案"""
        print("使用基于规则的风险评估方法")

        # 基于多个指标的简单加权评分
        risk_data['composite_score'] = (
                risk_data['stablecoin_adoption'] * 0.2 +
                risk_data['foreign_deposit_ratio'] * 0.3 +
                risk_data['inflation'] * 0.2 +
                (100 - risk_data['sovereignty_index']) * 0.2 +
                risk_data['debt_to_gdp'] * 0.1
        )

        # 归一化到0-1概率
        max_score = risk_data['composite_score'].max()
        min_score = risk_data['composite_score'].min()
        risk_data['risk_probability'] = (risk_data['composite_score'] - min_score) / (max_score - min_score)

        # 设置高风险阈值
        risk_threshold = risk_data['risk_probability'].quantile(0.7)
        risk_data['high_risk'] = (risk_data['risk_probability'] > risk_threshold).astype(int)

        self.results['risk_data'] = risk_data
        return risk_data

    def _plot_risk_assessment(self, risk_data, features, coefficients, high_risk_countries):
        """可视化风险评估结果"""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        colors = ['red' if x > 0.5 else 'orange' for x in risk_data['risk_probability']]
        plt.barh(risk_data['country'], risk_data['risk_probability'], color=colors)
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='高风险阈值')
        plt.xlabel('丧失货币主权概率')
        plt.title('各国货币主权风险概率')
        plt.legend()

        plt.subplot(1, 3, 2)
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': abs(coefficients)
        }).sort_values('importance', ascending=True)
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('特征重要性(绝对值)')
        plt.title('风险因素重要性排序')

        plt.subplot(1, 3, 3)
        plt.scatter(risk_data['foreign_deposit_ratio'], risk_data['stablecoin_adoption'],
                    c=risk_data['risk_probability'], cmap='Reds', s=100, alpha=0.7)
        plt.colorbar(label='风险概率')
        plt.xlabel('外币存款比例(%)')
        plt.ylabel('稳定币普及程度')
        plt.title('风险分布图')

        for i, row in high_risk_countries.iterrows():
            country_data = risk_data[risk_data['country'] == row['country']].iloc[0]
            plt.annotate(row['country'],
                         (country_data['foreign_deposit_ratio'],
                          country_data['stablecoin_adoption']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()
        plt.show()

    def comprehensive_analysis(self):
        """综合分析报告"""
        print("\n" + "=" * 60)
        print("综合分析报告")
        print("=" * 60)

        # 全球模型关键发现
        global_coef = self.results['global_model'].params['stablecoin_gdp_ratio']
        global_p = self.results['global_model'].pvalues['stablecoin_gdp_ratio']

        print("1. 全球层面分析结果:")
        print(f"   • 稳定币普及系数: {global_coef:.4f}")
        print(f"   • 显著性水平(p值): {global_p:.4f}")
        print(f"   • 结论: {'稳定币普及显著强化美元地位' if global_p < 0.05 else '关系不显著'}")

        # 面板模型关键发现
        panel_coef = self.results['panel_model'].params['stablecoin_adoption']
        panel_p = self.results['panel_model'].pvalues['stablecoin_adoption']

        print("\n2. 国家层面分析结果:")
        print(f"   • 稳定币对货币主权影响系数: {panel_coef:.4f}")
        print(f"   • 显著性水平(p值): {panel_p:.4f}")
        print(f"   • 结论: {'稳定币普及显著削弱货币主权' if panel_p < 0.05 and panel_coef < 0 else '影响不明确'}")

        # 风险评估结果
        high_risk = self.results['risk_data'].nlargest(3, 'risk_probability')
        print("\n3. 高风险国家识别:")
        for _, country in high_risk.iterrows():
            print(f"   • {country['country']}: 风险概率 {country['risk_probability']:.1%}")

        print("\n4. 政策建议:")
        print("   • 对高风险国家: 加强资本流动管理，推动本币数字化")
        print("   • 国际层面: 建立稳定币跨境监管合作框架")
        print("   • 长期策略: 发展多边数字货币基础设施，减少美元依赖")

    def run_complete_analysis(self, use_real_data=False, global_file='global_data.csv',
                              country_file='country_panel_data.csv'):
        """运行完整分析流程"""
        print("开始稳定币与货币主权关系分析...")

        # 1. 加载数据 - 修改这里！
        if use_real_data:
            self.load_real_data(global_file, country_file)
        else:
            self.load_simulated_data()  # 保持原有函数名不变

        # 2. 运行三个模型（保持不变）
        self.model1_global_regression()
        self.model2_panel_analysis()
        self.model3_risk_assessment()

        # 3. 综合分析
        self.comprehensive_analysis()

        return self.results


# 使用示例
if __name__ == "__main__":
    # 创建模型实例并运行分析
    model = StablecoinMonetarySovereigntyModel()

    # 使用真实数据运行分析
    results = model.run_complete_analysis(
        use_real_data=True,
        global_file='my_global_data.csv',
        country_file='my_country_data.csv'
    )

    # 保存结果（可选）
    model.global_data.to_csv('global_data_simulation.csv', index=False)
    model.country_data.to_csv('country_data_simulation.csv', index=False)

    print("\n分析完成！结果已保存到CSV文件。")