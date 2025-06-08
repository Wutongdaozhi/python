import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import warnings
import os
from data_processor import DataProcessor
from data_analyzer import DataAnalyzer
from visualizer import Visualizer
from model_builder import ModelBuilder

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """共享单车数据分析主函数"""
    try:
        print("=" * 50)
        print("欢迎使用共享单车数据分析系统")
        print("=" * 50)

        # 数据处理
        data_processor = DataProcessor()
        data, stations = data_processor.load_and_preprocess()

        # 数据分析
        data_analyzer = DataAnalyzer()
        analysis_results = data_analyzer.perform_analysis(data, stations)

        # 数据可视化
        visualizer = Visualizer()
        visualizer.generate_visualizations(analysis_results, data)

        # 模型构建
        model_builder = ModelBuilder()
        model_results, best_model, feature_names = model_builder.build_model(data)

        # 输出模型评估结果
        print("\n" + "=" * 50)
        print("=== 模型评估结果 ===")
        if model_results:
            for model_name, metrics in model_results.items():
                print(f"\n{model_name}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")

            # 输出特征重要性
            if best_model and feature_names:
                print("\n特征重要性:")
                feature_importance = pd.DataFrame({
                    '特征': feature_names,
                    '重要性': best_model.feature_importances_
                }).sort_values('重要性', ascending=False)
                print(feature_importance)
        else:
            print("模型评估失败")

        print("\n所有任务完成！")

    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()