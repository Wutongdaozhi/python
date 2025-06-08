import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class Visualizer:
    def __init__(self):
        """初始化可视化器"""
        # 创建输出目录
        os.makedirs('visualizations', exist_ok=True)

    def generate_visualizations(self, analysis_results, data):
        """生成共享单车使用分析的可视化图表"""
        try:
            print("\n开始生成可视化图表...")

            # 解构分析结果
            station_activity = analysis_results['station_activity']
            hourly_distribution = analysis_results['hourly_distribution']
            weather_impact = analysis_results['weather_impact']
            temp_impact = analysis_results['temp_impact']
            wind_impact = analysis_results['wind_impact']

            # 1. 时间分布柱状图
            self._plot_hourly_distribution(hourly_distribution)

            # 2. 天气类型影响柱状图
            self._plot_weather_impact(weather_impact)

            # 3. 温度影响柱状图
            self._plot_temp_impact(temp_impact)

            # 4. 风速影响柱状图
            self._plot_wind_impact(wind_impact)

            # 5. 热门站点柱状图
            self._plot_top_stations(station_activity)

            # 6. 按星期分布的骑行次数柱状图
            self._plot_weekday_distribution(data)

            print("\n可视化图表已保存到 'visualizations' 目录")

        except Exception as e:
            print(f"可视化出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _plot_hourly_distribution(self, hourly_distribution):
        """绘制骑行次数的小时分布柱状图"""
        plt.figure(figsize=(14, 7))
        ax = hourly_distribution.plot(kind='bar', color='skyblue')
        plt.title('骑行次数的小时分布', fontsize=16)
        plt.xlabel('小时', fontsize=14)
        plt.ylabel('骑行次数', fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数据标签
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        fontsize=10, color='black',
                        xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig('visualizations/hourly_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 小时分布图表已生成")

    def _plot_weather_impact(self, weather_impact):
        """绘制不同天气类型下的骑行次数柱状图"""
        plt.figure(figsize=(14, 7))
        ax = weather_impact.set_index('preciptype')['ride_count'].plot(kind='bar', color='green')
        plt.title('不同天气类型下的骑行次数', fontsize=16)
        plt.xlabel('天气类型', fontsize=14)
        plt.ylabel('骑行次数', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数据标签
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        fontsize=10, color='black',
                        xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig('visualizations/weather_type_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 天气类型影响图表已生成")

    def _plot_temp_impact(self, temp_impact):
        """绘制不同温度区间的骑行次数柱状图"""
        plt.figure(figsize=(14, 7))
        ax = temp_impact.set_index('temp_bin')['ride_count'].plot(kind='bar', color='orange')
        plt.title('不同温度区间的骑行次数', fontsize=16)
        plt.xlabel('温度区间', fontsize=14)
        plt.ylabel('骑行次数', fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数据标签
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        fontsize=10, color='black',
                        xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig('visualizations/temperature_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 温度影响图表已生成")

    def _plot_wind_impact(self, wind_impact):
        """绘制不同风速区间的骑行次数柱状图"""
        plt.figure(figsize=(14, 7))
        ax = sns.barplot(
            x='wind_bin',
            y='ride_count',
            data=wind_impact,
            palette="Blues_d",
            order=wind_impact['wind_bin']  # 确保顺序正确
        )
        plt.title('不同风速区间的骑行次数', fontsize=16)
        plt.xlabel('风速区间', fontsize=14)
        plt.ylabel('骑行次数', fontsize=14)
        plt.xticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加精确数值标签
        for i, row in wind_impact.iterrows():
            plt.text(
                i,
                row['ride_count'] + 0.05 * max(wind_impact['ride_count']),
                f"{int(row['ride_count'])}",
                ha='center',
                va='bottom',
                fontsize=12
            )

            # 在柱子上方添加平均风速信息
            plt.text(
                i,
                row['ride_count'] + 0.01 * max(wind_impact['ride_count']),
                f"平均风速: {row['avg_windspeed']:.1f} m/s",
                ha='center',
                va='bottom',
                fontsize=9,
                color='gray'
            )

        plt.tight_layout()
        plt.savefig('visualizations/wind_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 风速影响图表已生成")

    def _plot_top_stations(self, station_activity):
        """绘制热门站点骑行次数柱状图"""
        top_stations = station_activity.sort_values('ride_count', ascending=False).head(10)
        plt.figure(figsize=(14, 7))
        ax = plt.bar(top_stations['station_name'], top_stations['ride_count'], color='steelblue')
        plt.title('热门站点骑行次数', fontsize=16)
        plt.xlabel('站点名称', fontsize=14)
        plt.ylabel('骑行次数', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数据标签
        for i, rect in enumerate(ax):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('visualizations/top_stations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 热门站点图表已生成")

    def _plot_weekday_distribution(self, data):
        """绘制按星期分布的骑行次数柱状图"""
        weekday_mapping = {
            0: '星期一',
            1: '星期二',
            2: '星期三',
            3: '星期四',
            4: '星期五',
            5: '星期六',
            6: '星期日'
        }
        weekday_distribution = data['day_of_week'].value_counts().sort_index()
        weekday_distribution.index = weekday_distribution.index.map(weekday_mapping)

        plt.figure(figsize=(14, 7))
        ax = weekday_distribution.plot(kind='bar', color='purple')
        plt.title('按星期分布的骑行次数', fontsize=16)
        plt.xlabel('星期', fontsize=14)
        plt.ylabel('骑行次数', fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数据标签
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        fontsize=10, color='black',
                        xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig('visualizations/weekday_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 按星期分布的骑行次数图表已生成")