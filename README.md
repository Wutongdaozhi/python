# python数据分析项目----基于华盛顿特区共享单车数据的用户使用模式分析与优化建议

## 一、项目概述

本项目基于华盛顿特区共享单车骑行数据、站点信息及天气数据，通过数据预处理、时空特征分析、可视化展示及机器学习建模，深入挖掘用户使用模式，分析环境因素（如天气、温度、风速）对骑行行为的影响，并提供运营优化建议。项目代码结构清晰，包含数据处理、分析、可视化及建模模块，可复现完整的数据分析流程。

## 二、环境配置

### 1. 依赖库

```python
pandas          # 数据处理
numpy           # 数值计算
matplotlib      # 基础可视化
seaborn         # 统计可视化
scikit-learn    # 机器学习模型
lightgbm        # 梯度提升模型
```

### 2. 环境搭建

```bash
# 创建虚拟环境（可选）
python -m venv env
source env/bin/activate  # Linux/macOS
env\Scripts\activate     # Windows

# 安装依赖
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
```

## 三、目录结构

```plaintext
project/
├─ data/                # 原始数据及处理结果
│  ├─ cleaned_daily_rent_data.csv  # 骑行数据
│  ├─ cleaned_station_list.csv    # 站点数据
│  └─ cleaned_weather.csv         # 天气数据
├─ src/                 # 代码模块
│  ├─ data_processor.py  # 数据清洗与预处理
│  ├─ data_analyzer.py    # 特征分析与统计
│  ├─ visualizer.py      # 可视化生成
│  └─ model_builder.py   # 模型训练与评估
├─ visualizations/      # 生成的图表
│  ├─ hourly_distribution.png   # 小时分布
│  ├─ weather_type_impact.png   # 天气影响
│  └─ ...其他图表
├─ main.py             # 主程序入口
└─ README.md           # 项目说明
```

## 四、数据说明

| 数据集   | 字段说明                                                     | 记录数   |
| -------- | ------------------------------------------------------------ | -------- |
| 骑行数据 | started_at（时间）、start_station_id（站点 ID）、duration（时长）等 | 1,498 条 |
| 站点数据 | station_id（站点 ID）、station_name（站点名称）、经纬度等    | 916 个   |
| 天气数据 | datetime（时间）、temp（温度）、windspeed（风速）、preciptype（降水类型）等 | 245 条   |

## 五、使用说明

### 1. 数据预处理

运行`data_processor.py`清洗原始数据：

- 处理骑行时间缺失值，解析时长为分钟数
- 提取时间特征（小时、星期、周末标识）
- 天气数据分箱（温度、风速等级）
- 多表合并（骑行 + 站点 + 天气）

```python
# 执行数据处理
from src.data_processor import DataProcessor
processor = DataProcessor()
data, stations = processor.load_and_preprocess()
```

### 2. 时空特征分析

运行`data_analyzer.py`分析数据：

- 站点热度统计（骑行次数排名）
- 时间分布（小时、工作日 / 周末差异）
- 天气影响分析（降水、温度、风速与骑行次数的关联）

```python
from src.data_analyzer import DataAnalyzer
analyzer = DataAnalyzer()
analysis_results = analyzer.perform_analysis(data, stations)
```

### 3. 数据可视化

运行`visualizer.py`生成图表：

- 小时分布柱状图
- 天气类型影响柱状图
- 温度 / 风速区间骑行次数对比
- 热门站点排名图

```python
from src.visualizer import Visualizer
visualizer = Visualizer()
visualizer.generate_visualizations(analysis_results, data)
```

### 4. 模型训练与评估

运行`model_builder.py`构建预测模型：

- 特征工程（独热编码、历史骑行数据滞后特征）
- 对比随机森林与 LightGBM 算法
- 输出模型评估指标（R²、MAE、RMSE）及特征重要性

```python
from src.model_builder import ModelBuilder
builder = ModelBuilder()
model_results, best_model, feature_names = builder.build_model(data)
```

### 5. 主程序运行

直接执行`main.py`一站式运行全流程：

```bash
python main.py
```

## 六、关键输出

1. **可视化图表**：

   - `visualizations/hourly_distribution.png`：骑行次数小时分布
   - `visualizations/weather_type_impact.png`：不同天气类型骑行次数对比
   - `visualizations/top_stations.png`：热门站点排名

2. **模型结果**：

   - 评估指标：

     ```plaintext
     === 模型评估结果 ===
     随机森林:
       R²: -0.1936
       MAE: 1.1292
       RMSE: 1.7345
     LightGBM:
       R²: -0.2413
       MAE: 1.1192
       RMSE: 1.7688
     ```

   - 特征重要性：站点 ID、滞后 1 天骑行次数、风速、温度等

3. **优化建议**：

   - 高峰时段向热门站点增加车辆投放
   - 降水天气在交通枢纽增设临时停放点
   - 基于天气预测动态调整维护计划

## 七、注意事项

1. 可视化字体：若中文图表显示乱码，需手动设置 matplotlib 字体（见`visualizer.py`注释）
2. 模型优化：小样本数据导致 R² 为负，可尝试增加数据量或引入正则化策略
3. 扩展方向：可添加用户画像、地理围栏数据，或尝试 LSTM 等时序模型



通过以上步骤，可复现从数据处理到模型分析的完整流程，结合可视化结果与模型输出，为共享单车运营提供数据驱动的决策支持。
