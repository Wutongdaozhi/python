import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        """初始化数据处理器"""
        self.temp_bins = None
        self.temp_labels = ['极冷', '凉爽', '舒适', '炎热', '酷热']
        self.wind_bins = [0, 1.5, 3.3, 5.4, 7.9, np.inf]
        self.wind_labels = ['无风', '轻风', '微风', '和风', '大风']

    def load_and_preprocess(self, usage_file='cleaned_daily_rent_data.csv',
                            station_file='cleaned_station_list.csv',
                            weather_file='cleaned_weather.csv'):
        """加载并预处理共享单车使用数据"""
        try:
            print("正在加载数据...")

            # 读取数据文件
            daily_rent_data = pd.read_csv(usage_file, parse_dates=['started_at'])
            station_list = pd.read_csv(station_file)
            weather = pd.read_csv(weather_file, parse_dates=['datetime'])

            print(f"原始骑行数据量: {len(daily_rent_data)} 条记录")
            print(f"原始站点数据量: {len(station_list)} 个站点")
            print(f"原始天气数据量: {len(weather)} 条记录")

            # 数据校验：确保station_list包含station_id列
            if 'station_id' not in station_list.columns:
                if 'id' in station_list.columns:
                    station_list = station_list.rename(columns={'id': 'station_id'})
                    print("已将'id'列重命名为'station_id'")
                else:
                    raise ValueError("station_list.csv中缺少'station_id'或'id'列")

            # 数据校验：检查骑行数据时间完整性
            if not daily_rent_data['started_at'].isna().sum() == 0:
                print(f"骑行数据中 {daily_rent_data['started_at'].isna().sum()} 条记录时间缺失，将进行填充")
                daily_rent_data['started_at'] = daily_rent_data['started_at'].fillna(method='ffill').fillna(method='bfill')

            # 数据校验：检查站点ID完整性
            if daily_rent_data['start_station_id'].isna().sum() > 0:
                print(f"骑行数据中 {daily_rent_data['start_station_id'].isna().sum()} 条记录站点ID缺失，将删除")
                daily_rent_data = daily_rent_data.dropna(subset=['start_station_id'])

            # 解析duration为分钟数
            daily_rent_data['duration_minutes'] = daily_rent_data['duration'].apply(self._parse_duration)
            
            # 提取时间特征
            daily_rent_data = self._extract_time_features(daily_rent_data)

            # 天气数据增强处理
            print("正在处理天气数据...")
            weather = self._process_weather_data(weather)

            # 合并数据
            merged_data = self._merge_data(daily_rent_data, weather, station_list)

            # 数据校验：检查站点关联情况
            merged_data['station_name'] = merged_data['station_name'].fillna('未知站点')
            unknown_station_count = (merged_data['station_name'] == '未知站点').sum()
            if unknown_station_count > 0:
                print(f"警告：有 {unknown_station_count} 条记录关联到未知站点，需检查站点数据")

            # 诊断风速和温度数据
            self._diagnose_wind_and_temp(merged_data)

            print(f"数据预处理完成，最终数据量: {len(merged_data)} 条记录")

            # 数据断言
            assert len(merged_data) > 0, "预处理后数据为空，无法继续分析"
            assert merged_data['temp'].notna().sum() > 0, "温度数据全部缺失，无法分析"

            return merged_data, station_list

        except Exception as e:
            print(f"数据加载出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _parse_duration(self, duration_str):
        """解析时长字符串为分钟数"""
        if pd.isna(duration_str):
            return np.nan
        try:
            parts = str(duration_str).split()
            if 'days' in parts:
                days = int(parts[0])
                hms = parts[2].split(':')
            else:
                days = 0
                hms = parts[0].split(':')
            hours = int(hms[0])
            minutes = int(hms[1])
            return days * 1440 + hours * 60 + minutes
        except:
            return np.nan

    def _extract_time_features(self, data):
        """从时间戳提取时间特征"""
        data['hour'] = data['started_at'].dt.hour
        data['day_of_week'] = data['started_at'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['date'] = data['started_at'].dt.date
        data['date_hour'] = data['started_at'].dt.strftime('%Y-%m-%d %H')
        return data

    def _process_weather_data(self, weather):
        """处理天气数据，填充缺失值并进行特征工程"""
        # 检查天气数据的完整性
        weather_cols = ['temp', 'windspeed', 'preciptype']
        for col in weather_cols:
            if col not in weather.columns:
                print(f"警告：天气数据缺少{col}列，将使用默认值")
                if col == 'temp':
                    weather[col] = 25.0  # 默认温度
                elif col == 'windspeed':
                    weather[col] = 5.0  # 默认风速
                elif col == 'preciptype':
                    weather[col] = '无降水'  # 默认天气类型

        # 提取更多天气特征
        weather['temp'] = pd.to_numeric(weather['temp'], errors='coerce')
        weather['windspeed'] = pd.to_numeric(weather['windspeed'], errors='coerce')
        weather['date_hour'] = weather['datetime'].dt.strftime('%Y-%m-%d %H')

        # 风速单位转换（假设原单位为 km/h，转换为 m/s）
        weather['windspeed_m/s'] = weather['windspeed'] / 3.6

        # 使用国际标准蒲福风级分箱
        weather['wind_bin'] = pd.cut(
            weather['windspeed_m/s'], 
            bins=self.wind_bins, 
            labels=self.wind_labels, 
            include_lowest=True
        )

        return weather

    def _merge_data(self, daily_rent_data, weather, station_list):
        """合并骑行数据、天气数据和站点数据"""
        # 合并天气数据
        merged_data = pd.merge(daily_rent_data, weather, on='date_hour', how='left')

        # 数据校验：检查时间范围覆盖
        min_rent_time = daily_rent_data['started_at'].min()
        max_rent_time = daily_rent_data['started_at'].max()
        min_weather_time = weather['datetime'].min()
        max_weather_time = weather['datetime'].max()

        print(f"\n骑行数据时间范围: {min_rent_time} 到 {max_rent_time}")
        print(f"天气数据时间范围: {min_weather_time} 到 {max_weather_time}")

        if min_rent_time < min_weather_time or max_rent_time > max_weather_time:
            print(f"警告：天气数据时间范围未完全覆盖骑行数据，可能导致匹配不全")

        # 检查天气数据匹配率
        matched_count = merged_data['windspeed'].notna().sum()
        total_count = len(merged_data)
        match_percent = matched_count / total_count * 100
        print(f"\n天气数据匹配率: {matched_count}/{total_count} ({match_percent:.2f}%)")

        # 填充天气缺失值
        merged_data['preciptype'] = merged_data['preciptype'].fillna('无降水')

        # 按站点、小时和天气类型分组填充温度和风速
        # 首先尝试按站点填充
        station_wind_avg = merged_data.groupby('start_station_id')['windspeed'].transform('mean')
        merged_data['windspeed'] = merged_data['windspeed'].fillna(station_wind_avg)

        # 然后按小时和天气类型填充
        wind_fill_values = merged_data.groupby(['hour', 'preciptype'])['windspeed'].transform('mean')
        merged_data['windspeed'] = merged_data['windspeed'].fillna(wind_fill_values)

        # 温度填充 - 增强逻辑
        temp_fill = merged_data.groupby(['date', 'hour'])['temp'].transform('mean')
        merged_data['temp'] = merged_data['temp'].fillna(temp_fill).fillna(merged_data['temp'].mean())

        # 如果分组填充后仍有缺失值，使用全局均值
        merged_data['windspeed'] = merged_data['windspeed'].fillna(merged_data['windspeed'].mean())

        # 重新计算风速分箱（确保所有记录都有分箱）
        merged_data['windspeed_m/s'] = merged_data['windspeed'] / 3.6
        merged_data['wind_bin'] = pd.cut(
            merged_data['windspeed_m/s'],
            bins=self.wind_bins,
            labels=self.wind_labels,
            include_lowest=True
        )

        # 动态温度分箱
        self._calculate_temp_bins(merged_data)
        merged_data['temp_bin'] = pd.cut(
            merged_data['temp'],
            bins=self.temp_bins,
            labels=self.temp_labels[:len(self.temp_bins) - 1],  # 动态匹配标签数量
            right=False  # 左闭右开区间
        )

        # 添加站点名称信息
        station_list = station_list.rename(columns={'station_id': 'start_station_id'})
        merged_data = pd.merge(
            merged_data,
            station_list[['start_station_id', 'station_name']],
            on='start_station_id',
            how='left'
        )

        return merged_data

    def _calculate_temp_bins(self, data):
        """计算温度分箱边界"""
        print("\n正在优化温度分箱...")
        # 使用合并后的数据温度范围
        temp_min = data['temp'].min()
        temp_max = data['temp'].max()

        # 动态调整分箱参数
        temp_range = temp_max - temp_min
        if temp_range < 15:  # 最小温度范围设为15度
            buffer = (15 - temp_range) / 2
            temp_min -= buffer
            temp_max += buffer

        # 创建分箱边界（每5度一个区间）
        num_bins = int((temp_max - temp_min) // 5) + 1
        self.temp_bins = np.linspace(temp_min, temp_max, num=num_bins + 1)

        # 确保至少5个区间
        if len(self.temp_bins) < 5:
            self.temp_bins = np.linspace(temp_min - 5, temp_max + 5, 6)

    def _diagnose_wind_and_temp(self, data):
        """诊断风速和温度数据分布"""
        print("\n合并后风速分布详情:")
        print(data[['windspeed', 'windspeed_m/s', 'wind_bin']].describe())
        print("风速分箱分布:")
        print(data['wind_bin'].value_counts(dropna=False))

        print("\n合并后温度分布详情:")
        print(data['temp'].describe())
        print("温度分箱分布:")
        print(data['temp_bin'].value_counts(dropna=False))    