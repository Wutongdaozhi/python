import pandas as pd

class DataAnalyzer:
    def perform_analysis(self, data, stations):
        """执行共享单车数据的时空特征分析"""
        try:
            print("\n开始时空特征分析...")

            if data.empty:
                raise ValueError("数据为空，无法进行时空特征分析")

            # 1. 站点热度分析
            station_activity = self._analyze_station_activity(data)

            # 2. 时间分布分析
            hourly_distribution = self._analyze_hourly_distribution(data)

            # 3. 天气影响分析
            weather_impact = self._analyze_weather_impact(data)

            # 4. 温度影响分析
            temp_impact = self._analyze_temp_impact(data)

            # 5. 风速影响分析
            wind_impact = self._analyze_wind_impact(data)

            print("时空特征分析完成")
            
            return {
                'station_activity': station_activity,
                'hourly_distribution': hourly_distribution,
                'weather_impact': weather_impact,
                'temp_impact': temp_impact,
                'wind_impact': wind_impact
            }
        except Exception as e:
            print(f"时空特征分析出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _analyze_station_activity(self, data):
        """分析站点活动情况"""
        station_activity = data.groupby(['start_station_id', 'station_name']).agg(
            ride_count=('start_station_id', 'count')
        ).reset_index()
        return station_activity

    def _analyze_hourly_distribution(self, data):
        """分析骑行次数的小时分布"""
        hourly_distribution = data['hour'].value_counts().sort_index()
        
        # 确保覆盖完整24小时
        all_hours = pd.Series([0] * 24, index=range(24))
        hourly_distribution = all_hours.add(hourly_distribution, fill_value=0)
        
        return hourly_distribution

    def _analyze_weather_impact(self, data):
        """分析天气类型对骑行的影响"""
        weather_impact = data.groupby('preciptype').agg(
            ride_count=('start_station_id', 'count')
        ).reset_index()
        return weather_impact

    def _analyze_temp_impact(self, data):
        """分析温度对骑行的影响"""
        temp_impact = data.groupby('temp_bin').agg(
            ride_count=('start_station_id', 'count')
        ).reset_index()
        
        # 数据校验：打印温度区间骑行次数
        print("\n温度区间骑行次数统计：")
        print(temp_impact)
        
        return temp_impact

    def _analyze_wind_impact(self, data):
        """分析风速对骑行的影响"""
        wind_impact = data.groupby('wind_bin').agg(
            ride_count=('start_station_id', 'count'),
            avg_windspeed=('windspeed_m/s', 'mean')
        ).reset_index()
        
        # 数据校验：打印风速区间骑行次数
        print("\n风速区间骑行次数统计：")
        print(wind_impact)
        
        # 风速分布诊断
        print("\n风速分布详情:")
        print(data[['windspeed', 'windspeed_m/s', 'wind_bin']].describe())
        print("风速分箱分布:")
        print(data['wind_bin'].value_counts(dropna=False))
        
        return wind_impact    