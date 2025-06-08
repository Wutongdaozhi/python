import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb

class ModelBuilder:
    def build_model(self, data):
        """构建共享单车需求预测模型"""
        try:
            print("\n开始构建预测模型...")

            if data.empty:
                raise ValueError("数据为空，无法构建预测模型")

            # 特征工程
            feature_data = self._prepare_features(data)

            # 检查是否有足够数据
            if len(feature_data) < 50:
                print(f"警告：数据量不足 ({len(feature_data)}条)，模型可能表现不佳")

            # 划分训练集和测试集
            X, y = self._split_features_target(feature_data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练模型
            models = self._train_models(X_train_scaled, y_train)

            # 模型评估
            model_results = self._evaluate_models(models, X_test_scaled, y_test)

            print("模型构建完成")
            return model_results, models['LightGBM'], X.columns.tolist()

        except Exception as e:
            print(f"模型构建出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _prepare_features(self, data):
        """准备模型特征"""
        feature_data = data.copy()
        
        # 分类特征编码（包括天气特征）
        feature_data = pd.get_dummies(feature_data,
                                      columns=['preciptype', 'day_of_week', 'temp_bin', 'wind_bin'],
                                      dummy_na=True,
                                      drop_first=True)  # 避免多重共线性

        # 构建站点级预测特征
        station_features = feature_data.groupby(['start_station_id', 'date']).agg(
            ride_count=('start_station_id', 'count'),
            avg_temp=('temp', 'mean'),
            avg_windspeed=('windspeed', 'mean'),
            is_weekend=('is_weekend', 'first'),
            # 聚合天气分类特征
            **{col: ('{}_x'.format(col), 'first') for col in
               ['preciptype', 'temp_bin', 'wind_bin'] if '{}_x'.format(col) in feature_data.columns}
        ).reset_index()

        # 添加历史特征（前1天和前7天）
        station_features['lag_1_day'] = station_features.groupby('start_station_id')['ride_count'].shift(1)
        station_features['lag_7_day'] = station_features.groupby('start_station_id')['ride_count'].shift(7)

        # 删除含NaN的行
        station_features = station_features.dropna()

        # 确保所有特征为数值型
        station_features = station_features.select_dtypes(include=['number'])
        
        return station_features
    
    def _split_features_target(self, data):
        """划分特征和目标变量"""
        X = data.drop(['ride_count'], axis=1)
        y = data['ride_count']
        return X, y
    
    def _train_models(self, X_train, y_train):
        """训练多个预测模型"""
        # 1. 随机森林模型
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,  # 限制树深防止过拟合
            random_state=42
        )
        rf_model.fit(X_train, y_train)

        # 2. LightGBM模型
        lgb_model = lgb.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            random_state=42,
            importance_type='gain'
        )
        lgb_model.fit(X_train, y_train)

        return {
            '随机森林': rf_model,
            'LightGBM': lgb_model
        }
    
    def _evaluate_models(self, models, X_test, y_test):
        """评估模型性能"""
        results = {}
        for name, model in models.items():
            pred = model.predict(X_test)
            results[name] = {
                'R²': r2_score(y_test, pred),
                'MAE': mean_absolute_error(y_test, pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, pred))
            }
        return results    