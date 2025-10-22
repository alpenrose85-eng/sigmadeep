import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import json
import io
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="Анализатор сигма-фазы",
    page_icon="🔬",
    layout="wide"
)

# Заголовок приложения
st.title("🔬 Анализатор кинетики образования сигма-фазы в стали 12Х18Н12Т")
st.markdown("""
### Определение температурной зависимости по содержанию сигма-фазы, времени эксплуатации и номеру зерна
""")

class GrainSizeConverter:
    """Класс для преобразования номера зерна в физические параметры по ГОСТ 5639-82"""
    
    # Данные из ГОСТ 5639-82 (на основе предоставленной таблицы)
    GRAIN_DATA = {
        -3: {'area_mm2': 1.000, 'diameter_mm': 1.000, 'conditional_diameter_mm': 0.875, 'grains_per_mm2': 1.0},
        -2: {'area_mm2': 0.500, 'diameter_mm': 0.707, 'conditional_diameter_mm': 0.650, 'grains_per_mm2': 2.8},
        -1: {'area_mm2': 0.250, 'diameter_mm': 0.500, 'conditional_diameter_mm': 0.444, 'grains_per_mm2': 8.0},
        0:  {'area_mm2': 0.125, 'diameter_mm': 0.353, 'conditional_diameter_mm': 0.313, 'grains_per_mm2': 22.6},
        1:  {'area_mm2': 0.0625, 'diameter_mm': 0.250, 'conditional_diameter_mm': 0.222, 'grains_per_mm2': 64.0},
        2:  {'area_mm2': 0.0312, 'diameter_mm': 0.177, 'conditional_diameter_mm': 0.157, 'grains_per_mm2': 181.0},
        3:  {'area_mm2': 0.0156, 'diameter_mm': 0.125, 'conditional_diameter_mm': 0.111, 'grains_per_mm2': 512.0},
        4:  {'area_mm2': 0.00781, 'diameter_mm': 0.088, 'conditional_diameter_mm': 0.0783, 'grains_per_mm2': 1448.0},
        5:  {'area_mm2': 0.00390, 'diameter_mm': 0.062, 'conditional_diameter_mm': 0.0553, 'grains_per_mm2': 4096.0},
        6:  {'area_mm2': 0.00195, 'diameter_mm': 0.044, 'conditional_diameter_mm': 0.0391, 'grains_per_mm2': 11585.0},
        7:  {'area_mm2': 0.00098, 'diameter_mm': 0.031, 'conditional_diameter_mm': 0.0267, 'grains_per_mm2': 32768.0},
        8:  {'area_mm2': 0.00049, 'diameter_mm': 0.022, 'conditional_diameter_mm': 0.0196, 'grains_per_mm2': 92682.0},
        9:  {'area_mm2': 0.000244, 'diameter_mm': 0.015, 'conditional_diameter_mm': 0.0138, 'grains_per_mm2': 262144.0},
        10: {'area_mm2': 0.000122, 'diameter_mm': 0.011, 'conditional_diameter_mm': 0.0099, 'grains_per_mm2': 741485.0},
        11: {'area_mm2': 0.000061, 'diameter_mm': 0.0079, 'conditional_diameter_mm': 0.0069, 'grains_per_mm2': 2097152.0},
        12: {'area_mm2': 0.000030, 'diameter_mm': 0.0056, 'conditional_diameter_mm': 0.0049, 'grains_per_mm2': 5931008.0},
        13: {'area_mm2': 0.000015, 'diameter_mm': 0.0039, 'conditional_diameter_mm': 0.0032, 'grains_per_mm2': 16777216.0},
        14: {'area_mm2': 0.000008, 'diameter_mm': 0.0027, 'conditional_diameter_mm': 0.0027, 'grains_per_mm2': 47449064.0}
    }
    
    @classmethod
    def grain_number_to_area(cls, grain_number):
        """Преобразование номера зерна в среднюю площадь сечения (мм²)"""
        data = cls.GRAIN_DATA.get(grain_number)
        if data:
            return data['area_mm2']
        else:
            # Интерполяция для промежуточных значений
            numbers = sorted(cls.GRAIN_DATA.keys())
            if grain_number < numbers[0]:
                return cls.GRAIN_DATA[numbers[0]]['area_mm2']
            elif grain_number > numbers[-1]:
                return cls.GRAIN_DATA[numbers[-1]]['area_mm2']
            else:
                # Находим ближайшие известные значения
                lower = max([n for n in numbers if n <= grain_number])
                upper = min([n for n in numbers if n >= grain_number])
                if lower == upper:
                    return cls.GRAIN_DATA[lower]['area_mm2']
                # Линейная интерполяция в логарифмической шкале
                log_area_lower = np.log(cls.GRAIN_DATA[lower]['area_mm2'])
                log_area_upper = np.log(cls.GRAIN_DATA[upper]['area_mm2'])
                fraction = (grain_number - lower) / (upper - lower)
                log_area = log_area_lower + fraction * (log_area_upper - log_area_lower)
                return np.exp(log_area)
    
    @classmethod
    def grain_number_to_diameter(cls, grain_number, use_conditional=True):
        """Преобразование номера зерна в диаметр (мм)"""
        data = cls.GRAIN_DATA.get(grain_number)
        if data:
            return data['conditional_diameter_mm'] if use_conditional else data['diameter_mm']
        else:
            # Интерполяция для промежуточных значений
            numbers = sorted(cls.GRAIN_DATA.keys())
            if grain_number < numbers[0]:
                return cls.GRAIN_DATA[numbers[0]]['conditional_diameter_mm'] if use_conditional else cls.GRAIN_DATA[numbers[0]]['diameter_mm']
            elif grain_number > numbers[-1]:
                return cls.GRAIN_DATA[numbers[-1]]['conditional_diameter_mm'] if use_conditional else cls.GRAIN_DATA[numbers[-1]]['diameter_mm']
            else:
                lower = max([n for n in numbers if n <= grain_number])
                upper = min([n for n in numbers if n >= grain_number])
                if lower == upper:
                    return cls.GRAIN_DATA[lower]['conditional_diameter_mm'] if use_conditional else cls.GRAIN_DATA[lower]['diameter_mm']
                # Линейная интерполяция
                diam_lower = cls.GRAIN_DATA[lower]['conditional_diameter_mm'] if use_conditional else cls.GRAIN_DATA[lower]['diameter_mm']
                diam_upper = cls.GRAIN_DATA[upper]['conditional_diameter_mm'] if use_conditional else cls.GRAIN_DATA[upper]['diameter_mm']
                fraction = (grain_number - lower) / (upper - lower)
                return diam_lower + fraction * (diam_upper - diam_lower)
    
    @classmethod
    def grain_number_to_grains_per_mm2(cls, grain_number):
        """Преобразование номера зерна в количество зерен на 1 мм²"""
        data = cls.GRAIN_DATA.get(grain_number)
        if data:
            return data['grains_per_mm2']
        else:
            # Интерполяция для промежуточных значений
            numbers = sorted(cls.GRAIN_DATA.keys())
            if grain_number < numbers[0]:
                return cls.GRAIN_DATA[numbers[0]]['grains_per_mm2']
            elif grain_number > numbers[-1]:
                return cls.GRAIN_DATA[numbers[-1]]['grains_per_mm2']
            else:
                lower = max([n for n in numbers if n <= grain_number])
                upper = min([n for n in numbers if n >= grain_number])
                if lower == upper:
                    return cls.GRAIN_DATA[lower]['grains_per_mm2']
                # Линейная интерполяция в логарифмической шкале
                log_count_lower = np.log(cls.GRAIN_DATA[lower]['grains_per_mm2'])
                log_count_upper = np.log(cls.GRAIN_DATA[upper]['grains_per_mm2'])
                fraction = (grain_number - lower) / (upper - lower)
                log_count = log_count_lower + fraction * (log_count_upper - log_count_lower)
                return np.exp(log_count)
    
    @classmethod
    def calculate_grain_boundary_density(cls, grain_number):
        """
        Расчет плотности границ зерен (мм²/мм³)
        Используем условный диаметр из ГОСТ
        """
        d = cls.grain_number_to_diameter(grain_number, use_conditional=True)  # мм
        
        # Для сферических зерен: Sv = 3/R = 6/D
        Sv = 3.0 / (d / 2.0)  # мм²/мм³
        
        return Sv
    
    @classmethod
    def calculate_effective_surface_area(cls, grain_number):
        """
        Расчет эффективной площади поверхности для зарождения сигма-фазы
        Учитывает как границы зерен, так и плотность зерен
        """
        # Плотность границ зерен
        boundary_density = cls.calculate_grain_boundary_density(grain_number)
        
        # Количество зерен на единицу объема (приблизительно)
        grains_per_mm2 = cls.grain_number_to_grains_per_mm2(grain_number)
        Nv = grains_per_mm2 ** (3/2)  # Преобразование 2D -> 3D
        
        # Эффективная площадь (комбинация границ и объема)
        effective_area = boundary_density * (1 + 0.1 * np.log(Nv + 1))
        
        return effective_area
    
    @classmethod
    def calculate_activation_energy_factor(cls, grain_number):
        """
        Коэффициент влияния размера зерна на энергию активации
        Учитывает реальные геометрические параметры из ГОСТ
        """
        # Нормализуем относительно номера зерна 5 (базовый)
        ref_grain = 5
        Sv_ref = cls.calculate_effective_surface_area(ref_grain)
        Sv_current = cls.calculate_effective_surface_area(grain_number)
        
        return Sv_current / Sv_ref
    
    @classmethod
    def get_grain_info_table(cls):
        """Получить таблицу с информацией о всех размерах зерен"""
        grain_numbers = sorted(cls.GRAIN_DATA.keys())
        table_data = []
        
        for gn in grain_numbers:
            data = cls.GRAIN_DATA[gn]
            boundary_density = cls.calculate_grain_boundary_density(gn)
            effective_area = cls.calculate_effective_surface_area(gn)
            activation_factor = cls.calculate_activation_energy_factor(gn)
            
            table_data.append({
                'G': gn,
                'Площадь_мм2': data['area_mm2'],
                'Диаметр_мм': data['conditional_diameter_mm'],
                'Зерен_на_мм2': data['grains_per_mm2'],
                'Плотность_границ': boundary_density,
                'Эффективная_площадь': effective_area,
                'Коэффициент_активации': activation_factor
            })
        
        return pd.DataFrame(table_data)

class OutlierDetector:
    """Класс для обнаружения выбросов в экспериментальных данных"""
    
    @staticmethod
    def detect_iqr(data, multiplier=1.5):
        """Метод межквартильного размаха (IQR)"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        return outliers, clean_data
    
    @staticmethod
    def detect_isolation_forest(features, contamination=0.1):
        """Isolation Forest для многомерного обнаружения выбросов"""
        clf = IsolationForest(contamination=contamination, random_state=42)
        labels = clf.fit_predict(features)
        return labels

# Модифицированная модель с учетом плотности границ зерен
def sigma_phase_model_advanced(params, G, T, t):
    """
    Усовершенствованная модель с учетом плотности границ зерен
    """
    K0, a, b, n, T_sigma_min, T_sigma_max, alpha = params
    R = 8.314  # Универсальная газовая постоянная
    
    # Температурные ограничения
    T_min = T_sigma_min + 273.15
    T_max = T_sigma_max + 273.15
    
    # Эффективная температура с учетом ограничений
    T_eff = np.where(T < T_min, T_min, T)
    T_eff = np.where(T_eff > T_max, T_max, T_eff)
    
    # Температурный фактор
    temp_factor = 1 / (1 + np.exp(-0.1 * (T - (T_min + 50)))) * 1 / (1 + np.exp(0.1 * (T - (T_max - 50))))
    
    # Базовая энергия активации
    Q_base = a + b * G
    
    # Влияние плотности границ зерен
    grain_boundary_factor = np.array([GrainSizeConverter.calculate_activation_energy_factor(g) for g in G])
    Q_effective = Q_base * (1 + alpha * (grain_boundary_factor - 1))
    
    K = K0 * np.exp(-Q_effective / (R * T_eff)) * temp_factor
    
    sigma = 1 - np.exp(-K * (t ** n))
    return sigma

class SigmaPhaseAnalyzer:
    def __init__(self):
        self.params = None
        self.R2 = None
        self.rmse = None
        self.outlier_info = None
        self.original_data = None
        self.clean_data = None
        self.model_version = "2.0"
        self.creation_date = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()
        self.use_advanced_model = True
        
    def detect_outliers(self, data, method='iqr', contamination=0.1):
        """Обнаружение выбросов в данных"""
        # Преобразуем температуру в Кельвины для анализа
        T_kelvin = data['T'] + 273.15
        features = np.column_stack([data['G'].values, T_kelvin.values, data['t'].values, data['f_exp (%)'].values])
        
        if method == 'iqr':
            outlier_flags = np.zeros(len(data), dtype=bool)
            
            for i, col in enumerate(['f_exp (%)', 't', 'T']):
                if col == 'T':
                    values = T_kelvin.values
                else:
                    values = data[col].values
                    
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = (values < lower_bound) | (values > upper_bound)
                outlier_flags = outlier_flags | col_outliers
            
        elif method == 'isolation_forest':
            labels = OutlierDetector.detect_isolation_forest(features, contamination)
            outlier_flags = labels == -1
        
        elif method == 'residual':
            return None, data
        
        outlier_data = data[outlier_flags]
        clean_data = data[~outlier_flags]
        
        return outlier_data, clean_data
    
    def fit_model(self, data, remove_outliers=True, outlier_method='iqr', contamination=0.1):
        """Подгонка модели с опцией удаления выбросов"""
        try:
            self.last_modified = datetime.now().isoformat()
            self.original_data = data.copy()
            
            if remove_outliers:
                outlier_data, clean_data = self.detect_outliers(data, outlier_method, contamination)
                self.clean_data = clean_data
                self.outlier_info = {
                    'outlier_data': outlier_data,
                    'method': outlier_method,
                    'contamination': contamination,
                    'outlier_count': len(outlier_data) if outlier_data is not None else 0,
                    'total_count': len(data)
                }
            else:
                self.clean_data = data
                self.outlier_info = {
                    'outlier_data': None,
                    'method': 'none',
                    'outlier_count': 0,
                    'total_count': len(data)
                }
            
            # Подготовка данных для подгонки
            G = self.clean_data['G'].values
            T_celsius = self.clean_data['T'].values
            T_kelvin = T_celsius + 273.15  # Конвертация в Кельвины
            t = self.clean_data['t'].values
            sigma_exp = self.clean_data['f_exp (%)'].values / 100.0  # Конвертация % в доли
            
            if self.use_advanced_model:
                # Усовершенствованная модель с плотностью границ
                initial_guess = [1e10, 200000, 10000, 1.0, 550.0, 900.0, 0.1]
                bounds = (
                    [1e5, 100000, 0, 0.1, 500.0, 850.0, 0.0],
                    [1e15, 500000, 50000, 4.0, 600.0, 950.0, 1.0]
                )
                
                self.params, _ = curve_fit(
                    lambda x, K0, a, b, n, T_min, T_max, alpha: 
                    sigma_phase_model_advanced([K0, a, b, n, T_min, T_max, alpha], G, T_kelvin, t),
                    np.arange(len(G)), sigma_exp,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=10000
                )
            else:
                # Базовая модель (для обратной совместимости)
                initial_guess = [1e10, 200000, 10000, 1.0, 550.0, 900.0]
                bounds = (
                    [1e5, 100000, 0, 0.1, 500.0, 850.0],
                    [1e15, 500000, 50000, 4.0, 600.0, 950.0]
                )
                
                # Простая модель без учета плотности границ
                def simple_sigma_model(params, G, T_kelvin, t):
                    K0, a, b, n, T_sigma_min, T_sigma_max = params
                    R = 8.314
                    T_min = T_sigma_min + 273.15
                    T_max = T_sigma_max + 273.15
                    
                    T_eff = np.where(T_kelvin < T_min, T_min, T_kelvin)
                    T_eff = np.where(T_eff > T_max, T_max, T_eff)
                    
                    temp_factor = 1 / (1 + np.exp(-0.1 * (T_kelvin - (T_min + 50)))) * 1 / (1 + np.exp(0.1 * (T_kelvin - (T_max - 50))))
                    
                    Q = a + b * G
                    K = K0 * np.exp(-Q / (R * T_eff)) * temp_factor
                    
                    sigma = 1 - np.exp(-K * (t ** n))
                    return sigma
                
                self.params, _ = curve_fit(
                    lambda x, K0, a, b, n, T_min, T_max: 
                    simple_sigma_model([K0, a, b, n, T_min, T_max], G, T_kelvin, t),
                    np.arange(len(G)), sigma_exp,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=10000
                )
            
            # Расчет метрик качества
            if self.use_advanced_model:
                sigma_pred = sigma_phase_model_advanced(self.params, G, T_kelvin, t) * 100
            else:
                # Используем простую модель для предсказания
                def simple_sigma_model(params, G, T_kelvin, t):
                    K0, a, b, n, T_sigma_min, T_sigma_max = params
                    R = 8.314
                    T_min = T_sigma_min + 273.15
                    T_max = T_sigma_max + 273.15
                    
                    T_eff = np.where(T_kelvin < T_min, T_min, T_kelvin)
                    T_eff = np.where(T_eff > T_max, T_max, T_eff)
                    
                    temp_factor = 1 / (1 + np.exp(-0.1 * (T_kelvin - (T_min + 50)))) * 1 / (1 + np.exp(0.1 * (T_kelvin - (T_max - 50))))
                    
                    Q = a + b * G
                    K = K0 * np.exp(-Q / (R * T_eff)) * temp_factor
                    
                    sigma = 1 - np.exp(-K * (t ** n))
                    return sigma
                
                sigma_pred = simple_sigma_model(self.params, G, T_kelvin, t) * 100
                
            sigma_exp_percent = sigma_exp * 100
            self.R2 = r2_score(sigma_exp_percent, sigma_pred)
            self.rmse = np.sqrt(mean_squared_error(sigma_exp_percent, sigma_pred))
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка при подгонке модели: {str(e)}")
            # Пробуем упрощенную модель
            try:
                st.info("Пробуем упрощенную модель...")
                self.use_advanced_model = False
                return self.fit_model(data, remove_outliers, outlier_method, contamination)
            except:
                return False
    
    def predict_temperature(self, G, sigma_percent, t):
        """Предсказание температуры по известным параметрам"""
        if self.params is None:
            raise ValueError("Модель не обучена!")
        
        R = 8.314
        sigma = sigma_percent / 100.0
        
        try:
            if self.use_advanced_model:
                K0, a, b, n, T_sigma_min, T_sigma_max, alpha = self.params
                # Упрощенный расчет для единичного значения
                grain_boundary_factor = GrainSizeConverter.calculate_activation_energy_factor(G)
                Q_effective = (a + b * G) * (1 + alpha * (grain_boundary_factor - 1))
            else:
                K0, a, b, n, T_sigma_min, T_sigma_max = self.params
                Q_effective = a + b * G
            
            term = -np.log(1 - sigma) / (K0 * (t ** n))
            if term <= 0:
                return None
            
            T_kelvin = -Q_effective / (R * np.log(term))
            T_celsius = T_kelvin - 273.15
            
            # Применяем температурные ограничения
            if T_celsius < T_sigma_min:
                return T_sigma_min
            elif T_celsius > T_sigma_max:
                return T_sigma_max
            else:
                return T_celsius
                
        except:
            return None
    
    def plot_results_with_outliers(self, data):
        """Визуализация результатов с выделением выбросов"""
        if self.params is None:
            return None
        
        # Предсказанные значения для всех данных
        G_all = data['G'].values
        T_celsius_all = data['T'].values
        T_kelvin_all = T_celsius_all + 273.15
        t_all = data['t'].values
        sigma_exp_all = data['f_exp (%)'].values
        
        if self.use_advanced_model:
            sigma_pred_all = sigma_phase_model_advanced(self.params, G_all, T_kelvin_all, t_all) * 100
        else:
            # Простая модель для предсказания
            def simple_sigma_model(params, G, T_kelvin, t):
                K0, a, b, n, T_sigma_min, T_sigma_max = params
                R = 8.314
                T_min = T_sigma_min + 273.15
                T_max = T_sigma_max + 273.15
                
                T_eff = np.where(T_kelvin < T_min, T_min, T_kelvin)
                T_eff = np.where(T_eff > T_max, T_max, T_eff)
                
                temp_factor = 1 / (1 + np.exp(-0.1 * (T_kelvin - (T_min + 50)))) * 1 / (1 + np.exp(0.1 * (T_kelvin - (T_max - 50))))
                
                Q = a + b * G
                K = K0 * np.exp(-Q / (R * T_eff)) * temp_factor
                
                sigma = 1 - np.exp(-K * (t ** n))
                return sigma
            
            sigma_pred_all = simple_sigma_model(self.params, G_all, T_kelvin_all, t_all) * 100
        
        # Определяем, какие точки являются выбросами
        is_outlier = np.zeros(len(data), dtype=bool)
        outlier_indices = []
        if self.outlier_info and self.outlier_info['outlier_data'] is not None:
            outlier_indices = self.outlier_info['outlier_data'].index
            is_outlier = data.index.isin(outlier_indices)
        
        # Создание графиков
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Экспериментальные vs Предсказанные значения',
                'Распределение остатков',
                'Временные зависимости',
                'Температурные зависимости'
            )
        )
        
        # График 1: Предсказанные vs экспериментальные с выбросами
        clean_mask = ~is_outlier
        outlier_mask = is_outlier
        
        # Чистые данные
        fig.add_trace(
            go.Scatter(x=sigma_exp_all[clean_mask], y=sigma_pred_all[clean_mask], 
                      mode='markers', name='Чистые данные',
                      marker=dict(color='blue', size=8)),
            row=1, col=1
        )
        
        # Выбросы
        if np.any(outlier_mask):
            fig.add_trace(
                go.Scatter(x=sigma_exp_all[outlier_mask], y=sigma_pred_all[outlier_mask],
                          mode='markers', name='Выбросы',
                          marker=dict(color='red', size=10, symbol='x')),
                row=1, col=1
            )
        
        # Линия идеального соответствия
        max_val = max(sigma_exp_all.max(), sigma_pred_all.max())
        fig.add_trace(
            go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                      name='Идеальное соответствие', line=dict(dash='dash', color='black')),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text='Экспериментальные значения f_exp (%)', row=1, col=1)
        fig.update_yaxes(title_text='Предсказанные значения (%)', row=1, col=1)
        
        # График 2: Распределение остатков
        residuals = sigma_pred_all - sigma_exp_all
        fig.add_trace(
            go.Histogram(x=residuals, name='Распределение остатков',
                        marker_color='lightblue'),
            row=1, col=2
        )
        fig.update_xaxes(title_text='Остатки (%)', row=1, col=2)
        fig.update_yaxes(title_text='Частота', row=1, col=2)
        
        # График 3: Временные зависимости
        unique_temps = sorted(data['T'].unique())
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, temp in enumerate(unique_temps):
            if i >= len(colors):
                break
                
            temp_data = data[data['T'] == temp]
            temp_outliers = temp_data[temp_data.index.isin(outlier_indices)] if len(outlier_indices) > 0 else pd.DataFrame()
            temp_clean = temp_data[~temp_data.index.isin(outlier_indices)] if len(outlier_indices) > 0 else temp_data
            
            # Чистые данные
            if len(temp_clean) > 0:
                fig.add_trace(
                    go.Scatter(x=temp_clean['t'], y=temp_clean['f_exp (%)'],
                              mode='markers', name=f'Чистые {temp}°C',
                              marker=dict(color=colors[i], size=8)),
                    row=2, col=1
                )
            
            # Выбросы
            if len(temp_outliers) > 0:
                fig.add_trace(
                    go.Scatter(x=temp_outliers['t'], y=temp_outliers['f_exp (%)'],
                              mode='markers', name=f'Выбросы {temp}°C',
                              marker=dict(color=colors[i], size=10, symbol='x')),
                    row=2, col=1
                )
        
        fig.update_xaxes(title_text='Время t (ч)', row=2, col=1)
        fig.update_yaxes(title_text='Сигма-фаза f_exp (%)', row=2, col=1)
        
        # График 4: Температурные зависимости
        unique_times = sorted(data['t'].unique())[:3]  # Первые 3 времени
        for i, time_val in enumerate(unique_times):
            if i >= len(colors):
                break
                
            time_data = data[data['t'] == time_val]
            time_outliers = time_data[time_data.index.isin(outlier_indices)] if len(outlier_indices) > 0 else pd.DataFrame()
            time_clean = time_data[~time_data.index.isin(outlier_indices)] if len(outlier_indices) > 0 else time_data
            
            # Чистые данные
            if len(time_clean) > 0:
                fig.add_trace(
                    go.Scatter(x=time_clean['T'], y=time_clean['f_exp (%)'],
                              mode='markers', name=f'Чистые {time_val} ч',
                              marker=dict(color=colors[i], size=8)),
                    row=2, col=2
                )
        
        fig.update_xaxes(title_text='Температура T (°C)', row=2, col=2)
        fig.update_yaxes(title_text='Сигма-фаза f_exp (%)', row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True)
        return fig
    
    def to_dict(self):
        """Сериализация модели в словарь"""
        return {
            'params': self.params.tolist() if self.params is not None else None,
            'R2': self.R2,
            'rmse': self.rmse,
            'outlier_info': self.outlier_info,
            'original_data': self.original_data.to_dict() if self.original_data is not None else None,
            'clean_data': self.clean_data.to_dict() if self.clean_data is not None else None,
            'model_version': self.model_version,
            'creation_date': self.creation_date,
            'last_modified': self.last_modified,
            'use_advanced_model': self.use_advanced_model
        }
    
    @classmethod
    def from_dict(cls, data_dict):
        """Десериализация модели из словаря"""
        analyzer = cls()
        analyzer.params = np.array(data_dict['params']) if data_dict['params'] is not None else None
        analyzer.R2 = data_dict['R2']
        analyzer.rmse = data_dict['rmse']
        analyzer.outlier_info = data_dict['outlier_info']
        
        if data_dict['original_data'] is not None:
            analyzer.original_data = pd.DataFrame(data_dict['original_data'])
        if data_dict['clean_data'] is not None:
            analyzer.clean_data = pd.DataFrame(data_dict['clean_data'])
            
        analyzer.model_version = data_dict.get('model_version', '1.0')
        analyzer.creation_date = data_dict.get('creation_date')
        analyzer.last_modified = data_dict.get('last_modified')
        analyzer.use_advanced_model = data_dict.get('use_advanced_model', True)
        
        return analyzer

def main():
    # Инициализация сессии
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    
    # Боковая панель для загрузки данных и управления проектом
    st.sidebar.header("📁 Управление проектом")
    
    # Загрузка/сохранение проекта
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("💾 Сохранить проект"):
            if st.session_state.analyzer is not None and st.session_state.current_data is not None:
                project_data = {
                    'analyzer': st.session_state.analyzer.to_dict(),
                    'current_data': st.session_state.current_data.to_dict()
                }
                
                project_json = json.dumps(project_data, indent=2)
                st.download_button(
                    label="Скачать проект",
                    data=project_json,
                    file_name=f"sigma_phase_project_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            else:
                st.error("Нет данных для сохранения")
    
    with col2:
        uploaded_project = st.sidebar.file_uploader(
            "Загрузить проект",
            type=['json'],
            key="project_uploader"
        )
        
        if uploaded_project is not None:
            try:
                project_data = json.load(uploaded_project)
                st.session_state.analyzer = SigmaPhaseAnalyzer.from_dict(project_data['analyzer'])
                st.session_state.current_data = pd.DataFrame(project_data['current_data'])
                st.sidebar.success("Проект успешно загружен!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Ошибка загрузки проекта: {str(e)}")
    
    # Настройки обработки выбросов
    st.sidebar.header("🎯 Настройки обработки выбросов")
    
    remove_outliers = st.sidebar.checkbox("Удалять выбросы", value=True)
    
    if remove_outliers:
        outlier_method = st.sidebar.selectbox(
            "Метод обнаружения выбросов",
            ['iqr', 'isolation_forest'],
            format_func=lambda x: {
                'iqr': 'Межквартильный размах (IQR)',
                'isolation_forest': 'Isolation Forest'
            }[x]
        )
        
        contamination = st.sidebar.slider(
            "Ожидаемая доля выбросов", 
            min_value=0.01, max_value=0.3, value=0.1, step=0.01
        )
    else:
        outlier_method = 'none'
        contamination = 0.1
    
    # Пример данных с новыми названиями колонок
    sample_data = pd.DataFrame({
        'G': [3, 3, 5, 5, 8, 8, 9, 9, 3, 5, 8],
        'T': [600, 650, 600, 700, 650, 700, 600, 700, 600, 650, 750],
        't': [2000, 4000, 4000, 2000, 6000, 4000, 8000, 6000, 2000, 4000, 4000],
        'f_exp (%)': [5.2, 12.5, 8.1, 15.3, 18.7, 25.1, 22.4, 35.2, 12.8, 25.6, 2.1]
    })
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False, decimal=',').encode('utf-8')
    
    sample_csv = convert_df_to_csv(sample_data)
    
    st.sidebar.download_button(
        label="📥 Скачать пример данных (CSV)",
        data=sample_csv,
        file_name="sample_sigma_phase_data.csv",
        mime="text/csv"
    )
    
    # Загрузка данных
    st.sidebar.header("📊 Загрузка данных")
    
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите файл с экспериментальными данными",
        type=['csv', 'xlsx', 'xls'],
        help="Поддерживаемые форматы: CSV, Excel (.xlsx, .xls)"
    )
    
    # Обработка загруженного файла
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Для CSV с запятой как разделителем десятичных
                data = pd.read_csv(uploaded_file, decimal=',')
            else:
                # Для Excel файлов
                if uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file, engine='openpyxl')
                else:  # .xls
                    data = pd.read_excel(uploaded_file, engine='xlrd')
            
            # Проверяем необходимые колонки
            required_columns = ['G', 'T', 't', 'f_exp (%)']
            if all(col in data.columns for col in required_columns):
                # Округляем значения до тысячных
                data['f_exp (%)'] = data['f_exp (%)'].round(3)
                st.session_state.current_data = data
                st.sidebar.success("Данные успешно загружены!")
            else:
                st.sidebar.error("В файле отсутствуют необходимые колонки")
                st.sidebar.info("Необходимые колонки: G, T, t, f_exp (%)")
                
        except Exception as e:
            st.sidebar.error(f"Ошибка чтения файла: {str(e)}")
    
    # Если данных нет, используем пример
    if st.session_state.current_data is None:
        st.info("👈 Пожалуйста, загрузите файл с данными или используйте пример данных")
        st.session_state.current_data = sample_data
    
    # Показ загруженных данных
    st.header("📊 Экспериментальные данные")
    
    # Детальная информация о размерах зерен из ГОСТ
    st.subheader("📐 Данные о размерах зерен по ГОСТ 5639-82")
    
    if st.checkbox("Показать полную таблицу ГОСТ"):
        gost_table = GrainSizeConverter.get_grain_info_table()
        st.dataframe(gost_table, use_container_width=True)
    
    # Информация о зернах в данных
    if st.session_state.current_data is not None:
        unique_grain_numbers = sorted(st.session_state.current_data['G'].unique())
        
        st.write("**Характеристики зерен в экспериментальных данных:**")
        cols = st.columns(min(5, len(unique_grain_numbers)))
        
        for i, grain_num in enumerate(unique_grain_numbers):
            with cols[i % 5]:
                area = GrainSizeConverter.grain_number_to_area(grain_num)
                diameter = GrainSizeConverter.grain_number_to_diameter(grain_num)
                boundary_density = GrainSizeConverter.calculate_grain_boundary_density(grain_num)
                activation_factor = GrainSizeConverter.calculate_activation_energy_factor(grain_num)
                
                st.metric(
                    f"Номер {grain_num}",
                    f"{diameter*1000:.1f} мкм",
                    f"Плотность: {boundary_density:.0f} мм²/мм³"
                )
                st.caption(f"Площадь: {area:.4f} мм²")
                st.caption(f"Коэф. активации: {activation_factor:.3f}")
    
    # Редактирование данных
    st.write("**Редактирование данных (округление до тысячных):**")
    edited_data = st.data_editor(
        st.session_state.current_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "f_exp (%)": st.column_config.NumberColumn(
                format="%.3f"
            )
        }
    )
    
    # Округляем значения после редактирования
    if 'f_exp (%)' in edited_data.columns:
        edited_data['f_exp (%)'] = edited_data['f_exp (%)'].round(3)
    
    if not edited_data.equals(st.session_state.current_data):
        st.session_state.current_data = edited_data
        st.session_state.analyzer = None  # Сбрасываем модель при изменении данных
        st.rerun()
    
    # Анализ данных
    st.header("🔍 Анализ данных и обнаружение выбросов")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🎯 Подобрать параметры модели", use_container_width=True):
            analyzer = SigmaPhaseAnalyzer()
            
            with st.spinner("Идет подбор параметров модели и анализ выбросов..."):
                success = analyzer.fit_model(
                    st.session_state.current_data, 
                    remove_outliers=remove_outliers,
                    outlier_method=outlier_method,
                    contamination=contamination
                )
            
            if success:
                st.session_state.analyzer = analyzer
                st.success("✅ Модель успешно обучена!")
                st.rerun()
    
    if st.session_state.analyzer is not None:
        analyzer = st.session_state.analyzer
        
        # Информация о выбросах
        if remove_outliers and analyzer.outlier_info['outlier_count'] > 0:
            st.subheader("🚨 Обнаруженные выбросы")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Количество выбросов", analyzer.outlier_info['outlier_count'])
            with col2:
                st.metric("Доля выбросов", 
                         f"{analyzer.outlier_info['outlier_count']/analyzer.outlier_info['total_count']:.1%}")
            
            st.write("**Выбросы:**")
            st.dataframe(analyzer.outlier_info['outlier_data'])
        
        # Параметры модели
        st.subheader("📈 Параметры модели")
        
        if analyzer.params is not None:
            if analyzer.use_advanced_model:
                K0, a, b, n, T_sigma_min, T_sigma_max, alpha = analyzer.params
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("K₀", f"{K0:.2e}")
                    st.metric("a", f"{a:.2f}")
                with col2:
                    st.metric("b", f"{b:.2f}")
                    st.metric("n", f"{n:.3f}")
                with col3:
                    st.metric("T_min (°C)", f"{T_sigma_min:.1f}")
                    st.metric("T_max (°C)", f"{T_sigma_max:.1f}")
                with col4:
                    st.metric("α", f"{alpha:.3f}")
                    st.metric("Модель", "Расширенная")
                    
            else:
                K0, a, b, n, T_sigma_min, T_sigma_max = analyzer.params
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("K₀", f"{K0:.2e}")
                    st.metric("a", f"{a:.2f}")
                with col2:
                    st.metric("b", f"{b:.2f}")
                    st.metric("n", f"{n:.3f}")
                with col3:
                    st.metric("T_min (°C)", f"{T_sigma_min:.1f}")
                    st.metric("T_max (°C)", f"{T_sigma_max:.1f}")
                with col4:
                    st.metric("Модель", "Базовая")
            
            # Метрики качества
            st.subheader("📊 Метрики качества модели")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R²", f"{analyzer.R2:.4f}")
            with col2:
                st.metric("RMSE", f"{analyzer.rmse:.2f}%")
            
            # Визуализация
            st.subheader("📈 Визуализация результатов")
            fig = analyzer.plot_results_with_outliers(st.session_state.current_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Калькулятор температуры
            st.header("🧮 Калькулятор температуры эксплуатации")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                G_input = st.number_input("Номер зерна (G)", 
                                        min_value=-3.0, max_value=14.0, 
                                        value=5.0, step=0.1)
            with col2:
                sigma_input = st.number_input("Содержание сигма-фазы f_exp (%)", 
                                            min_value=0.0, max_value=50.0,
                                            value=10.0, step=0.1,
                                            format="%.3f",
                                            help="От 0% до 50%")
            with col3:
                t_input = st.number_input("Время эксплуатации t (ч)", 
                                        min_value=100, max_value=100000,
                                        value=4000, step=100)
            
            if st.button("🔍 Рассчитать температуру", key="calc_temp"):
                try:
                    T_celsius = analyzer.predict_temperature(G_input, sigma_input, t_input)
                    
                    if T_celsius is not None:
                        if analyzer.use_advanced_model:
                            T_sigma_min = analyzer.params[4]
                            T_sigma_max = analyzer.params[5]
                        else:
                            T_sigma_min = analyzer.params[4]
                            T_sigma_max = analyzer.params[5]
                        
                        st.success(f"""
                        ### Результат расчета:
                        - **Температура эксплуатации:** {T_celsius:.1f}°C
                        - При номере зерна: {G_input}
                        - Содержании сигма-фазы: {sigma_input:.3f}%
                        - Наработке: {t_input} ч
                        - **Температурный диапазон модели:** {T_sigma_min:.1f}°C - {T_sigma_max:.1f}°C
                        """)
                        
                        # Проверка на границы диапазона
                        if T_celsius <= T_sigma_min + 10:
                            st.warning("⚠️ Расчетная температура близка к нижней границе образования сигма-фазы")
                        elif T_celsius >= T_sigma_max - 10:
                            st.warning("⚠️ Расчетная температура близка к верхней границе растворения сигма-фазы")
                    else:
                        st.error("Не удалось рассчитать температуру. Проверьте входные параметры.")
                        
                except Exception as e:
                    st.error(f"Ошибка при расчете: {str(e)}")

if __name__ == "__main__":
    main()
