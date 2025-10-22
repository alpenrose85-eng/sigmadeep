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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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

class ComplexDataParser:
    """Класс для парсинга сложных Excel файлов с данными сигма-фазы"""
    
    @staticmethod
    def parse_complex_excel(file_path):
        """Парсинг сложного Excel файла с экспериментальными данными"""
        try:
            # Читаем файл
            df = pd.read_excel(file_path, sheet_name=0, header=None)
            
            results = []
            
            # Проходим по строкам с данными (начиная со строки 2, так как строка 1 - заголовок)
            for i in range(2, len(df)):
                row = df.iloc[i]
                
                # Проверяем, есть ли основные данные в первых 4 колонках
                if pd.notna(row[0]) and pd.notna(row[1]) and pd.notna(row[2]) and pd.notna(row[3]):
                    try:
                        G = float(row[0])
                        T = float(row[1])
                        t = float(row[2])
                        f_exp = float(row[3])
                        
                        # Проверяем корректность данных
                        if (G in [3, 5, 8, 9, 10] and 
                            T in [600, 650, 700] and 
                            t in [2000, 4000, 6000, 8000] and
                            0 <= f_exp <= 10):
                            
                            results.append({
                                'G': G,
                                'T': T, 
                                't': t,
                                'f_exp (%)': f_exp
                            })
                    except (ValueError, TypeError):
                        continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            raise Exception(f"Ошибка парсинга файла: {e}")

    @staticmethod
    def extract_all_data(uploaded_file):
        """Извлечение всех данных из загруженного файла"""
        try:
            # Создаем временный файл
            with open("temp_file.xlsx", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Парсим данные
            data = ComplexDataParser.parse_complex_excel("temp_file.xlsx")
            
            # Удаляем временный файл
            import os
            if os.path.exists("temp_file.xlsx"):
                os.remove("temp_file.xlsx")
                
            # ВОЗВРАЩАЕМ ТОЛЬКО ЕСЛИ ЕСТЬ ДАННЫЕ
            if data is not None and len(data) > 0:
                return data
            else:
                return None
                
        except Exception as e:
            st.error(f"❌ Ошибка извлечения данных: {e}")
            return None

class DataValidator:
    """Класс для валидации и нормализации данных"""
    
    @staticmethod
    def normalize_column_names(df):
        """Нормализует названия колонок к стандартному формату"""
        column_mapping = {
            'Номер_зерна': 'G', 'Номер зерна': 'G', 'Зерно': 'G',
            'Температура': 'T', 'Температура_C': 'T', 'Температура °C': 'T',
            'Время': 't', 'Время_ч': 't', 'Время, ч': 't',
            'Сигма_фаза': 'f_exp (%)', 'Сигма-фаза': 'f_exp (%)', 
            'Сигма_фаза_%': 'f_exp (%)', 'Сигма фаза': 'f_exp (%)',
            'Grain': 'G', 'Grain_number': 'G', 'Grain size': 'G',
            'Temperature': 'T', 'Temp': 'T', 'Temperature_C': 'T',
            'Time': 't', 'Time_h': 't', 'Hours': 't',
            'Sigma_phase': 'f_exp (%)', 'Sigma': 'f_exp (%)', 
            'Sigma_%': 'f_exp (%)', 'f_exp': 'f_exp (%)'
        }
        
        df_normalized = df.copy()
        new_columns = {}
        
        for col in df.columns:
            col_clean = str(col).strip()
            if col_clean in column_mapping:
                new_columns[col] = column_mapping[col_clean]
            else:
                new_columns[col] = col_clean
        
        df_normalized.columns = [new_columns[col] for col in df.columns]
        return df_normalized
    
    @staticmethod
    def validate_data(df):
        """Проверяет наличие обязательных колонок и корректность данных"""
        required_columns = ['G', 'T', 't', 'f_exp (%)']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Отсутствуют обязательные колонки: {missing_columns}"
        
        try:
            df['G'] = pd.to_numeric(df['G'], errors='coerce')
            df['T'] = pd.to_numeric(df['T'], errors='coerce')
            df['t'] = pd.to_numeric(df['t'], errors='coerce')
            df['f_exp (%)'] = pd.to_numeric(df['f_exp (%)'], errors='coerce')
        except Exception as e:
            return False, f"Ошибка преобразования типов данных: {e}"
        
        if df[required_columns].isna().any().any():
            return False, "Обнаружены пустые или некорректные значения в данных"
        
        if (df['G'] < -3).any() or (df['G'] > 14).any():
            return False, "Номер зерна должен быть в диапазоне от -3 до 14"
        
        if (df['T'] < 500).any() or (df['T'] > 1000).any():
            st.warning("⚠️ Некоторые температуры выходят за typical диапазон 500-1000°C")
        
        if (df['f_exp (%)'] < 0).any() or (df['f_exp (%)'] > 50).any():
            st.warning("⚠️ Некоторые значения содержания сигма-фазы выходят за typical диапазон 0-50%")
        
        DataValidator.validate_time_range(df['t'])
        
        return True, "Данные валидны"
    
    @staticmethod
    def validate_time_range(t_values):
        """Проверка диапазона времени эксплуатации"""
        max_time = 500000
        if (t_values > max_time).any():
            st.warning(f"⚠️ Обнаружены значения времени эксплуатации свыше {max_time} часов")
        return True

class GrainSizeConverter:
    """Класс для преобразования номера зерна в физические параметры по ГОСТ 5639-82"""
    
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
            numbers = sorted(cls.GRAIN_DATA.keys())
            if grain_number < numbers[0]:
                return cls.GRAIN_DATA[numbers[0]]['area_mm2']
            elif grain_number > numbers[-1]:
                return cls.GRAIN_DATA[numbers[-1]]['area_mm2']
            else:
                lower = max([n for n in numbers if n <= grain_number])
                upper = min([n for n in numbers if n >= grain_number])
                if lower == upper:
                    return cls.GRAIN_DATA[lower]['area_mm2']
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
                diam_lower = cls.GRAIN_DATA[lower]['conditional_diameter_mm'] if use_conditional else cls.GRAIN_DATA[lower]['diameter_mm']
                diam_upper = cls.GRAIN_DATA[upper]['conditional_diameter_mm'] if use_conditional else cls.GRAIN_DATA[upper]['diameter_mm']
                fraction = (grain_number - lower) / (upper - lower)
                return diam_lower + fraction * (diam_upper - diam_lower)
    
    @classmethod
    def calculate_grain_boundary_density(cls, grain_number):
        """Расчет плотности границ зерен (мм²/мм³)"""
        d = cls.grain_number_to_diameter(grain_number, use_conditional=True)
        Sv = 3.0 / (d / 2.0)
        return Sv
    
    @classmethod
    def calculate_activation_energy_factor(cls, grain_number):
        """Коэффициент влияния размера зерна на энергию активации"""
        ref_grain = 5
        Sv_ref = cls.calculate_grain_boundary_density(ref_grain)
        Sv_current = cls.calculate_grain_boundary_density(grain_number)
        return Sv_current / Sv_ref

class SigmaPhaseAnalyzer:
    def __init__(self):
        self.params = None
        self.R2 = None
        self.rmse = None
        self.mape = None
        self.model_type = None
        self.final_formula = ""
        
    def fit_model(self, data, model_type="avrami_saturation"):
        """Подгонка выбранной модели с физическими ограничениями"""
        try:
            # Фильтруем данные для рабочего диапазона 580-630°C
            working_data = data[(data['T'] >= 580) & (data['T'] <= 630)].copy()
            
            # Если в рабочем диапазоне мало данных, используем все данные но с весами
            if len(working_data) < 8:
                weights = np.ones(len(data))
                # Даем больший вес точкам в рабочем диапазоне
                weights[(data['T'] >= 580) & (data['T'] <= 630)] = 3.0
                weights[(data['T'] >= 550) & (data['T'] < 580)] = 1.5
                weights[(data['T'] > 630) & (data['T'] <= 700)] = 1.5
                used_data = data
            else:
                weights = np.ones(len(working_data))
                used_data = working_data
            
            G = used_data['G'].values
            T = used_data['T'].values + 273.15  # в Кельвины
            t = used_data['t'].values
            f_exp = used_data['f_exp (%)'].values
            
            self.model_type = model_type
            
            if model_type == "avrami_saturation":
                success = self._fit_avrami_model(G, T, t, f_exp, weights)
            elif model_type == "power_law":
                success = self._fit_power_law_model(G, T, t, f_exp, weights)
            elif model_type == "logistic":
                success = self._fit_logistic_model(G, T, t, f_exp, weights)
            elif model_type == "ensemble":
                success = self._fit_ensemble_model(G, T, t, f_exp, weights)
            else:
                st.error(f"Неизвестный тип модели: {model_type}")
                return False
                
            if success:
                self._generate_final_formula()
                
            return success
            
        except Exception as e:
            st.error(f"Ошибка при подгонке модели: {e}")
            return False
    
    def _fit_avrami_model(self, G, T, t, f_exp, weights):
        """Модель Аврами с насыщением и физическими ограничениями"""
        # Начальные приближения с учетом физики процесса
        initial_guess = [12.0, 1e12, 250000, 1.2, 0.15]  # f_max, K0, Q, n, alpha
        
        # Границы с физическими ограничениями
        bounds = (
            [5.0, 1e8, 200000, 0.8, 0.05],   # нижние границы
            [25.0, 1e16, 350000, 2.5, 0.3]    # верхние границы
        )
        
        def model(params, G, T, t):
            f_max, K0, Q, n, alpha = params
            R = 8.314
            
            # Эффект размера зерна
            grain_effect = 1 + alpha * (G - 8)
            
            # Константа скорости с учетом энергии активации
            K = K0 * np.exp(-Q / (R * T)) * grain_effect
            
            # Модель Аврами
            return f_max * (1 - np.exp(-K * (t ** n)))
        
        try:
            self.params, _ = curve_fit(
                lambda x, f_max, K0, Q, n, alpha: model([f_max, K0, Q, n, alpha], G, T, t),
                np.arange(len(G)), f_exp,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
                sigma=1.0/weights  # веса для точек данных
            )
            
            f_pred = model(self.params, G, T, t)
            self.R2 = r2_score(f_exp, f_pred)
            self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
            self.mape = np.mean(np.abs((f_exp - f_pred) / np.maximum(f_exp, 0.1))) * 100
            
            return True
            
        except Exception as e:
            st.warning(f"Оптимизация Аврами не сошлась, пробуем упрощенную модель: {e}")
            return self._fit_simplified_avrami(G, T, t, f_exp, weights)
    
    def _fit_simplified_avrami(self, G, T, t, f_exp, weights):
        """Упрощенная модель Аврами с фиксированными параметрами"""
        initial_guess = [10.0, 1e10, 220000, 0.1]  # f_max, K0, Q, alpha
        
        bounds = (
            [5.0, 1e8, 180000, 0.05],
            [20.0, 1e14, 280000, 0.2]
        )
        
        def model(params, G, T, t):
            f_max, K0, Q, alpha = params
            R = 8.314
            grain_effect = 1 + alpha * (G - 8)
            K = K0 * np.exp(-Q / (R * T)) * grain_effect
            # Фиксируем n = 1 для упрощения
            return f_max * (1 - np.exp(-K * t))
        
        self.params, _ = curve_fit(
            lambda x, f_max, K0, Q, alpha: model([f_max, K0, Q, alpha], G, T, t),
            np.arange(len(G)), f_exp,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000,
            sigma=1.0/weights
        )
        
        # Добавляем фиксированный n = 1 к параметрам
        self.params = np.append(self.params, 1.0)
        
        f_pred = model(self.params[:-1], G, T, t)
        self.R2 = r2_score(f_exp, f_pred)
        self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
        self.mape = np.mean(np.abs((f_exp - f_pred) / np.maximum(f_exp, 0.1))) * 100
        
        return True
    
    def _fit_power_law_model(self, G, T, t, f_exp, weights):
        """Степенная модель с физическими ограничениями"""
        initial_guess = [2.0, 0.5, -18000, 0.7, 0.08]  # A, B, C, D, E
        
        bounds = (
            [0.1, 0.0, -30000, 0.3, 0.02],
            [10.0, 2.0, -12000, 1.5, 0.2]
        )
        
        def model(params, G, T, t):
            A, B, C, D, E = params
            R = 8.314
            # Экспоненциальная зависимость от температуры (обратная)
            temp_effect = np.exp(C / (R * T))
            time_effect = t ** D
            grain_effect = 1 + E * (G - 8)
            return A * temp_effect * time_effect * grain_effect + B
        
        self.params, _ = curve_fit(
            lambda x, A, B, C, D, E: model([A, B, C, D, E], G, T, t),
            np.arange(len(G)), f_exp,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000,
            sigma=1.0/weights
        )
        
        f_pred = model(self.params, G, T, t)
        self.R2 = r2_score(f_exp, f_pred)
        self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
        self.mape = np.mean(np.abs((f_exp - f_pred) / np.maximum(f_exp, 0.1))) * 100
        return True
    
    def _fit_logistic_model(self, G, T, t, f_exp, weights):
        """Логистическая модель с насыщением"""
        initial_guess = [15.0, 1e-6, 2000, 0.12, -15000]  # f_max, k, t0, alpha, beta
        
        bounds = (
            [8.0, 1e-8, 500, 0.05, -25000],
            [30.0, 1e-4, 5000, 0.25, -8000]
        )
        
        def model(params, G, T, t):
            f_max, k, t0, alpha, beta = params
            R = 8.314
            temp_factor = np.exp(beta / (R * T))
            grain_factor = 1 + alpha * (G - 8)
            rate = k * temp_factor * grain_factor
            return f_max / (1 + np.exp(-rate * (t - t0)))
        
        self.params, _ = curve_fit(
            lambda x, f_max, k, t0, alpha, beta: model([f_max, k, t0, alpha, beta], G, T, t),
            np.arange(len(G)), f_exp,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000,
            sigma=1.0/weights
        )
        
        f_pred = model(self.params, G, T, t)
        self.R2 = r2_score(f_exp, f_pred)
        self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
        self.mape = np.mean(np.abs((f_exp - f_pred) / np.maximum(f_exp, 0.1))) * 100
        return True
    
    def _fit_ensemble_model(self, G, T, t, f_exp, weights):
        """Ансамблевая модель с приоритетом на рабочий диапазон"""
        initial_guess = [10.0, 1e10, 220000, 1.0, 0.1, 0.5, -15000]
        
        bounds = (
            [5.0, 1e8, 180000, 0.5, 0.05, 0.1, -25000],
            [20.0, 1e14, 280000, 1.8, 0.2, 2.0, -8000]
        )
        
        def model(params, G, T, t):
            f_max, K0, Q, n, alpha, w, beta = params
            R = 8.314
            
            # Аврами компонент (основной)
            grain_effect_avrami = 1 + alpha * (G - 8)
            K_avrami = K0 * np.exp(-Q / (R * T)) * grain_effect_avrami
            f_avrami = f_max * (1 - np.exp(-K_avrami * (t ** n)))
            
            # Корректирующий компонент для учета нелинейностей
            temp_effect_power = np.exp(beta / (R * T))
            f_power = w * temp_effect_power * (t ** 0.3) * (1 + 0.03 * (G - 8))
            
            return f_avrami + f_power
        
        self.params, _ = curve_fit(
            lambda x, f_max, K0, Q, n, alpha, w, beta: model([f_max, K0, Q, n, alpha, w, beta], G, T, t),
            np.arange(len(G)), f_exp,
            p0=initial_guess,
            bounds=bounds,
            maxfev=15000,
            sigma=1.0/weights
        )
        
        f_pred = model(self.params, G, T, t)
        self.R2 = r2_score(f_exp, f_pred)
        self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
        self.mape = np.mean(np.abs((f_exp - f_pred) / np.maximum(f_exp, 0.1))) * 100
        return True

    def _generate_final_formula(self):
        """Генерация читаемой формулы модели"""
        if self.params is None:
            self.final_formula = "Модель не обучена"
            return
            
        if self.model_type == "avrami_saturation":
            f_max, K0, Q, n, alpha = self.params
            self.final_formula = f"""
**Модель Аврами с насыщением:**
            f(G, T, t) = {f_max:.3f} × [1 - exp(-K × t^{n:.3f})]
K = {K0:.3e} × exp(-{Q/1000:.1f} кДж/моль / (R × T)) × [1 + {alpha:.3f} × (G - 8)]
            """
        elif self.model_type == "power_law":
            A, B, C, D, E = self.params
            self.final_formula = f"""
**Степенная модель:**
            f(G, T, t) = {A:.3f} × exp({C:.0f} / (R × T)) × t^{D:.3f} × [1 + {E:.3f} × (G - 8)] + {B:.3f}
            """
        elif self.model_type == "logistic":
            f_max, k, t0, alpha, beta = self.params
            self.final_formula = f"""
**Логистическая модель:**
            f(G, T, t) = {f_max:.3f} / [1 + exp(-k × (t - {t0:.0f}))]
k = {k:.3e} × exp({beta:.0f} / (R × T)) × [1 + {alpha:.3f} × (G - 8)]
            """
        elif self.model_type == "ensemble":
            f_max, K0, Q, n, alpha, w, beta = self.params
            self.final_formula = f"""
**Ансамблевая модель:**
            f(G, T, t) = f_avrami + f_power

f_avrami = {f_max:.3f} × [1 - exp(-K_avrami × t^{n:.3f})]
K_avrami = {K0:.3e} × exp(-{Q/1000:.1f} кДж/моль / (R × T)) × [1 + {alpha:.3f} × (G - 8)]

f_power = {w:.3f} × exp({beta:.0f} / (R × T)) × t^0.3 × [1 + 0.03 × (G - 8)]
            """
      
        self.final_formula += "\n**R = 8.314 Дж/(моль·К) - универсальная газовая постоянная**\n**T - температура в Кельвинах (T[°C] + 273.15)**"
    
    def predict_temperature(self, G, sigma_percent, t):
        """Предсказание температуры с физическими ограничениями"""
        if self.params is None:
            raise ValueError("Модель не обучена!")
        
        sigma = sigma_percent
        
        # Бисекционный поиск в рабочем диапазоне 580-630°C
        T_min, T_max = 580, 630
        
        for i in range(50):  # уменьшили количество итераций для скорости
            T_mid = (T_min + T_max) / 2
            f_pred = self._evaluate_model(G, T_mid, t)
            
            if abs(f_pred - sigma) < 0.5:  # более строгая точность
                return T_mid
            
            if f_pred < sigma:
                T_min = T_mid
            else:
                T_max = T_mid
        
        # Если не сошлось в рабочем диапазоне, расширяем поиск
        final_T = (T_min + T_max) / 2
        
        # Проверяем физические ограничения
        if final_T < 550:
            st.warning("⚠️ Расчетная температура ниже физического предела образования сигма-фазы (550°C)")
            return 550
        elif final_T > 700:
            st.warning("⚠️ Расчетная температура выше типичного диапазона для стали 12Х18Н12Т")
            return 700
            
        return final_T
    
    def _evaluate_model(self, G, T, t):
        """Вычисление модели для данных параметров"""
        if self.params is None:
            return 0.0
            
        T_kelvin = T + 273.15
        
        if self.model_type == "avrami_saturation":
            f_max, K0, Q, n, alpha = self.params
            R = 8.314
            grain_effect = 1 + alpha * (G - 8)
            K = K0 * np.exp(-Q / (R * T_kelvin)) * grain_effect
            return f_max * (1 - np.exp(-K * (t ** n)))
        
        elif self.model_type == "power_law":
            A, B, C, D, E = self.params
            R = 8.314
            temp_effect = np.exp(C / (R * T_kelvin))
            time_effect = t ** D
            grain_effect = 1 + E * (G - 8)
            return A * temp_effect * time_effect * grain_effect + B
        
        elif self.model_type == "logistic":
            f_max, k, t0, alpha, beta = self.params
            R = 8.314
            temp_factor = np.exp(beta / (R * T_kelvin))
            grain_factor = 1 + alpha * (G - 8)
            rate = k * temp_factor * grain_factor
            return f_max / (1 + np.exp(-rate * (t - t0)))
        
        elif self.model_type == "ensemble":
            f_max, K0, Q, n, alpha, w, beta = self.params
            R = 8.314
            
            grain_effect_avrami = 1 + alpha * (G - 8)
            K_avrami = K0 * np.exp(-Q / (R * T_kelvin)) * grain_effect_avrami
            f_avrami = f_max * (1 - np.exp(-K_avrami * (t ** n)))
            
            temp_effect_power = np.exp(beta / (R * T_kelvin))
            f_power = w * temp_effect_power * (t ** 0.3) * (1 + 0.03 * (G - 8))
            
            return f_avrami + f_power
        
        return 0.0
    
    def calculate_validation_metrics(self, data):
        """Расчет метрик валидации"""
        if self.params is None:
            return None
        
        G = data['G'].values
        T = data['T'].values
        t = data['t'].values
        f_exp = data['f_exp (%)'].values
        
        f_pred = np.array([self._evaluate_model(g, temp, time) for g, temp, time in zip(G, T, t)])
        
        residuals = f_pred - f_exp
        relative_errors = (residuals / np.maximum(f_exp, 0.1)) * 100
        
        valid_mask = np.isfinite(relative_errors) & (f_exp > 0.1)
        f_exp_valid = f_exp[valid_mask]
        f_pred_valid = f_pred[valid_mask]
        residuals_valid = residuals[valid_mask]
        relative_errors_valid = relative_errors[valid_mask]
        
        if len(f_exp_valid) == 0:
            return None
            
        mae = np.mean(np.abs(residuals_valid))
        rmse = np.sqrt(mean_squared_error(f_exp_valid, f_pred_valid))
        mape = np.mean(np.abs(relative_errors_valid))
        r2 = r2_score(f_exp_valid, f_pred_valid)
        
        validation_results = {
            'data': data.copy(),
            'predictions': f_pred,
            'residuals': residuals,
            'relative_errors': relative_errors,
            'metrics': {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
        }
        
        return validation_results

    def calculate_working_range_metrics(self, data):
        """Специальные метрики для рабочего диапазона 580-630°C"""
        if self.params is None:
            return None
        
        working_data = data[(data['T'] >= 580) & (data['T'] <= 630)]
        
        if len(working_data) == 0:
            return None
            
        G = working_data['G'].values
        T = working_data['T'].values
        t = working_data['t'].values
        f_exp = working_data['f_exp (%)'].values
        
        f_pred = np.array([self._evaluate_model(g, temp, time) for g, temp, time in zip(G, T, t)])
        
        residuals = f_pred - f_exp
        relative_errors = (residuals / np.maximum(f_exp, 0.1)) * 100
        
        working_metrics = {
            'MAE': np.mean(np.abs(residuals)),
            'RMSE': np.sqrt(mean_squared_error(f_exp, f_pred)),
            'MAPE': np.mean(np.abs(relative_errors)),
            'R2': r2_score(f_exp, f_pred),
            'MaxError': np.max(np.abs(residuals)),
            'DataPoints': len(working_data)
        }
        
        return working_metrics

def read_uploaded_file(uploaded_file):
    """Чтение загруженного файла"""
    try:
        # Сначала пробуем стандартное чтение
        if uploaded_file.name.endswith('.csv'):
            try:
                data = pd.read_csv(uploaded_file, decimal=',', encoding='utf-8')
            except:
                try:
                    data = pd.read_csv(uploaded_file, decimal=',', encoding='cp1251')
                except:
                    data = pd.read_csv(uploaded_file, decimal='.', encoding='utf-8')
        else:
            try:
                # Пробуем прочитать как простой файл
                if uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    data = pd.read_excel(uploaded_file, engine='xlrd')
                
                # СНАЧАЛА проверяем что данные загрузились
                if data is not None and len(data) > 0:
                    # ТЕПЕРЬ нормализуем названия колонок
                    data_normalized = DataValidator.normalize_column_names(data)
                    # Проверяем есть ли нужные колонки после нормализации
                    if not all(col in data_normalized.columns for col in ['G', 'T', 't', 'f_exp (%)']):
                        # Если нет нужных колонок, пробуем парсить как сложный файл
                        st.warning("⚠️ Обнаружен сложный формат данных. Применяем специальный парсер...")
                        data = ComplexDataParser.extract_all_data(uploaded_file)
                    else:
                        data = data_normalized
                else:
                    # Если данные пустые, пробуем сложный парсер
                    st.warning("⚠️ Данные не загрузились. Пробуем специальный парсер...")
                    data = ComplexDataParser.extract_all_data(uploaded_file)
                    
            except Exception as e:
                st.warning(f"⚠️ Стандартное чтение не удалось: {e}. Пробуем специальный парсер...")
                data = ComplexDataParser.extract_all_data(uploaded_file)
        
        return data
        
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {e}")
        return None

def main():
    # Инициализация сессии
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'excluded_points' not in st.session_state:
        st.session_state.excluded_points = set()
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "ensemble"
    
    # Создание вкладок
    tab1, tab2, tab3 = st.tabs(["📊 Данные и модель", "🧮 Калькулятор", "📈 Валидация модели"])
    
    # Боковая панель
    st.sidebar.header("🎯 Настройки модели")
    
    # Выбор модели
    model_type = st.sidebar.selectbox(
        "Выберите модель",
        ["avrami_saturation", "power_law", "logistic", "ensemble"],
        format_func=lambda x: {
            "avrami_saturation": "Аврами с насыщением",
            "power_law": "Степенная модель",
            "logistic": "Логистическая модель", 
            "ensemble": "Ансамблевая модель"
        }[x],
        key="model_selector"
    )
    
    # Пример данных
    sample_data = pd.DataFrame({
        'G': [8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        'T': [600, 600, 600, 600, 650, 650, 650, 650, 600, 600, 600, 600, 650, 650, 650, 650, 600, 600, 600, 600, 650, 650, 650, 650, 700, 700],
        't': [2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000],
        'f_exp (%)': [1.76, 0.68, 0.94, 1.09, 0.67, 1.2, 1.48, 1.13, 0.87, 1.28, 2.83, 3.25, 1.88, 2.29, 3.25, 2.89, 1.261, 2.04, 2.38, 3.3, 3.2, 4.26, 5.069, 5.41, 3.3, 5.0]
    })
    
    # Загрузка данных
    st.sidebar.header("📊 Загрузка данных")
    
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите файл с экспериментальными данными",
        type=['csv', 'xlsx', 'xls']
    )
    
    # Кнопка для принудительного парсинга сложных файлов
    if uploaded_file is not None and uploaded_file.name.endswith(('.xlsx', '.xls')):
        if st.sidebar.button("🔧 Принудительный парсинг сложного файла"):
            with st.spinner("Парсим сложную структуру файла..."):
                data = ComplexDataParser.extract_all_data(uploaded_file)
                if data is not None and len(data) > 0:
                    data = DataValidator.normalize_column_names(data)
                    is_valid, message = DataValidator.validate_data(data)
                    if is_valid:
                        data['f_exp (%)'] = data['f_exp (%)'].round(3)
                        st.session_state.current_data = data
                        st.sidebar.success(f"✅ Извлечено {len(data)} записей!")
                        st.rerun()
                else:
                    st.sidebar.error("❌ Не удалось извлечь данные из файла")

    # ОСНОВНАЯ ЗАГРУЗКА
    if uploaded_file is not None:
        data = read_uploaded_file(uploaded_file)
        if data is not None and len(data) > 0:
            # Нормализуем названия колонок ЕСЛИ еще не нормализованы
            if 'G' not in data.columns or 'T' not in data.columns or 't' not in data.columns or 'f_exp (%)' not in data.columns:
                data = DataValidator.normalize_column_names(data)
            
            is_valid, message = DataValidator.validate_data(data)
            if is_valid:
                data['f_exp (%)'] = data['f_exp (%)'].round(3)
                st.session_state.current_data = data
                st.sidebar.success("✅ Данные успешно загружены!")
            else:
                st.sidebar.error(f"❌ Ошибка валидации: {message}")
        else:
            st.sidebar.error("❌ Не удалось загрузить данные из файла")

    if st.session_state.current_data is None:
        st.session_state.current_data = sample_data

    # ВКЛАДКА 1: Данные и модель
    with tab1:
        st.header("📊 Управление данными")
        
        st.info("💡 **Снимите галочки с точек, которые хотите исключить из анализа**")
        
        # Создаем копию данных с чекбоксами
        display_data = st.session_state.current_data.copy()
        display_data['№'] = range(1, len(display_data) + 1)
        display_data['Использовать'] = [i not in st.session_state.excluded_points for i in range(len(display_data))]
        
        # Показываем таблицу с чекбоксами
        edited_df = st.data_editor(
            display_data,
            column_config={
                "№": st.column_config.NumberColumn(width="small"),
                "Использовать": st.column_config.CheckboxColumn(
                    width="small",
                    help="Снимите галочку чтобы исключить точку"
                ),
                "G": st.column_config.NumberColumn(width="small"),
                "T": st.column_config.NumberColumn(width="small"),
                "t": st.column_config.NumberColumn(width="small"),
                "f_exp (%)": st.column_config.NumberColumn(format="%.3f", width="small")
            },
            column_order=["№", "Использовать", "G", "T", "t", "f_exp (%)"],
            use_container_width=True,
            height=400
        )
        
        # Обновляем список исключенных точек
        new_excluded = set()
        for i, used in enumerate(edited_df['Использовать']):
            if not used:
                new_excluded.add(i)
        
        if new_excluded != st.session_state.excluded_points:
            st.session_state.excluded_points = new_excluded
            st.session_state.analyzer = None
            st.session_state.validation_results = None
        
        # Статистика
        total = len(display_data)
        excluded = len(st.session_state.excluded_points)
        included = total - excluded
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего точек", total)
        col2.metric("Используется", included, delta=f"-{excluded}" if excluded > 0 else None)
        col3.metric("Исключено", excluded)
        
        if excluded > 0:
            st.warning(f"Исключенные точки: {[i+1 for i in sorted(st.session_state.excluded_points)]}")
            if st.button("🔄 Включить все точки"):
                st.session_state.excluded_points = set()
                st.rerun()
        
        # Подготовка данных для анализа
        analysis_data = st.session_state.current_data.copy()
        if st.session_state.excluded_points:
            analysis_data = analysis_data.drop(list(st.session_state.excluded_points)).reset_index(drop=True)
        
        # Подбор модели
        st.header("🎯 Подбор параметров модели")
        
        model_names = {
            'avrami_saturation': 'Аврами с насыщением', 
            'power_law': 'Степенная', 
            'logistic': 'Логистическая', 
            'ensemble': 'Ансамблевая'
        }
        st.write(f"**Выбрана модель:** {model_names[model_type]}")
        
        if st.button("🚀 Обучить модель", type="primary", use_container_width=True):
            if len(analysis_data) < 5:
                st.error("❌ Слишком мало данных для обучения. Нужно как минимум 5 точек.")
            else:
                analyzer = SigmaPhaseAnalyzer()
                with st.spinner("Подбираем параметры модели..."):
                    success = analyzer.fit_model(analysis_data, model_type)
                
                if success:
                    st.session_state.analyzer = analyzer
                    validation_results = analyzer.calculate_validation_metrics(analysis_data)
                    st.session_state.validation_results = validation_results
                    st.success(f"✅ Модель обучена! R² = {analyzer.R2:.4f}")
                    st.rerun()
        
        # Показ результатов
        if st.session_state.analyzer is not None:
            analyzer = st.session_state.analyzer
            
            st.subheader("📈 Параметры модели")
            if analyzer.model_type == "ensemble":
                f_max, K0, Q, n, alpha, w, beta = analyzer.params
                cols = st.columns(4)
                cols[0].metric("f_max", f"{f_max:.3f}%")
                cols[1].metric("K₀", f"{K0:.2e}")
                cols[2].metric("Q", f"{Q/1000:.1f} кДж/моль")
                cols[3].metric("n", f"{n:.3f}")
                cols[0].metric("α", f"{alpha:.3f}")
                cols[1].metric("w", f"{w:.3f}")
                cols[2].metric("β", f"{beta:.0f}")
            
            st.subheader("📊 Метрики качества")
            col1, col2 = st.columns(2)
            col1.metric("R²", f"{analyzer.R2:.4f}")
            col2.metric("RMSE", f"{analyzer.rmse:.3f}%")
            
            # Метрики для рабочего диапазона
            st.subheader("📊 Метрики в рабочем диапазоне (580-630°C)")
            working_metrics = analyzer.calculate_working_range_metrics(analysis_data)
            
            if working_metrics:
                cols = st.columns(3)
                cols[0].metric("Точек в диапазоне", working_metrics['DataPoints'])
                cols[1].metric("R² рабоч.", f"{working_metrics['R2']:.4f}")
                cols[2].metric("RMSE рабоч.", f"{working_metrics['RMSE']:.3f}%")
            else:
                st.info("Нет данных в рабочем диапазоне 580-630°C для расчета метрик")
            
            st.subheader("🧮 Формула модели")
            st.markdown(analyzer.final_formula)

    # ВКЛАДКА 2: Калькулятор
    with tab2:
        st.header("🧮 Калькулятор температуры")
        
        if st.session_state.analyzer is not None:
            analyzer = st.session_state.analyzer
            
            col1, col2, col3 = st.columns(3)
            with col1:
                G_input = st.number_input("Номер зерна (G)", value=8.0, min_value=-3.0, max_value=14.0, step=0.1)
            with col2:
                sigma_input = st.number_input("Содержание сигма-фазы (%)", value=2.0, min_value=0.1, max_value=20.0, step=0.1)
            with col3:
                t_input = st.number_input("Время (ч)", value=4000, min_value=100, max_value=500000, step=100)
            
            if st.button("🔍 Рассчитать температуру", use_container_width=True):
                try:
                    T_pred = analyzer.predict_temperature(G_input, sigma_input, t_input)
                    
                    # Оценка достоверности предсказания
                    if 580 <= T_pred <= 630:
                        st.success(f"**Расчетная температура эксплуатации:** {T_pred:.1f}°C")
                        st.info("✅ Температура в оптимальном рабочем диапазоне")
                    elif 550 <= T_pred < 580:
                        st.success(f"**Расчетная температура эксплуатации:** {T_pred:.1f}°C")
                        st.warning("⚠️ Температура близка к нижнему пределу образования сигма-фазы")
                    else:
                        st.success(f"**Расчетная температура эксплуатации:** {T_pred:.1f}°C")
                        st.warning("⚠️ Температура вне типичного рабочего диапазона")
                        
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")
        else:
            st.info("👆 Сначала обучите модель на вкладке 'Данные и модель'")

    # ВКЛАДКА 3: Валидация
    with tab3:
        st.header("📈 Валидация модели")
        
        if st.session_state.analyzer is not None and st.session_state.validation_results is not None:
            analyzer = st.session_state.analyzer
            validation = st.session_state.validation_results
            
            # Метрики
            metrics = validation['metrics']
            st.subheader("📊 Метрики качества")
            cols = st.columns(4)
            cols[0].metric("R²", f"{metrics['R2']:.4f}")
            cols[1].metric("MAE", f"{metrics['MAE']:.3f}%")
            cols[2].metric("RMSE", f"{metrics['RMSE']:.3f}%")
            cols[3].metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            # График
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=validation['data']['f_exp (%)'],
                y=validation['predictions'],
                mode='markers',
                name='Предсказания',
                marker=dict(size=10, color='blue')
            ))
            max_val = max(validation['data']['f_exp (%)'].max(), validation['predictions'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                name='Идеально',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title='Предсказание vs Эксперимент',
                xaxis_title='Эксперимент (%)',
                yaxis_title='Модель (%)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Таблица
            st.subheader("📋 Детальное сравнение")
            comp_df = validation['data'].copy()
            comp_df['f_pred (%)'] = validation['predictions'].round(3)
            comp_df['Ошибка (%)'] = validation['residuals'].round(3)
            comp_df['Отн. ошибка (%)'] = validation['relative_errors'].round(1)
            st.dataframe(comp_df, use_container_width=True)
            
        else:
            st.info("👆 Сначала обучите модель на вкладке 'Данные и модель'")

if __name__ == "__main__":
    main()
