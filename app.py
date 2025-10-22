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
        """Подгонка выбранной модели"""
        try:
            G = data['G'].values
            T = data['T'].values + 273.15
            t = data['t'].values
            f_exp = data['f_exp (%)'].values
            
            self.model_type = model_type
            
            if model_type == "avrami_saturation":
                success = self._fit_avrami_model(G, T, t, f_exp)
            elif model_type == "power_law":
                success = self._fit_power_law_model(G, T, t, f_exp)
            elif model_type == "logistic":
                success = self._fit_logistic_model(G, T, t, f_exp)
            elif model_type == "ensemble":
                success = self._fit_ensemble_model(G, T, t, f_exp)
            else:
                st.error(f"Неизвестный тип модели: {model_type}")
                return False
                
            if success:
                self._generate_final_formula()
                
            return success
            
        except Exception as e:
            st.error(f"Ошибка при подгонке модели: {e}")
            return False
    
    def _fit_avrami_model(self, G, T, t, f_exp):
        """Модель Аврами с насыщением"""
        initial_guess = [8.0, 1e10, 200000, 1.0, 0.1]
        bounds = (
            [1.0, 1e5, 100000, 0.1, -1.0],
            [15.0, 1e15, 400000, 3.0, 1.0]
        )
        
        def model(params, G, T, t):
            f_max, K0, Q, n, alpha = params
            R = 8.314
            grain_effect = 1 + alpha * (G - 8)
            K = K0 * np.exp(-Q / (R * T)) * grain_effect
            return f_max * (1 - np.exp(-K * (t ** n)))
        
        self.params, _ = curve_fit(
            lambda x, f_max, K0, Q, n, alpha: model([f_max, K0, Q, n, alpha], G, T, t),
            np.arange(len(G)), f_exp,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000
        )
        
        f_pred = model(self.params, G, T, t)
        self.R2 = r2_score(f_exp, f_pred)
        self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
        self.mape = np.mean(np.abs((f_exp - f_pred) / f_exp)) * 100
        return True
    
    def _fit_power_law_model(self, G, T, t, f_exp):
        """Степенная модель"""
        initial_guess = [1.0, 0.1, -10000, 0.5, 0.01]
        bounds = (
            [0.1, -1.0, -50000, 0.1, -0.1],
            [10.0, 1.0, -1000, 2.0, 0.1]
        )
        
        def model(params, G, T, t):
            A, B, C, D, E = params
            R = 8.314
            temp_effect = np.exp(C / (R * T))
            time_effect = t ** D
            grain_effect = 1 + E * (G - 8)
            return A * temp_effect * time_effect * grain_effect + B
        
        self.params, _ = curve_fit(
            lambda x, A, B, C, D, E: model([A, B, C, D, E], G, T, t),
            np.arange(len(G)), f_exp,
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000
        )
        
        f_pred = model(self.params, G, T, t)
        self.R2 = r2_score(f_exp, f_pred)
        self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
        self.mape = np.mean(np.abs((f_exp - f_pred) / f_exp)) * 100
        return True
    
    def _fit_logistic_model(self, G, T, t, f_exp):
        """Логистическая модель"""
        initial_guess = [8.0, 1e-4, 1000, 0.1, -10000]
        bounds = (
            [1.0, 1e-8, 100, -1.0, -50000],
            [15.0, 1e-2, 10000, 1.0, -1000]
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
            maxfev=5000
        )
        
        f_pred = model(self.params, G, T, t)
        self.R2 = r2_score(f_exp, f_pred)
        self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
        self.mape = np.mean(np.abs((f_exp - f_pred) / f_exp)) * 100
        return True
    
    def _fit_ensemble_model(self, G, T, t, f_exp):
        """Ансамблевая модель"""
        initial_guess = [5.0, 1e8, 150000, 0.5, 0.1, 0.1, -20000]
        bounds = (
            [1.0, 1e5, 100000, 0.1, -1.0, 0.01, -50000],
            [15.0, 1e12, 300000, 2.0, 1.0, 1.0, -1000]
        )
        
        def model(params, G, T, t):
            f_max, K0, Q, n, alpha, w, beta = params
            R = 8.314
            
            # Аврами компонент
            grain_effect_avrami = 1 + alpha * (G - 8)
            K_avrami = K0 * np.exp(-Q / (R * T)) * grain_effect_avrami
            f_avrami = f_max * (1 - np.exp(-K_avrami * (t ** n)))
            
            # Степенной компонент
            temp_effect_power = np.exp(beta / (R * T))
            f_power = w * temp_effect_power * (t ** 0.5) * (1 + 0.05 * (G - 8))
            
            return f_avrami + f_power
        
        self.params, _ = curve_fit(
            lambda x, f_max, K0, Q, n, alpha, w, beta: model([f_max, K0, Q, n, alpha, w, beta], G, T, t),
            np.arange(len(G)), f_exp,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        
        f_pred = model(self.params, G, T, t)
        self.R2 = r2_score(f_exp, f_pred)
        self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
        self.mape = np.mean(np.abs((f_exp - f_pred) / f_exp)) * 100
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

f_power = {w:.3f} × exp({beta:.0f} / (R × T)) × t^0.5 × [1 + 0.05 × (G - 8)]
"""
        
        self.final_formula += "\n**R = 8.314 Дж/(моль·К) - универсальная газовая постоянная**\n**T - температура в Кельвинах (T[°C] + 273.15)**"
    
    def predict_temperature(self, G, sigma_percent, t):
        """Предсказание температуры"""
        if self.params is None:
            raise ValueError("Модель не обучена!")
        
        sigma = sigma_percent
        
        # Бисекционный поиск
        T_min, T_max = 500, 900
        
        for i in range(100):
            T_mid = (T_min + T_max) / 2
            f_pred = self._evaluate_model(G, T_mid, t)
            
            if abs(f_pred - sigma) < 1.0:
                return T_mid
            
            if f_pred < sigma:
                T_min = T_mid
            else:
                T_max = T_mid
        
        return (T_min + T_max) / 2
    
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
            f_power = w * temp_effect_power * (t ** 0.5) * (1 + 0.05 * (G - 8))
            
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
        relative_errors = (residuals / f_exp) * 100
        
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

def read_uploaded_file(uploaded_file):
    """Чтение загруженного файла"""
    try:
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
                if uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    data = pd.read_excel(uploaded_file, engine='xlrd')
            except Exception as e:
                st.error(f"❌ Ошибка чтения Excel файла: {str(e)}")
                return None
        
        return data
        
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {str(e)}")
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
    
    if uploaded_file is not None:
        data = read_uploaded_file(uploaded_file)
        if data is not None:
            data = DataValidator.normalize_column_names(data)
            is_valid, message = DataValidator.validate_data(data)
            if is_valid:
                data['f_exp (%)'] = data['f_exp (%)'].round(3)
                st.session_state.current_data = data
                st.sidebar.success("✅ Данные успешно загружены!")
    
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

        
            
