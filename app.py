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
    
    # Данные из ГОСТ 5639-82
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
    def calculate_activation_energy_factor(cls, grain_number):
        """
        Коэффициент влияния размера зерна на энергию активации
        Учитывает реальные геометрические параметры из ГОСТ
        """
        # Нормализуем относительно номера зерна 5 (базовый)
        ref_grain = 5
        Sv_ref = cls.calculate_grain_boundary_density(ref_grain)
        Sv_current = cls.calculate_grain_boundary_density(grain_number)
        
        return Sv_current / Sv_ref

class AdvancedSigmaPhaseAnalyzer:
    def __init__(self):
        self.params = None
        self.R2 = None
        self.rmse = None
        self.mape = None
        self.model_type = None
        self.creation_date = datetime.now().isoformat()
        self.final_formula = ""
        
    def fit_ensemble_model(self, data):
        """Ансамблевая модель - комбинация нескольких подходов"""
        try:
            G = data['G'].values
            T = data['T'].values + 273.15
            t = data['t'].values
            f_exp = data['f_exp (%)'].values
            
            # Параметры для ансамбля: f_max, K0, Q, n, alpha, w, beta
            initial_guess = [5.0, 1e8, 150000, 0.5, 0.1, 0.1, -20000]
            bounds = (
                [1.0, 1e5, 100000, 0.1, -1.0, 0.01, -50000],
                [15.0, 1e12, 300000, 2.0, 1.0, 1.0, -1000]
            )
            
            def model(params, G, T, t):
                f_max, K0, Q, n, alpha, w, beta = params
                R = 8.314
                
                # Компонент Аврами
                grain_effect_avrami = 1 + alpha * (G - 8)
                K_avrami = K0 * np.exp(-Q / (R * T)) * grain_effect_avrami
                f_avrami = f_max * (1 - np.exp(-K_avrami * (t ** n)))
                
                # Компонент степенного закона
                temp_effect_power = np.exp(beta / (R * T))
                f_power = w * temp_effect_power * (t ** 0.5) * (1 + 0.05 * (G - 8))
                
                # Комбинированная модель
                f_pred = f_avrami + f_power
                return np.clip(f_pred, 0, 15)
            
            self.params, _ = curve_fit(
                lambda x, f_max, K0, Q, n, alpha, w, beta: model([f_max, K0, Q, n, alpha, w, beta], G, T, t),
                np.arange(len(G)), f_exp,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            # Расчет метрик
            f_pred = model(self.params, G, T, t)
            self.R2 = r2_score(f_exp, f_pred)
            self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
            self.mape = np.mean(np.abs((f_exp - f_pred) / f_exp)) * 100
            self.model_type = "ensemble"
            
            # Генерация финальной формулы
            self._generate_final_formula()
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка в ансамблевой модели: {e}")
            return False
    
    def _generate_final_formula(self):
        """Генерация читаемой формулы модели"""
        if self.params is None:
            self.final_formula = "Модель не обучена"
            return
            
        f_max, K0, Q, n, alpha, w, beta = self.params
        
        self.final_formula = f"""
**Финальная формула ансамблевой модели:**
f(G, T, t) = f_avrami(G, T, t) + f_power(G, T, t)

где:

f_avrami(G, T, t) = {f_max:.3f} × [1 - exp(-K_avrami × t^{n:.3f})]
K_avrami = {K0:.3e} × exp(-{Q/1000:.1f} кДж/моль / (R × T)) × [1 + {alpha:.3f} × (G - 8)]

f_power(G, T, t) = {w:.3f} × exp({beta:.0f} / (R × T)) × t^0.5 × [1 + 0.05 × (G - 8)]

R = 8.314 Дж/(моль·К) - универсальная газовая постоянная
T - температура в Кельвинах (T[°C] + 273.15)
        
**Расшифровка параметров:**
- `f_max = {f_max:.3f} %` - максимальное содержание сигма-фазы
- `K0 = {K0:.3e}` - предэкспоненциальный множитель
- `Q = {Q/1000:.1f} кДж/моль` - энергия активации
- `n = {n:.3f}` - показатель степени в модели Аврами
- `α = {alpha:.3f}` - коэффициент влияния размера зерна
- `w = {w:.3f}` - вес степенного компонента
- `β = {beta:.0f}` - температурный коэффициент в степенном законе
"""
    
    def predict_temperature(self, G, sigma_percent, t, method="bisection"):
        """Предсказание температуры разными методами"""
        if self.params is None:
            raise ValueError("Модель не обучена!")
        
        sigma = sigma_percent
        
        if method == "bisection":
            return self._predict_temperature_bisection(G, sigma, t)
        else:
            return self._predict_temperature_analytic(G, sigma, t)
    
    def _predict_temperature_bisection(self, G, sigma, t, tol=1.0, max_iter=100):
        """Бисекционный поиск температуры"""
        T_min, T_max = 500, 900  # Реалистичный диапазон
        
        for i in range(max_iter):
            T_mid = (T_min + T_max) / 2
            f_pred = self._evaluate_model(G, T_mid, t)
            
            if abs(f_pred - sigma) < tol:
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
        f_max, K0, Q, n, alpha, w, beta = self.params
        R = 8.314
        
        # Аврами компонент
        grain_effect_avrami = 1 + alpha * (G - 8)
        K_avrami = K0 * np.exp(-Q / (R * T_kelvin)) * grain_effect_avrami
        f_avrami = f_max * (1 - np.exp(-K_avrami * (t ** n)))
        
        # Степенной компонент
        temp_effect_power = np.exp(beta / (R * T_kelvin))
        f_power = w * temp_effect_power * (t ** 0.5) * (1 + 0.05 * (G - 8))
        
        return f_avrami + f_power
    
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
        
        # Фильтрация бесконечных значений
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
    """Чтение загруженного файла с обработкой ошибок"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Пробуем разные кодировки и разделители
            try:
                data = pd.read_csv(uploaded_file, decimal=',', encoding='utf-8')
            except:
                try:
                    data = pd.read_csv(uploaded_file, decimal=',', encoding='cp1251')
                except:
                    data = pd.read_csv(uploaded_file, decimal='.', encoding='utf-8')
        else:
            # Для Excel файлов
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    try:
                        data = pd.read_excel(uploaded_file, engine='openpyxl')
                    except ImportError:
                        st.error("❌ Для чтения .xlsx файлов требуется библиотека openpyxl")
                        st.info("Установите её командой: `pip install openpyxl`")
                        return None
                else:  # .xls
                    try:
                        data = pd.read_excel(uploaded_file, engine='xlrd')
                    except ImportError:
                        st.error("❌ Для чтения .xls файлов требуется библиотека xlrd")
                        st.info("Установите её командой: `pip install xlrd`")
                        return None
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
    
    # Создание вкладок
    tab1, tab2, tab3 = st.tabs(["📊 Данные и модель", "🧮 Калькулятор", "📈 Валидация модели"])
    
    # Боковая панель
    st.sidebar.header("📁 Управление проектом")
    
    # Загрузка/сохранение проекта
    if st.session_state.analyzer is not None and st.session_state.current_data is not None:
        if st.sidebar.button("💾 Сохранить проект"):
            project_data = {
                'analyzer': st.session_state.analyzer.__dict__,
                'current_data': st.session_state.current_data.to_dict()
            }
            
            project_json = json.dumps(project_data, indent=2)
            st.sidebar.download_button(
                label="Скачать проект",
                data=project_json,
                file_name=f"sigma_phase_project_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    # Настройки обработки выбросов
    st.sidebar.header("🎯 Настройки обработки выбросов")
    remove_outliers = st.sidebar.checkbox("Удалять выбросы", value=True)
    
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
        type=['csv', 'xlsx', 'xls'],
        help="Поддерживаемые форматы: CSV, Excel (.xlsx, .xls)"
    )
    
    # Обработка загруженного файла
    if uploaded_file is not None:
        data = read_uploaded_file(uploaded_file)
        
        if data is not None:
            # Нормализуем названия колонок
            data = DataValidator.normalize_column_names(data)
            
            # Валидируем данные
            is_valid, message = DataValidator.validate_data(data)
            
            if is_valid:
                # Округляем значения до тысячных
                data['f_exp (%)'] = data['f_exp (%)'].round(3)
                st.session_state.current_data = data
                st.sidebar.success("✅ Данные успешно загружены и валидированы!")
                st.sidebar.info(f"Загружено {len(data)} строк")
            else:
                st.sidebar.error(f"❌ {message}")
                # Показываем какие колонки есть в файле
                st.sidebar.info(f"Найденные колонки: {list(data.columns)}")
    
    # Если данных нет, используем пример
    if st.session_state.current_data is None:
        st.session_state.current_data = sample_data

    # ВКЛАДКА 1: Данные и модель
    with tab1:
        st.header("📊 Экспериментальные данные")
        
        # Показываем информацию о колонках
        if st.session_state.current_data is not None:
            st.info(f"**Структура данных:** {len(st.session_state.current_data)} строк × {len(st.session_state.current_data.columns)} колонок")
            st.write("**Загруженные колонки:**", list(st.session_state.current_data.columns))
        
        # Создаем копию данных для редактирования с чекбоксами
        display_data = st.session_state.current_data.copy()
        display_data['Включить'] = [i not in st.session_state.excluded_points for i in range(len(display_data))]
        
        # Редактор данных с возможностью исключения точек
        edited_data = st.data_editor(
            display_data,
            column_config={
                "Включить": st.column_config.CheckboxColumn(
                    "Включить в анализ",
                    help="Снимите галочку чтобы исключить точку из анализа"
                ),
                "f_exp (%)": st.column_config.NumberColumn(format="%.3f"),
                "G": st.column_config.NumberColumn(format="%d"),
                "T": st.column_config.NumberColumn(format="%.1f"),
                "t": st.column_config.NumberColumn(format="%d")
            },
            disabled=["G", "T", "t", "f_exp (%)"],
            use_container_width=True
        )
        
        # Обновляем список исключенных точек
        new_excluded = set()
        for i, included in enumerate(edited_data['Включить']):
            if not included:
                new_excluded.add(i)
        
        if new_excluded != st.session_state.excluded_points:
            st.session_state.excluded_points = new_excluded
            st.session_state.analyzer = None
            st.session_state.validation_results = None
            st.rerun()
        
        # Статистика данных
        total_points = len(st.session_state.current_data)
        excluded_count = len(st.session_state.excluded_points)
        included_count = total_points - excluded_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Всего точек", total_points)
        with col2:
            st.metric("Включено в анализ", included_count)
        with col3:
            st.metric("Исключено", excluded_count)
        
        if excluded_count > 0:
            st.info(f"Исключенные точки: {sorted(st.session_state.excluded_points)}")
            
            if st.button("🔄 Включить все точки", key="include_all"):
                st.session_state.excluded_points = set()
                st.session_state.analyzer = None
                st.session_state.validation_results = None
                st.rerun()
        
        # Анализ данных
        st.header("🔍 Анализ данных")
        
        if st.session_state.current_data is not None and 'G' in st.session_state.current_data.columns:
            # Информация о зернах в данных
            unique_grain_numbers = sorted(st.session_state.current_data['G'].unique())
            
            st.subheader("📐 Характеристики зерен в данных")
            cols = st.columns(min(5, len(unique_grain_numbers)))
            
            for i, grain_num in enumerate(unique_grain_numbers):
                with cols[i % 5]:
                    diameter = GrainSizeConverter.grain_number_to_diameter(grain_num)
                    boundary_density = GrainSizeConverter.calculate_grain_boundary_density(grain_num)
                    activation_factor = GrainSizeConverter.calculate_activation_energy_factor(grain_num)
                    
                    st.metric(
                        f"G = {grain_num}",
                        f"{diameter*1000:.1f} мкм",
                        f"Плотность: {boundary_density:.0f} мм²/мм³"
                    )
                    st.caption(f"Коэф. активации: {activation_factor:.3f}")
        
        # Подбор модели
        st.header("🎯 Подбор параметров модели")
        
        # Подготовка данных для анализа (исключаем выбранные точки)
        analysis_data = st.session_state.current_data.copy()
        if st.session_state.excluded_points:
            analysis_data = analysis_data.drop(list(st.session_state.excluded_points)).reset_index(drop=True)
        
        if st.button("🚀 Запустить подбор параметров", use_container_width=True):
            if analysis_data is not None and all(col in analysis_data.columns for col in ['G', 'T', 't', 'f_exp (%)']):
                analyzer = AdvancedSigmaPhaseAnalyzer()
                
                with st.spinner("Подбираем параметры ансамблевой модели..."):
                    success = analyzer.fit_ensemble_model(analysis_data)
                
                if success:
                    st.session_state.analyzer = analyzer
                    validation_results = analyzer.calculate_validation_metrics(analysis_data)
                    st.session_state.validation_results = validation_results
                    
                    st.success(f"✅ Модель успешно обучена! R² = {analyzer.R2:.4f}")
                    st.rerun()
            else:
                st.error("❌ Для подбора модели необходимы колонки: G, T, t, f_exp (%)")
        
        # Показ результатов модели
        if st.session_state.analyzer is not None:
            analyzer = st.session_state.analyzer
            
            # Параметры модели
            st.subheader("📈 Параметры модели")
            
            if analyzer.params is not None:
                f_max, K0, Q, n, alpha, w, beta = analyzer.params
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("f_max", f"{f_max:.3f}%")
                    st.metric("K₀", f"{K0:.2e}")
                with col2:
                    st.metric("Q", f"{Q/1000:.1f} кДж/моль")
                    st.metric("n", f"{n:.3f}")
                with col3:
                    st.metric("α", f"{alpha:.3f}")
                    st.metric("w", f"{w:.3f}")
                
                st.metric("β", f"{beta:.0f}")
                
                # Метрики качества
                st.subheader("📊 Метрики качества модели")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R²", f"{analyzer.R2:.4f}")
                with col2:
                    st.metric("RMSE", f"{analyzer.rmse:.2f}%")

    # ВКЛАДКА 2: Калькулятор
    with tab2:
        st.header("🧮 Калькулятор температуры эксплуатации")
        
        if st.session_state.analyzer is not None:
            analyzer = st.session_state.analyzer
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                G_input = st.number_input("Номер зерна (G)", 
                                        min_value=-3.0, max_value=14.0, 
                                        value=8.0, step=0.1)
            with col2:
                sigma_input = st.number_input("Содержание сигма-фазы f_exp (%)", 
                                            min_value=0.0, max_value=50.0,
                                            value=2.0, step=0.1,
                                            format="%.3f")
            with col3:
                t_input = st.number_input("Время эксплуатации t (ч)", 
                                        min_value=100, max_value=500000,
                                        value=4000, step=1000)

            # Информация о диапазоне
            if t_input > 100000:
                st.info("🔍 Расчет выполняется для длительной эксплуатации (свыше 100000 часов)")
            
            if st.button("🔍 Рассчитать температуру", key="calc_temp"):
                try:
                    T_celsius = analyzer.predict_temperature(G_input, sigma_input, t_input)
                    
                    if T_celsius is not None:
                        st.success(f"""
                        ### Результат расчета:
                        - **Температура эксплуатации:** {T_celsius:.1f}°C
                        - При номере зерна: {G_input}
                        - Содержании сигма-фазы: {sigma_input:.3f}%
                        - Наработке: {t_input} ч
                        """)
                        
                        # Дополнительная информация для больших времен
                        if t_input > 200000:
                            st.info("💡 **Примечание:** Расчет для времени эксплуатации свыше 200000 часов требует осторожной интерпретации результатов")
                        
                    else:
                        st.error("Не удалось рассчитать температуру. Проверьте входные параметры.")
                        
                except Exception as e:
                    st.error(f"Ошибка при расчете: {str(e)}")
        else:
            st.info("👆 Сначала обучите модель на вкладке 'Данные и модель'")

    # ВКЛАДКА 3: Валидация модели
    with tab3:
        st.header("📈 Валидация модели")
        
        if st.session_state.analyzer is not None and st.session_state.validation_results is not None:
            analyzer = st.session_state.analyzer
            validation = st.session_state.validation_results
            
            # Метрики валидации
            st.subheader("📊 Метрики качества модели")
            metrics = validation['metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{metrics['R2']:.4f}")
                st.metric("MAE", f"{metrics['MAE']:.3f}%")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.3f}%")
            with col3:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            with col4:
                st.metric("Количество точек", f"{len(validation['data'])}")
            
            # Оценка качества
            if metrics['MAPE'] < 15:
                st.success("✅ Отличное качество модели!")
            elif metrics['MAPE'] < 25:
                st.warning("⚠️ Удовлетворительное качество модели")
            else:
                st.error("❌ Попробуйте исключить больше точек или проверить данные")
            
            # Финальная формула
            st.subheader("🧮 Финальная формула модели")
            st.markdown(analyzer.final_formula)
            
            # Таблица сравнения
            st.subheader("📋 Сравнение экспериментальных и расчетных значений")
            
            comparison_df = validation['data'].copy()
            comparison_df['f_pred (%)'] = validation['predictions']
            comparison_df['Абс. ошибка (%)'] = validation['residuals']
            comparison_df['Отн. ошибка (%)'] = validation['relative_errors']
            comparison_df['f_pred (%)'] = comparison_df['f_pred (%)'].round(3)
            comparison_df['Абс. ошибка (%)'] = comparison_df['Абс. ошибка (%)'].round(3)
            comparison_df['Отн. ошибка (%)'] = comparison_df['Отн. ошибка (%)'].round(1)
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Графики валидации
            st.subheader("📈 Графики валидации")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # График предсказания vs эксперимент
                fig1 = go.Figure()
                
                fig1.add_trace(go.Scatter(
                    x=validation['data']['f_exp (%)'],
                    y=validation['predictions'],
                    mode='markers',
                    name='Точки данных',
                    marker=dict(size=8, color='blue', opacity=0.6)
                ))
                
                # Линия идеального предсказания
                max_val = max(validation['data']['f_exp (%)'].max(), validation['predictions'].max())
                fig1.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Идеальное предсказание',
                    line=dict(color='red', dash='dash')
                ))
                
                fig1.update_layout(
                    title='Предсказание vs Эксперимент',
                    xaxis_title='Экспериментальное значение f_exp (%)',
                    yaxis_title='Расчетное значение f_pred (%)',
                    showlegend=True
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # График остатков
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=validation['predictions'],
                    y=validation['residuals'],
                    mode='markers',
                    name='Остатки',
                    marker=dict(size=8, color='green', opacity=0.6)
                ))
                
                # Нулевая линия
                fig2.add_trace(go.Scatter(
                    x=[validation['predictions'].min(), validation['predictions'].max()],
                    y=[0, 0],
                    mode='lines',
                    name='Нулевая линия',
                    line=dict(color='red', dash='dash')
                ))
                
                fig2.update_layout(
                    title='Остатки модели',
                    xaxis_title='Расчетное значение f_pred (%)',
                    yaxis_title='Остаток (f_pred - f_exp) (%)',
                    showlegend=True
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Статистика по ошибкам
            st.subheader("📈 Статистика ошибок")
            
            abs_errors = np.abs(validation['residuals'])
            rel_errors = np.abs(validation['relative_errors'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Макс. абс. ошибка", f"{abs_errors.max():.3f}%")
            with col2:
                st.metric("Макс. отн. ошибка", f"{rel_errors.max():.2f}%")
            with col3:
                st.metric("Средняя абс. ошибка", f"{abs_errors.mean():.3f}%")
            with col4:
                st.metric("Средняя отн. ошибка", f"{rel_errors.mean():.2f}%")
                
        else:
            st.info("👆 Сначала обучите модель на вкладке 'Данные и модель'")

if __name__ == "__main__":
    main()
