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

class DataValidator:
    """Класс для валидации и нормализации данных"""
    
    @staticmethod
    def normalize_column_names(df):
        """Нормализует названия колонок к стандартному формату"""
        column_mapping = {
            # Русские варианты
            'Номер_зерна': 'G', 'Номер зерна': 'G', 'Зерно': 'G',
            'Температура': 'T', 'Температура_C': 'T', 'Температура °C': 'T',
            'Время': 't', 'Время_ч': 't', 'Время, ч': 't',
            'Сигма_фаза': 'f_exp (%)', 'Сигма-фаза': 'f_exp (%)', 
            'Сигма_фаза_%': 'f_exp (%)', 'Сигма фаза': 'f_exp (%)',
            
            # Английские варианты
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
        
        # Проверяем наличие колонок
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Отсутствуют обязательные колонки: {missing_columns}"
        
        # Проверяем типы данных
        try:
            df['G'] = pd.to_numeric(df['G'], errors='coerce')
            df['T'] = pd.to_numeric(df['T'], errors='coerce')
            df['t'] = pd.to_numeric(df['t'], errors='coerce')
            df['f_exp (%)'] = pd.to_numeric(df['f_exp (%)'], errors='coerce')
        except Exception as e:
            return False, f"Ошибка преобразования типов данных: {e}"
        
        # Проверяем на NaN значения
        if df[required_columns].isna().any().any():
            return False, "Обнаружены пустые или некорректные значения в данных"
        
        # Проверяем диапазоны значений
        if (df['G'] < -3).any() or (df['G'] > 14).any():
            return False, "Номер зерна должен быть в диапазоне от -3 до 14"
        
        if (df['T'] < 500).any() or (df['T'] > 1000).any():
            st.warning("⚠️ Некоторые температуры выходят за typical диапазон 500-1000°C")
        
        if (df['f_exp (%)'] < 0).any() or (df['f_exp (%)'] > 50).any():
            st.warning("⚠️ Некоторые значения содержания сигма-фазы выходят за typical диапазон 0-50%")
        
        # Проверяем диапазон времени
        DataValidator.validate_time_range(df['t'])
        
        return True, "Данные валидны"
    
    @staticmethod
    def validate_time_range(t_values):
        """Проверка диапазона времени эксплуатации"""
        max_time = 500000  # часов
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
            
            # Усовершенствованная модель с плотностью границ
            initial_guess = [1e10, 200000, 10000, 1.0, 550.0, 900.0, 0.1]
            bounds = (
                [1e5, 100000, 0, 0.1, 500.0, 850.0, 0.0],
                [1e15, 500000, 50000, 4.0, 600.0, 950.0, 1.0]
            )
            
            self.params, _ = curve_fit(
                lambda x, K0, a, b, n, T_min, T_max, alpha: 
                self.sigma_phase_model_advanced([K0, a, b, n, T_min, T_max, alpha], G, T_kelvin, t),
                np.arange(len(G)), sigma_exp,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            # Расчет метрик качества
            sigma_pred = self.sigma_phase_model_advanced(self.params, G, T_kelvin, t) * 100
            sigma_exp_percent = sigma_exp * 100
            self.R2 = r2_score(sigma_exp_percent, sigma_pred)
            self.rmse = np.sqrt(mean_squared_error(sigma_exp_percent, sigma_pred))
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка при подгонке модели: {str(e)}")
            return False
    
    def sigma_phase_model_advanced(self, params, G, T, t):
        """Усовершенствованная модель с учетом плотности границ зерен"""
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
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=contamination, random_state=42)
            labels = clf.fit_predict(features)
            outlier_flags = labels == -1
        
        outlier_data = data[outlier_flags]
        clean_data = data[~outlier_flags]
        
        return outlier_data, clean_data
    
    def predict_temperature(self, G, sigma_percent, t):
        """Предсказание температуры по известным параметрам"""
        if self.params is None:
            raise ValueError("Модель не обучена!")
        
        R = 8.314
        sigma = sigma_percent / 100.0
        
        try:
            K0, a, b, n, T_sigma_min, T_sigma_max, alpha = self.params
            
            # Упрощенный расчет для единичного значения
            grain_boundary_factor = GrainSizeConverter.calculate_activation_energy_factor(G)
            Q_effective = (a + b * G) * (1 + alpha * (grain_boundary_factor - 1))
            
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

    def predict_temperature_improved(self, G, sigma_percent, t):
        """Улучшенное предсказание температуры для больших времен эксплуатации"""
        if self.params is None:
            raise ValueError("Модель не обучена!")
        
        R = 8.314
        sigma = sigma_percent / 100.0
        
        try:
            K0, a, b, n, T_sigma_min, T_sigma_max, alpha = self.params
            
            # Упрощенный расчет для единичного значения
            grain_boundary_factor = GrainSizeConverter.calculate_activation_energy_factor(G)
            Q_effective = (a + b * G) * (1 + alpha * (grain_boundary_factor - 1))
            
            # Защита от численных ошибок при очень больших временах
            if sigma >= 1.0:
                sigma = 0.999  # Ограничиваем для избежания log(0)
            
            term = -np.log(1 - sigma) / (K0 * (t ** n))
            
            if term <= 0:
                # Для очень больших времен возвращаем минимальную температуру
                return T_sigma_min
            
            T_kelvin = -Q_effective / (R * np.log(term))
            T_celsius = T_kelvin - 273.15
            
            # Применяем температурные ограничения
            if T_celsius < T_sigma_min:
                return T_sigma_min
            elif T_celsius > T_sigma_max:
                return T_sigma_max
            else:
                return T_celsius
                
        except (ValueError, ZeroDivisionError) as e:
            # При численных ошибках возвращаем граничное значение
            st.warning(f"⚠️ При расчете для t={t} ч возникла численная ошибка. Используется минимальная температура.")
            return T_sigma_min

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
        'G': [3, 3, 5, 5, 8, 8, 9, 9, 3, 5, 8],
        'T': [600, 650, 600, 700, 650, 700, 600, 700, 600, 650, 750],
        't': [2000, 4000, 4000, 2000, 6000, 4000, 8000, 6000, 2000, 4000, 4000],
        'f_exp (%)': [5.2, 12.5, 8.1, 15.3, 18.7, 25.1, 22.4, 35.2, 12.8, 25.6, 2.1]
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
        st.info("👈 Пожалуйста, загрузите файл с данными или используйте пример данных")
        st.session_state.current_data = sample_data
    
    # Показ загруженных данных
    st.header("📊 Экспериментальные данные")
    
    # Показываем информацию о колонках
    if st.session_state.current_data is not None:
        st.info(f"**Структура данных:** {len(st.session_state.current_data)} строк × {len(st.session_state.current_data.columns)} колонок")
        st.write("**Загруженные колонки:**", list(st.session_state.current_data.columns))
    
    # Редактирование данных
    if st.session_state.current_data is not None:
        edited_data = st.data_editor(
            st.session_state.current_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "f_exp (%)": st.column_config.NumberColumn(format="%.3f"),
                "G": st.column_config.NumberColumn(format="%d"),
                "T": st.column_config.NumberColumn(format="%.1f"),
                "t": st.column_config.NumberColumn(format="%d")
            }
        )
        
        # Округляем значения после редактирования
        if 'f_exp (%)' in edited_data.columns:
            edited_data['f_exp (%)'] = edited_data['f_exp (%)'].round(3)
        
        if not edited_data.equals(st.session_state.current_data):
            st.session_state.current_data = edited_data
            st.session_state.analyzer = None
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
    
    # Кнопка подбора параметров модели
    if st.button("🎯 Подобрать параметры модели", use_container_width=True):
        if st.session_state.current_data is not None and all(col in st.session_state.current_data.columns for col in ['G', 'T', 't', 'f_exp (%)']):
            analyzer = SigmaPhaseAnalyzer()
            
            with st.spinner("Идет подбор параметров модели..."):
                success = analyzer.fit_model(
                    st.session_state.current_data, 
                    remove_outliers=remove_outliers
                )
            
            if success:
                st.session_state.analyzer = analyzer
                st.success("✅ Модель успешно обучена!")
                st.rerun()
        else:
            st.error("❌ Для подбора модели необходимы колонки: G, T, t, f_exp (%)")
    
    # Показ результатов модели
    if st.session_state.analyzer is not None:
        analyzer = st.session_state.analyzer
        
        # Параметры модели
        st.subheader("📈 Параметры модели")
        
        if analyzer.params is not None:
            K0, a, b, n, T_sigma_min, T_sigma_max, alpha = analyzer.params
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("K₀", f"{K0:.2e}")
                st.metric("a", f"{a:.2f}")
            with col2:
                st.metric("b", f"{b:.2f}")
                st.metric("n", f"{n:.3f}")
            with col3:
                st.metric("α", f"{alpha:.3f}")
            
            st.metric("Температурный диапазон", f"{T_sigma_min:.1f}°C - {T_sigma_max:.1f}°C")
            
            # Метрики качества
            st.subheader("📊 Метрики качества модели")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R²", f"{analyzer.R2:.4f}")
            with col2:
                st.metric("RMSE", f"{analyzer.rmse:.2f}%")
            
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
                                            format="%.3f")
            with col3:
                t_input = st.number_input("Время эксплуатации t (ч)", 
                                        min_value=100, max_value=500000,  # ИЗМЕНЕНО: 500000 часов
                                        value=4000, step=1000)  # Увеличили шаг для удобства

            # Информация о диапазоне
            if t_input > 100000:
                st.info("🔍 Расчет выполняется для длительной эксплуатации (свыше 100000 часов)")
            
            if st.button("🔍 Рассчитать температуру", key="calc_temp"):
                try:
                    # Используем улучшенный метод для больших времен
                    if t_input > 100000:
                        T_celsius = analyzer.predict_temperature_improved(G_input, sigma_input, t_input)
                    else:
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
