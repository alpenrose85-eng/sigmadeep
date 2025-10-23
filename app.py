import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.optimize import curve_fit
import io
import warnings
warnings.filterwarnings('ignore')

# Данные по размерам зерен из ГОСТ
GRAIN_DATA = {
    'G': [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'a_v': [1.000, 0.500, 0.250, 0.125, 0.0625, 0.0312, 0.0156, 0.00781, 0.00390, 
            0.00195, 0.00098, 0.00049, 0.000244, 0.000122, 0.000061, 0.000030, 0.000015, 0.000008],
    'd_av': [1.000, 0.707, 0.500, 0.353, 0.250, 0.177, 0.125, 0.088, 0.062, 
             0.044, 0.031, 0.022, 0.015, 0.011, 0.0079, 0.0056, 0.0039, 0.0027]
}

grain_df = pd.DataFrame(GRAIN_DATA)
grain_df['inv_sqrt_a_v'] = 1 / np.sqrt(grain_df['a_v'])
grain_df['ln_inv_sqrt_a_v'] = np.log(grain_df['inv_sqrt_a_v'])

class SigmaPhaseModel:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.r2 = None
        self.rmse = None
        self.mae = None
        
    def fit(self, X, y):
        """Линейная регрессия с использованием метода наименьших квадратов"""
        # Добавляем столбец для intercept
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Решаем нормальное уравнение: (X^T X)^{-1} X^T y
        try:
            coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
            self.intercept_ = coefficients[0]
            self.coef_ = coefficients[1:]
            
            # Расчет метрик
            y_pred = self.predict_ln_d(X)
            self.r2 = self.calculate_r2(y, y_pred)
            self.rmse = self.calculate_rmse(y, y_pred)
            self.mae = self.calculate_mae(y, y_pred)
            
        except np.linalg.LinAlgError:
            st.error("Ошибка: матрица вырождена. Проверьте данные на мультиколлинеарность.")
            return None
        
        return self
    
    def predict_ln_d(self, X):
        """Предсказание ln(d)"""
        return self.intercept_ + X @ self.coef_
    
    def calculate_r2(self, y_true, y_pred):
        """Расчет коэффициента детерминации R²"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def calculate_rmse(self, y_true, y_pred):
        """Расчет RMSE"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def calculate_mae(self, y_true, y_pred):
        """Расчет MAE"""
        return np.mean(np.abs(y_true - y_pred))
    
    def predict_temperature(self, d_sigma, time_hours, grain_size):
        """Предсказание температуры по модели"""
        if self.coef_ is None:
            raise ValueError("Модель не обучена!")
            
        # Получаем данные по зерну
        grain_info = grain_df[grain_df['G'] == grain_size]
        if len(grain_info) == 0:
            raise ValueError(f"Номер зерна {grain_size} не найден в базе данных")
            
        ln_inv_sqrt_a_v = grain_info['ln_inv_sqrt_a_v'].iloc[0]
        
        # Расчет по модели: ln(d_σ) = β₀ + β₁×ln(t) + β₂×(1/T) + β₃×ln(1/√a_v)
        # Преобразуем для получения температуры: 1/T = [ln(d_σ) - β₀ - β₁×ln(t) - β₃×ln(1/√a_v)] / β₂
        ln_d_sigma = np.log(d_sigma)
        ln_time = np.log(time_hours)
        
        numerator = ln_d_sigma - self.intercept_ - self.coef_[0] * ln_time - self.coef_[2] * ln_inv_sqrt_a_v
        inv_T = numerator / self.coef_[1]
        
        T_kelvin = 1 / inv_T
        T_celsius = T_kelvin - 273.15
        
        return T_celsius

class AdvancedSigmaPhaseModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_r2 = -np.inf
        
    def calculate_trunin_parameter(self, T_kelvin, time_hours):
        """Расчет параметра Трунина: P = T(logτ - 2logT + 26.3)"""
        return T_kelvin * (np.log10(time_hours) - 2 * np.log10(T_kelvin) + 26.3)
    
    def model1_power_law(self, params, t, T, G):
        """Степенная модель: d = A * t^m * exp(-Q/RT) * f(G)"""
        A, m, Q, p = params
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        return A * (t ** m) * np.exp(-Q / (8.314 * (T + 273.15))) * fG
    
    def model2_saturating_growth(self, params, t, T, G):
        """Модель насыщающего роста: d = d_max * [1 - exp(-k * t^n * exp(-Q/RT))]"""
        d_max, k, n, Q, p = params
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        rate = k * np.exp(-Q / (8.314 * (T + 273.15))) * fG
        return d_max * (1 - np.exp(-rate * (t ** n)))
    
    def model3_modified_power(self, params, t, T, G):
        """Модифицированная степенная модель: d = A * t^m * T^n * f(G)"""
        A, m, n, p = params
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        return A * (t ** m) * (T ** n) * fG
    
    def model4_trunin_parameter(self, params, t, T, G):
        """Модель с параметром Трунина: d = A * P^m * f(G)"""
        A, m, p = params
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        T_kelvin = T + 273.15
        P = self.calculate_trunin_parameter(T_kelvin, t)
        return A * (P ** m) * fG
    
    def model5_combined(self, params, t, T, G):
        """Комбинированная модель: d = A * t^m * exp(-Q/RT) * P^n * f(G)"""
        A, m, Q, n, p = params
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        T_kelvin = T + 273.15
        P = self.calculate_trunin_parameter(T_kelvin, t)
        return A * (t ** m) * np.exp(-Q / (8.314 * T_kelvin)) * (P ** n) * fG
    
    def fit_models(self, df):
        """Обучение всех моделей"""
        t_data = df['t'].values
        T_data = df['T'].values
        G_data = df['G'].values
        d_data = df['d'].values
        
        # Рассчитываем параметр Трунина для всех данных
        T_kelvin_data = T_data + 273.15
        P_data = self.calculate_trunin_parameter(T_kelvin_data, t_data)
        
        models_config = {
            'model1_power_law': {
                'function': self.model1_power_law,
                'bounds': ([0.1, 0.01, 1000, 0.1], [10, 1, 50000, 2]),
                'initial_guess': [1, 0.1, 10000, 0.5]
            },
            'model2_saturating_growth': {
                'function': self.model2_saturating_growth,
                'bounds': ([1, 1e-6, 0.1, 1000, 0.1], [10, 1e-2, 2, 50000, 2]),
                'initial_guess': [3, 1e-4, 0.5, 10000, 0.5]
            },
            'model3_modified_power': {
                'function': self.model3_modified_power,
                'bounds': ([0.1, 0.01, 0.1, 0.1], [10, 1, 2, 2]),
                'initial_guess': [1, 0.1, 1, 0.5]
            },
            'model4_trunin_parameter': {
                'function': self.model4_trunin_parameter,
                'bounds': ([0.1, 0.1, 0.1], [10, 5, 2]),
                'initial_guess': [1, 1, 0.5]
            },
            'model5_combined': {
                'function': self.model5_combined,
                'bounds': ([0.1, 0.01, 1000, 0.1, 0.1], [10, 1, 50000, 5, 2]),
                'initial_guess': [1, 0.1, 10000, 1, 0.5]
            }
        }
        
        for model_name, config in models_config.items():
            try:
                def wrapper(X, *params):
                    t, T, G = X
                    return config['function'](params, t, T, G)
                
                popt, pcov = curve_fit(
                    wrapper, 
                    (t_data, T_data, G_data), 
                    d_data,
                    p0=config['initial_guess'],
                    bounds=config['bounds'],
                    maxfev=10000
                )
                
                # Расчет предсказаний и R²
                predictions = wrapper((t_data, T_data, G_data), *popt)
                r2 = 1 - np.sum((d_data - predictions) ** 2) / np.sum((d_data - np.mean(d_data)) ** 2)
                
                self.models[model_name] = {
                    'params': popt,
                    'r2': r2,
                    'predictions': predictions,
                    'rmse': np.sqrt(np.mean((d_data - predictions) ** 2)),
                    'mae': np.mean(np.abs(d_data - predictions))
                }
                
                if r2 > self.best_r2:
                    self.best_r2 = r2
                    self.best_model = model_name
                    
            except Exception as e:
                st.warning(f"Модель {model_name} не сошлась: {str(e)}")
    
    def predict_temperature(self, model_name, d_sigma, time_hours, grain_size):
        """Предсказание температуры по выбранной модели"""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не обучена")
            
        params = self.models[model_name]['params']
        grain_info = grain_df[grain_df['G'] == grain_size].iloc[0]
        
        try:
            if model_name == 'model1_power_law':
                # d = A * t^m * exp(-Q/RT) * f(G)
                A, m, Q, p = params
                fG = grain_info['inv_sqrt_a_v'] ** p
                term = d_sigma / (A * (time_hours ** m) * fG)
                if term <= 0:
                    raise ValueError("Некорректные параметры для расчета")
                inv_T = -np.log(term) * 8.314 / Q
                T_kelvin = 1 / inv_T
                return T_kelvin - 273.15
                
            elif model_name == 'model2_saturating_growth':
                # d = d_max * [1 - exp(-k * t^n * exp(-Q/RT) * f(G))]
                d_max, k, n, Q, p = params
                fG = grain_info['inv_sqrt_a_v'] ** p
                if d_sigma >= d_max:
                    return 1000  # Максимальная температура при насыщении
                term = -np.log(1 - d_sigma / d_max) / (k * (time_hours ** n) * fG)
                if term <= 0:
                    raise ValueError("Некорректные параметры для расчета")
                inv_T = -np.log(term) * 8.314 / Q
                T_kelvin = 1 / inv_T
                return T_kelvin - 273.15
                
            elif model_name == 'model3_modified_power':
                # d = A * t^m * T^n * f(G)
                A, m, n, p = params
                fG = grain_info['inv_sqrt_a_v'] ** p
                denominator = A * (time_hours ** m) * fG
                if denominator <= 0:
                    raise ValueError("Некорректные параметры для расчета")
                T = (d_sigma / denominator) ** (1/n)
                return T
                
            elif model_name == 'model4_trunin_parameter':
                # d = A * P^m * f(G)
                A, m, p = params
                fG = grain_info['inv_sqrt_a_v'] ** p
                
                # Решаем уравнение: d = A * [T(logτ - 2logT + 26.3)]^m * f(G)
                # Это требует численного решения
                def equation(T_kelvin):
                    P = self.calculate_trunin_parameter(T_kelvin, time_hours)
                    return A * (P ** m) * fG - d_sigma
                
                # Ищем корень в диапазоне 773-1173 K (500-900°C)
                from scipy.optimize import root_scalar
                result = root_scalar(equation, bracket=[773, 1173], method='brentq')
                
                if result.converged:
                    return result.root - 273.15
                else:
                    raise ValueError("Не удалось найти решение для температуры")
                    
            elif model_name == 'model5_combined':
                # d = A * t^m * exp(-Q/RT) * P^n * f(G)
                A, m, Q, n, p = params
                fG = grain_info['inv_sqrt_a_v'] ** p
                
                def equation(T_kelvin):
                    P = self.calculate_trunin_parameter(T_kelvin, time_hours)
                    term = A * (time_hours ** m) * (P ** n) * fG
                    return term * np.exp(-Q / (8.314 * T_kelvin)) - d_sigma
                
                from scipy.optimize import root_scalar
                result = root_scalar(equation, bracket=[773, 1173], method='brentq')
                
                if result.converged:
                    return result.root - 273.15
                else:
                    raise ValueError("Не удалось найти решение для температуры")
                
        except Exception as e:
            raise ValueError(f"Ошибка расчета температуры: {str(e)}")

def read_excel_file(uploaded_file):
    """Чтение Excel файла с обработкой различных форматов"""
    try:
        # Пробуем разные способы чтения
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            return df
        except Exception as e:
            st.warning(f"Не удалось прочитать с openpyxl: {e}. Пробуем другой способ...")
            try:
                df = pd.read_excel(uploaded_file, engine='xlrd')
                return df
            except:
                # Последняя попытка - без указания движка
                df = pd.read_excel(uploaded_file)
                return df
    except Exception as e:
        st.error(f"Не удалось прочитать файл. Убедитесь, что это корректный Excel файл.")
        return None

def main():
    st.set_page_config(page_title="Sigma Phase Analyzer", layout="wide")
    st.title("🔬 Анализатор сигма-фазы в стали 12Х18Н12Т")
    
    # Инициализация session state
    if 'excluded_points' not in st.session_state:
        st.session_state.excluded_points = set()
    
    # Создаем вкладки
    tab1, tab2 = st.tabs(["📊 Анализ данных и калибровка модели", "🧮 Калькулятор температуры"])
    
    with tab1:
        st.header("Калибровка физической модели")
        
        # Загрузка данных
        st.subheader("1. Загрузка данных")
        uploaded_file = st.file_uploader("Загрузите Excel файл с данными", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                df = read_excel_file(uploaded_file)
                
                if df is None:
                    st.stop()
                
                required_columns = ['G', 'T', 't', 'd']
                
                if all(col in df.columns for col in required_columns):
                    st.success("✅ Данные успешно загружены!")
                    
                    # Фильтруем некорректные температуры
                    df_clean = df[(df['T'] >= 500) & (df['T'] <= 900)].copy()
                    if len(df_clean) < len(df):
                        st.warning(f"Исключено {len(df) - len(df_clean)} точек с температурами вне диапазона 500-900°C")
                    
                    # Добавляем индекс для идентификации точек
                    df_clean = df_clean.reset_index(drop=True)
                    df_clean['point_id'] = df_clean.index
                    
                    # Показываем статистику
                    st.subheader("Статистика данных")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Количество измерений", len(df_clean))
                    with col2:
                        st.metric("Диапазон температур", f"{df_clean['T'].min()} - {df_clean['T'].max()} °C")
                    with col3:
                        st.metric("Диапазон времени", f"{df_clean['t'].min()} - {df_clean['t'].max()} ч")
                    with col4:
                        st.metric("Номера зерен", ", ".join(map(str, sorted(df_clean['G'].unique()))))
                    
                    # Визуализация данных
                    st.subheader("📈 Визуализация экспериментальных данных")
                    
                    # График зависимости от времени с интерактивным выбором
                    selection = alt.selection_point(fields=['point_id'], empty='none')
                    
                    time_chart = alt.Chart(df_clean).mark_circle(size=60).encode(
                        x=alt.X('t:Q', title='Время (ч)'),
                        y=alt.Y('d:Q', title='Диаметр (мкм²)'),
                        color=alt.condition(
                            selection,
                            alt.Color('T:Q', scale=alt.Scale(scheme='redyellowblue'), title='Температура (°C)'),
                            alt.value('lightgray')
                        ),
                        tooltip=['point_id', 'G', 'T', 't', 'd'],
                        opacity=alt.condition(selection, alt.value(1), alt.value(0.3))
                    ).properties(
                        width=600,
                        height=400,
                        title='Зависимость диаметра от времени и температуры (выделите точки для исключения)'
                    ).add_params(selection).facet(
                        column='G:N'
                    )
                    
                    st.altair_chart(time_chart)
                    
                    # Управление исключением точек
                    st.subheader("2. Управление исключением точек")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Показываем текущие исключенные точки
                        if st.session_state.excluded_points:
                            st.write("**Исключенные точки:**", sorted(st.session_state.excluded_points))
                        else:
                            st.write("**Исключенные точки:** нет")
                    
                    with col2:
                        # Кнопки управления
                        if st.button("Очистить все исключения"):
                            st.session_state.excluded_points = set()
                            st.rerun()
                    
                    # Ручной ввод точек для исключения
                    st.write("**Ручное управление исключениями:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        point_to_exclude = st.number_input("ID точки для исключения", 
                                                         min_value=0, 
                                                         max_value=len(df_clean)-1, 
                                                         value=0)
                    
                    with col2:
                        if st.button("Исключить точку"):
                            st.session_state.excluded_points.add(point_to_exclude)
                            st.rerun()
                    
                    with col3:
                        if st.button("Вернуть точку"):
                            if point_to_exclude in st.session_state.excluded_points:
                                st.session_state.excluded_points.remove(point_to_exclude)
                                st.rerun()
                    
                    # Фильтруем данные по исключенным точкам
                    df_filtered = df_clean[~df_clean['point_id'].isin(st.session_state.excluded_points)].copy()
                    
                    st.info(f"**Точек для анализа:** {len(df_filtered)} из {len(df_clean)}")
                    
                    # Сравнение моделей
                    st.subheader("3. Сравнение физических моделей")
                    
                    if len(df_filtered) >= 4:
                        advanced_model = AdvancedSigmaPhaseModel()
                        with st.spinner("Обучение моделей..."):
                            advanced_model.fit_models(df_filtered)
                        
                        # Таблица сравнения моделей
                        comparison_data = []
                        for model_name, model_info in advanced_model.models.items():
                            comparison_data.append({
                                'Модель': model_name,
                                'R²': model_info['r2'],
                                'RMSE': model_info['rmse'],
                                'MAE': model_info['mae'],
                                'Параметры': [f"{x:.4f}" for x in model_info['params']]
                            })
                        
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            comparison_df = comparison_df.sort_values('R²', ascending=False)
                            
                            # Отображаем таблицу
                            st.dataframe(comparison_df)
                            
                            # Выбор лучшей модели
                            best_model_name = advanced_model.best_model
                            best_model_info = advanced_model.models[best_model_name]
                            
                            st.success(f"🎯 Лучшая модель: **{best_model_name}** (R² = {best_model_info['r2']:.4f}, RMSE = {best_model_info['rmse']:.4f})")
                            
                            # Описания моделей
                            model_descriptions = {
                                'model1_power_law': 'd = A × t^m × exp(-Q/RT) × f(G)',
                                'model2_saturating_growth': 'd = d_max × [1 - exp(-k × t^n × exp(-Q/RT) × f(G))]',
                                'model3_modified_power': 'd = A × t^m × T^n × f(G)', 
                                'model4_trunin_parameter': 'd = A × P^m × f(G)  (P = T(logτ - 2logT + 26.3))',
                                'model5_combined': 'd = A × t^m × exp(-Q/RT) × P^n × f(G)'
                            }
                            
                            st.write(f"**Уравнение лучшей модели:** {model_descriptions.get(best_model_name, 'Неизвестно')}")
                            
                            # Сохранение лучшей модели
                            st.session_state['advanced_model'] = advanced_model
                            st.session_state['best_model_name'] = best_model_name
                            st.session_state['training_data'] = df_filtered
                            
                            # Визуализация предсказаний лучшей модели
                            st.subheader("📊 Валидация лучшей модели")
                            
                            plot_data = pd.DataFrame({
                                'Фактический': df_filtered['d'],
                                'Предсказанный': best_model_info['predictions'],
                                'Зерно': df_filtered['G'],
                                'Температура': df_filtered['T'],
                                'Время': df_filtered['t'],
                                'Точка': df_filtered['point_id']
                            })
                            
                            validation_chart = alt.Chart(plot_data).mark_circle(size=60).encode(
                                x=alt.X('Фактический:Q', title='Фактический диаметр (мкм²)'),
                                y=alt.Y('Предсказанный:Q', title='Предсказанный диаметр (мкм²)'),
                                color='Зерно:N',
                                tooltip=['Фактический', 'Предсказанный', 'Зерно', 'Температура', 'Время', 'Точка']
                            ).properties(
                                width=500,
                                height=400,
                                title=f'Предсказания модели {best_model_name}'
                            )
                            
                            line = alt.Chart(pd.DataFrame({
                                'x': [plot_data['Фактический'].min(), plot_data['Фактический'].max()],
                                'y': [plot_data['Фактический'].min(), plot_data['Фактический'].max()]
                            })).mark_line(color='red', strokeDash=[5,5]).encode(
                                x='x:Q',
                                y='y:Q'
                            )
                            
                            st.altair_chart(validation_chart + line)
                            
                        else:
                            st.error("Ни одна из моделей не сошлась. Попробуйте изменить данные или границы параметров.")
                    else:
                        st.warning("⚠️ Недостаточно данных для обучения модели. Нужно минимум 4 измерения.")
                        
                else:
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    st.error(f"❌ В файле отсутствуют столбцы: {missing_cols}")
                    
            except Exception as e:
                st.error(f"❌ Ошибка при обработке данных: {str(e)}")
        else:
            st.info("📁 Загрузите Excel файл с колонками: G, T, t, d")
    
    with tab2:
        st.header("🧮 Калькулятор температуры эксплуатации")
        
        if 'advanced_model' in st.session_state:
            model = st.session_state['advanced_model']
            best_model_name = st.session_state['best_model_name']
            training_data = st.session_state.get('training_data', None)
            
            st.success(f"✅ Используется модель: **{best_model_name}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                grain_number = st.selectbox("Номер зерна (G)", options=grain_df['G'].tolist())
            with col2:
                time_hours = st.number_input("Время эксплуатации (ч)", min_value=1, value=5000, step=100)
            with col3:
                d_sigma = st.number_input("Эквивалентный диаметр сигма-фазы (мкм²)", 
                                        min_value=0.1, value=2.0, step=0.1)
            
            if st.button("🎯 Рассчитать температуру", type="primary"):
                try:
                    temperature = model.predict_temperature(best_model_name, d_sigma, time_hours, grain_number)
                    
                    # Проверка диапазона
                    if temperature < 550:
                        st.error(f"""
                        ⚠️ **Рассчитанная температура: {temperature:.1f} °C**
                        
                        **Внимание:** Температура ниже 550°C - сигма-фаза практически не выделяется
                        """)
                    elif temperature > 900:
                        st.error(f"""
                        ⚠️ **Рассчитанная температура: {temperature:.1f} °C**
                        
                        **Внимание:** Температура выше 900°C - сигма-фаза не выделяется
                        """)
                    elif 590 <= temperature <= 630:
                        st.success(f"""
                        ✅ **Оптимальный диапазон: {temperature:.1f} °C**
                        
                        **Модель работает с максимальной точностью**
                        """)
                    else:
                        st.warning(f"""
                        📊 **Рассчитанная температура: {temperature:.1f} °C**
                        
                        **Внимание:** Температура вне оптимального диапазона 590-630°C
                        """)
                    
                    # Дополнительная информация
                    with st.expander("🔍 Детали расчета"):
                        grain_info = grain_df[grain_df['G'] == grain_number].iloc[0]
                        st.write(f"**Параметры зерна №{grain_number}:**")
                        st.write(f"- Средняя площадь сечения: {grain_info['a_v']:.6f} мм²")
                        st.write(f"- Средний диаметр: {grain_info['d_av']:.3f} мм")
                        st.write(f"- 1/√a_v = {grain_info['inv_sqrt_a_v']:.2f} мм⁻¹")
                        
                        if best_model_name == 'model4_trunin_parameter' or best_model_name == 'model5_combined':
                            T_kelvin = temperature + 273.15
                            P = model.calculate_trunin_parameter(T_kelvin, time_hours)
                            st.write(f"**Параметр Трунина:** P = {P:.1f}")
                            
                except Exception as e:
                    st.error(f"❌ Ошибка при расчете: {str(e)}")
        else:
            st.warning("📊 Сначала обучите модель во вкладке 'Анализ данных'")

if __name__ == "__main__":
    main()
