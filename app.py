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
    
    def model4_simple_power(self, params, t, T, G):
        """Простая степенная модель: d = A * t^m * T^n * G^p"""
        A, m, n, p = params
        return A * (t ** m) * (T ** n) * (G ** p)
    
    def fit_models(self, df):
        """Обучение всех моделей"""
        t_data = df['t'].values
        T_data = df['T'].values
        G_data = df['G'].values
        d_data = df['d'].values
        
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
            'model4_simple_power': {
                'function': self.model4_simple_power,
                'bounds': ([0.1, 0.01, 0.1, -1], [10, 1, 2, 1]),
                'initial_guess': [1, 0.1, 1, 0.1]
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
                    'rmse': np.sqrt(np.mean((d_data - predictions) ** 2))
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
                
            elif model_name == 'model4_simple_power':
                # d = A * t^m * T^n * G^p
                A, m, n, p = params
                denominator = A * (time_hours ** m) * (grain_size ** p)
                if denominator <= 0:
                    raise ValueError("Некорректные параметры для расчета")
                T = (d_sigma / denominator) ** (1/n)
                return T
                
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

def prepare_data(df, excluded_indices=[]):
    """Подготовка данных для регрессии"""
    df_clean = df.drop(excluded_indices).copy()
    
    # Фильтруем нулевые и отрицательные значения
    df_clean = df_clean[df_clean['d'] > 0].copy()
    
    # Добавляем данные по зернам
    df_clean = df_clean.merge(grain_df[['G', 'ln_inv_sqrt_a_v']], on='G', how='left')
    
    # Преобразуем переменные
    df_clean['ln_d'] = np.log(df_clean['d'])
    df_clean['ln_t'] = np.log(df_clean['t'])
    df_clean['inv_T'] = 1 / (df_clean['T'] + 273.15)  # T в Кельвинах
    
    # Создаем матрицу признаков
    X = df_clean[['ln_t', 'inv_T', 'ln_inv_sqrt_a_v']].values
    y = df_clean['ln_d'].values
    
    return X, y, df_clean

def create_validation_charts(df_clean, y, y_pred):
    """Создание графиков валидации с использованием Altair"""
    
    # Данные для графиков
    plot_data = pd.DataFrame({
        'actual': np.exp(y),
        'predicted': np.exp(y_pred),
        'residuals': np.exp(y) - np.exp(y_pred),
        'temperature': df_clean['T'],
        'grain_size': df_clean['G'],
        'time': df_clean['t']
    })
    
    # График 1: Предсказанные vs Фактические значения
    chart1 = alt.Chart(plot_data).mark_circle(size=60).encode(
        x=alt.X('actual:Q', title='Фактический диаметр (мкм²)'),
        y=alt.Y('predicted:Q', title='Предсказанный диаметр (мкм²)'),
        color='temperature:Q',
        tooltip=['actual', 'predicted', 'temperature', 'grain_size', 'time']
    ).properties(
        width=400,
        height=300,
        title='Предсказанные vs Фактические значения'
    )
    
    # Линия идеального предсказания
    min_val = plot_data[['actual', 'predicted']].min().min()
    max_val = plot_data[['actual', 'predicted']].max().max()
    line_data = pd.DataFrame({
        'x': [min_val, max_val],
        'y': [min_val, max_val]
    })
    
    line = alt.Chart(line_data).mark_line(color='red', strokeDash=[5,5]).encode(
        x='x:Q',
        y='y:Q'
    )
    
    chart1 = chart1 + line
    
    # График 2: Остатки
    chart2 = alt.Chart(plot_data).mark_circle(size=60).encode(
        x=alt.X('predicted:Q', title='Предсказанный диаметр (мкм²)'),
        y=alt.Y('residuals:Q', title='Остатки'),
        color='temperature:Q',
        tooltip=['predicted', 'residuals', 'temperature']
    ).properties(
        width=400,
        height=300,
        title='Остатки модели'
    )
    
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q')
    chart2 = chart2 + zero_line
    
    # График 3: Распределение ошибок
    chart3 = alt.Chart(plot_data).mark_bar().encode(
        x=alt.X('residuals:Q', bin=alt.Bin(maxbins=15), title='Ошибка предсказания'),
        y=alt.Y('count()', title='Частота')
    ).properties(
        width=400,
        height=300,
        title='Распределение ошибок'
    )
    
    return chart1, chart2, chart3

def main():
    st.set_page_config(page_title="Sigma Phase Analyzer", layout="wide")
    st.title("🔬 Анализатор сигма-фазы в стали 12Х18Н12Т")
    
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
                    
                    # Показываем данные
                    with st.expander("📋 Просмотр данных"):
                        st.dataframe(df_clean)
                    
                    # Визуализация данных
                    st.subheader("📈 Визуализация экспериментальных данных")
                    
                    # График зависимости от времени
                    time_chart = alt.Chart(df_clean).mark_circle(size=60).encode(
                        x=alt.X('t:Q', title='Время (ч)'),
                        y=alt.Y('d:Q', title='Диаметр (мкм²)'),
                        color=alt.Color('T:Q', scale=alt.Scale(scheme='redyellowblue'), title='Температура (°C)'),
                        tooltip=['G', 'T', 't', 'd']
                    ).properties(
                        width=600,
                        height=400,
                        title='Зависимость диаметра от времени и температуры'
                    ).facet(
                        column='G:N'
                    )
                    st.altair_chart(time_chart)
                    
                    # Сравнение моделей
                    st.subheader("🔍 Сравнение физических моделей")
                    
                    advanced_model = AdvancedSigmaPhaseModel()
                    with st.spinner("Обучение моделей..."):
                        advanced_model.fit_models(df_clean)
                    
                    # Таблица сравнения моделей
                    comparison_data = []
                    for model_name, model_info in advanced_model.models.items():
                        comparison_data.append({
                            'Модель': model_name,
                            'R²': model_info['r2'],
                            'RMSE': model_info['rmse'],
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
                            'model4_simple_power': 'd = A × t^m × T^n × G^p'
                        }
                        
                        st.write(f"**Уравнение лучшей модели:** {model_descriptions.get(best_model_name, 'Неизвестно')}")
                        
                        # Сохранение лучшей модели
                        st.session_state['advanced_model'] = advanced_model
                        st.session_state['best_model_name'] = best_model_name
                        st.session_state['training_data'] = df_clean
                        
                        # Визуализация предсказаний лучшей модели
                        st.subheader("📊 Валидация лучшей модели")
                        
                        plot_data = pd.DataFrame({
                            'Фактический': df_clean['d'],
                            'Предсказанный': best_model_info['predictions'],
                            'Зерно': df_clean['G'],
                            'Температура': df_clean['T'],
                            'Время': df_clean['t']
                        })
                        
                        validation_chart = alt.Chart(plot_data).mark_circle(size=60).encode(
                            x=alt.X('Фактический:Q', title='Фактический диаметр (мкм²)'),
                            y=alt.Y('Предсказанный:Q', title='Предсказанный диаметр (мкм²)'),
                            color='Зерно:N',
                            tooltip=['Фактический', 'Предсказанный', 'Зерно', 'Температура', 'Время']
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
                        
                        # Показываем параметры модели
                        st.subheader("📋 Параметры лучшей модели")
                        param_names = {
                            'model1_power_law': ['A', 'm', 'Q', 'p'],
                            'model2_saturating_growth': ['d_max', 'k', 'n', 'Q', 'p'],
                            'model3_modified_power': ['A', 'm', 'n', 'p'],
                            'model4_simple_power': ['A', 'm', 'n', 'p']
                        }
                        
                        params = best_model_info['params']
                        names = param_names.get(best_model_name, [f'Param_{i}' for i in range(len(params))])
                        
                        for name, value in zip(names, params):
                            st.write(f"**{name}** = {value:.6f}")
                        
                    else:
                        st.error("Ни одна из моделей не сошлась. Попробуйте изменить данные или границы параметров.")
                        
                else:
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    st.error(f"❌ В файле отсутствуют столбцы: {missing_cols}")
                    
            except Exception as e:
                st.error(f"❌ Ошибка при обработке данных: {str(e)}")
        else:
            st.info("📁 Загрузите Excel файл с колонками: G, T, t, d")
            
            # Пример данных
            with st.expander("📋 Пример формата данных"):
                example_data = pd.DataFrame({
                    'G': [3, 5, 8, 9],
                    'T': [600, 650, 700, 600],
                    't': [2000, 4000, 6000, 8000],
                    'd': [5.2, 8.7, 12.3, 6.8]
                })
                st.dataframe(example_data)
                st.write("**G** - номер зерна, **T** - температура (°C), **t** - время (ч), **d** - диаметр (мкм²)")
    
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
                        
                        if training_data is not None:
                            similar_data = training_data[
                                (training_data['G'] == grain_number) & 
                                (training_data['t'].between(time_hours*0.5, time_hours*1.5))
                            ]
                            if len(similar_data) > 0:
                                st.write("**Ближайшие экспериментальные точки:**")
                                st.dataframe(similar_data)
                            
                except Exception as e:
                    st.error(f"❌ Ошибка при расчете: {str(e)}")
        else:
            st.warning("📊 Сначала обучите модель во вкладке 'Анализ данных'")

if __name__ == "__main__":
    main()
