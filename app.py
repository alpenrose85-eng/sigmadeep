import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Данные по размерам зерен из ГОСТ (только нужные номера 3-10)
GRAIN_DATA = {
    'G': [3, 4, 5, 6, 7, 8, 9, 10],
    'a_v': [0.0156, 0.00781, 0.00390, 0.00195, 0.00098, 0.00049, 0.000244, 0.000122],
    'd_av': [0.125, 0.088, 0.062, 0.044, 0.031, 0.022, 0.015, 0.011]
}

grain_df = pd.DataFrame(GRAIN_DATA)
grain_df['inv_sqrt_a_v'] = 1 / np.sqrt(grain_df['a_v'])
grain_df['ln_inv_sqrt_a_v'] = np.log(grain_df['inv_sqrt_a_v'])

class TruninSigmaModel:
    def __init__(self):
        self.models = {}
        self.grain_models = {}
        
    def calculate_trunin_parameter(self, T_kelvin, time_hours):
        """Расчет параметра Трунина: P = T(logτ - 2logT + 26.3)"""
        return T_kelvin * (np.log10(time_hours) - 2 * np.log10(T_kelvin) + 26.3)
    
    def linear_model(self, P, a, b):
        """Линейная модель: d = a * P + b"""
        return a * P + b
    
    def power_model(self, P, a, b):
        """Степенная модель: d = a * P^b"""
        return a * (P ** b)
    
    def exponential_model(self, P, a, b, c):
        """Экспоненциальная модель: d = a * exp(b * P) + c"""
        return a * np.exp(b * P) + c
    
    def fit_global_models(self, df):
        """Обучение глобальных моделей для всех зерен"""
        # Рассчитываем параметр Трунина для всех точек
        T_kelvin = df['T'] + 273.15
        P_values = self.calculate_trunin_parameter(T_kelvin, df['t'])
        
        d_values = df['d'].values
        
        models_config = {
            'linear': {
                'function': self.linear_model,
                'bounds': ([-10, -10], [10, 10]),
                'initial_guess': [0.1, 1.0]
            },
            'power': {
                'function': self.power_model,
                'bounds': ([0.001, 0.1], [10, 5]),
                'initial_guess': [0.1, 1.0]
            },
            'exponential': {
                'function': self.exponential_model,
                'bounds': ([0.001, 0.001, -10], [10, 1, 10]),
                'initial_guess': [1.0, 0.01, 0.0]
            }
        }
        
        for model_name, config in models_config.items():
            try:
                popt, pcov = curve_fit(
                    config['function'],
                    P_values,
                    d_values,
                    p0=config['initial_guess'],
                    bounds=config['bounds'],
                    maxfev=10000
                )
                
                predictions = config['function'](P_values, *popt)
                r2 = 1 - np.sum((d_values - predictions) ** 2) / np.sum((d_values - np.mean(d_values)) ** 2)
                rmse = np.sqrt(np.mean((d_values - predictions) ** 2))
                
                self.models[model_name] = {
                    'params': popt,
                    'r2': r2,
                    'rmse': rmse,
                    'predictions': predictions,
                    'function': config['function']
                }
                
            except Exception as e:
                st.warning(f"Глобальная модель {model_name} не сошлась: {str(e)}")
    
    def fit_grain_specific_models(self, df):
        """Обучение отдельных моделей для каждого номера зерна"""
        for grain_size in sorted(df['G'].unique()):
            if grain_size < 3 or grain_size > 10:
                continue
                
            grain_data = df[df['G'] == grain_size].copy()
            if len(grain_data) < 3:  # Нужно минимум 3 точки для модели
                continue
                
            T_kelvin = grain_data['T'] + 273.15
            P_values = self.calculate_trunin_parameter(T_kelvin, grain_data['t'])
            d_values = grain_data['d'].values
            
            grain_models = {}
            
            try:
                # Линейная модель для зерна
                popt_linear, _ = curve_fit(
                    self.linear_model,
                    P_values,
                    d_values,
                    p0=[0.1, 1.0],
                    bounds=([-10, -10], [10, 10]),
                    maxfev=5000
                )
                pred_linear = self.linear_model(P_values, *popt_linear)
                r2_linear = 1 - np.sum((d_values - pred_linear) ** 2) / np.sum((d_values - np.mean(d_values)) ** 2)
                
                grain_models['linear'] = {
                    'params': popt_linear,
                    'r2': r2_linear,
                    'function': self.linear_model
                }
            except:
                pass
            
            try:
                # Степенная модель для зерна
                popt_power, _ = curve_fit(
                    self.power_model,
                    P_values,
                    d_values,
                    p0=[0.1, 1.0],
                    bounds=([0.001, 0.1], [10, 5]),
                    maxfev=5000
                )
                pred_power = self.power_model(P_values, *popt_power)
                r2_power = 1 - np.sum((d_values - pred_power) ** 2) / np.sum((d_values - np.mean(d_values)) ** 2)
                
                grain_models['power'] = {
                    'params': popt_power,
                    'r2': r2_power,
                    'function': self.power_model
                }
            except:
                pass
            
            self.grain_models[grain_size] = grain_models
    
    def predict_temperature_global(self, model_name, d_sigma, time_hours, grain_size):
        """Предсказание температуры по глобальной модели"""
        if model_name not in self.models:
            raise ValueError(f"Глобальная модель {model_name} не обучена")
        
        model_info = self.models[model_name]
        
        # Решаем уравнение численно
        from scipy.optimize import root_scalar
        
        def equation(T_celsius):
            T_kelvin = T_celsius + 273.15
            P = self.calculate_trunin_parameter(T_kelvin, time_hours)
            predicted_d = model_info['function'](P, *model_info['params'])
            return predicted_d - d_sigma
        
        try:
            result = root_scalar(equation, bracket=[550, 900], method='brentq')
            if result.converged:
                return result.root
            else:
                raise ValueError("Не удалось найти решение")
        except:
            # Если численное решение не сходится, используем грубый поиск
            temperatures = np.linspace(550, 900, 100)
            errors = []
            for T in temperatures:
                T_kelvin = T + 273.15
                P = self.calculate_trunin_parameter(T_kelvin, time_hours)
                predicted_d = model_info['function'](P, *model_info['params'])
                errors.append(abs(predicted_d - d_sigma))
            
            return temperatures[np.argmin(errors)]
    
    def predict_temperature_grain_specific(self, grain_size, d_sigma, time_hours, model_type='best'):
        """Предсказание температуры по модели для конкретного зерна"""
        if grain_size not in self.grain_models:
            raise ValueError(f"Нет модели для зерна {grain_size}")
        
        grain_models = self.grain_models[grain_size]
        
        if not grain_models:
            raise ValueError(f"Нет подходящих моделей для зерна {grain_size}")
        
        # Выбираем лучшую модель или конкретный тип
        if model_type == 'best':
            best_r2 = -1
            best_model_name = None
            for model_name, model_info in grain_models.items():
                if model_info['r2'] > best_r2:
                    best_r2 = model_info['r2']
                    best_model_name = model_name
            model_name = best_model_name
        else:
            model_name = model_type
        
        if model_name not in grain_models:
            raise ValueError(f"Модель {model_type} не доступна для зерна {grain_size}")
        
        model_info = grain_models[model_name]
        
        # Решаем уравнение численно
        from scipy.optimize import root_scalar
        
        def equation(T_celsius):
            T_kelvin = T_celsius + 273.15
            P = self.calculate_trunin_parameter(T_kelvin, time_hours)
            predicted_d = model_info['function'](P, *model_info['params'])
            return predicted_d - d_sigma
        
        try:
            result = root_scalar(equation, bracket=[550, 900], method='brentq')
            if result.converged:
                return result.root, model_name, model_info['r2']
            else:
                raise ValueError("Не удалось найти решение")
        except:
            # Грубый поиск если численный метод не работает
            temperatures = np.linspace(550, 900, 200)
            errors = []
            for T in temperatures:
                T_kelvin = T + 273.15
                P = self.calculate_trunin_parameter(T_kelvin, time_hours)
                predicted_d = model_info['function'](P, *model_info['params'])
                errors.append(abs(predicted_d - d_sigma))
            
            best_idx = np.argmin(errors)
            return temperatures[best_idx], model_name, model_info['r2']

def read_excel_file(uploaded_file):
    """Чтение Excel файла"""
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Sigma Phase Analyzer", layout="wide")
    st.title("🔬 Анализатор сигма-фазы в стали 12Х18Н12Т")
    
    # Инициализация session state
    if 'excluded_points' not in st.session_state:
        st.session_state.excluded_points = set()
    
    # Создаем вкладки
    tab1, tab2 = st.tabs(["📊 Анализ данных и построение моделей", "🧮 Калькулятор температуры"])
    
    with tab1:
        st.header("Анализ данных и построение моделей")
        
        # Загрузка данных
        uploaded_file = st.file_uploader("Загрузите Excel файл с данными", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                df = read_excel_file(uploaded_file)
                
                if df is None:
                    st.stop()
                
                required_columns = ['G', 'T', 't', 'd']
                
                if all(col in df.columns for col in required_columns):
                    # Фильтруем данные: только зерна 3-10 и температуры >= 550°C
                    df_clean = df[
                        (df['G'].between(3, 10)) & 
                        (df['T'] >= 550) & 
                        (df['T'] <= 900) &
                        (df['d'] > 0)
                    ].copy()
                    
                    if len(df_clean) == 0:
                        st.error("Нет данных, удовлетворяющих критериям (зерна 3-10, температуры 550-900°C)")
                        st.stop()
                    
                    # Добавляем индекс и параметр Трунина
                    df_clean = df_clean.reset_index(drop=True)
                    df_clean['point_id'] = df_clean.index
                    
                    T_kelvin = df_clean['T'] + 273.15
                    df_clean['P_trunin'] = T_kelvin * (np.log10(df_clean['t']) - 2 * np.log10(T_kelvin) + 26.3)
                    
                    # Управление исключением точек
                    st.subheader("1. Управление данными")
                    
                    # Показываем статистику
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Всего точек", len(df_clean))
                    with col2:
                        st.metric("Номера зерен", ", ".join(map(str, sorted(df_clean['G'].unique()))))
                    with col3:
                        st.metric("Диапазон P Трунина", f"{df_clean['P_trunin'].min():.0f}-{df_clean['P_trunin'].max():.0f}")
                    with col4:
                        st.metric("Исключено точек", len(st.session_state.excluded_points))
                    
                    # Таблица данных с возможностью исключения
                    st.write("**Таблица данных (отметьте точки для исключения):**")
                    
                    for idx, row in df_clean.iterrows():
                        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 2, 1])
                        with col1:
                            excluded = st.checkbox(
                                "", 
                                value=idx in st.session_state.excluded_points,
                                key=f"exclude_{idx}"
                            )
                            if excluded:
                                st.session_state.excluded_points.add(idx)
                            else:
                                if idx in st.session_state.excluded_points:
                                    st.session_state.excluded_points.remove(idx)
                        with col2:
                            st.write(f"**{idx}**")
                        with col3:
                            st.write(f"G={row['G']}")
                        with col4:
                            st.write(f"T={row['T']}°C")
                        with col5:
                            st.write(f"t={row['t']}ч")
                        with col6:
                            st.write(f"d={row['d']:.3f} мкм²")
                        with col7:
                            st.write(f"P={row['P_trunin']:.0f}")
                    
                    # Кнопки управления
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Обновить модели"):
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Очистить исключения"):
                            st.session_state.excluded_points = set()
                            st.rerun()
                    
                    # Фильтруем данные
                    df_filtered = df_clean[~df_clean['point_id'].isin(st.session_state.excluded_points)].copy()
                    
                    st.info(f"**Точек для анализа:** {len(df_filtered)}")
                    
                    # Визуализация зависимости от параметра Трунина
                    st.subheader("2. Зависимость диаметра от параметра Трунина")
                    
                    chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
                        x=alt.X('P_trunin:Q', title='Параметр Трунина P'),
                        y=alt.Y('d:Q', title='Диаметр сигма-фазы (мкм²)'),
                        color=alt.Color('G:N', title='Номер зерна'),
                        tooltip=['G', 'T', 't', 'd', 'P_trunin']
                    ).properties(
                        width=800,
                        height=400,
                        title='Зависимость диаметра сигма-фазы от параметра Трунина'
                    )
                    
                    st.altair_chart(chart)
                    
                    # Построение моделей
                    st.subheader("3. Построение моделей")
                    
                    if len(df_filtered) >= 3:
                        model = TruninSigmaModel()
                        
                        # Обучаем глобальные модели
                        with st.spinner("Обучение глобальных моделей..."):
                            model.fit_global_models(df_filtered)
                        
                        # Обучаем модели для каждого зерна
                        with st.spinner("Обучение моделей для каждого зерна..."):
                            model.fit_grain_specific_models(df_filtered)
                        
                        # Показываем результаты глобальных моделей
                        st.write("**Глобальные модели (все зерна):**")
                        if model.models:
                            global_results = []
                            for model_name, model_info in model.models.items():
                                global_results.append({
                                    'Модель': model_name,
                                    'R²': model_info['r2'],
                                    'RMSE': f"{model_info['rmse']:.3f}",
                                    'Параметры': str([f"{p:.4f}" for p in model_info['params']])
                                })
                            
                            global_df = pd.DataFrame(global_results)
                            st.dataframe(global_df)
                        else:
                            st.warning("Глобальные модели не сошлись")
                        
                        # Показываем результаты для каждого зерна
                        st.write("**Модели для отдельных зерен:**")
                        grain_results = []
                        for grain_size in sorted(model.grain_models.keys()):
                            grain_models = model.grain_models[grain_size]
                            if grain_models:
                                best_r2 = max([m['r2'] for m in grain_models.values()])
                                best_model = [name for name, m in grain_models.items() if m['r2'] == best_r2][0]
                                grain_results.append({
                                    'Зерно': grain_size,
                                    'Лучшая модель': best_model,
                                    'R²': best_r2,
                                    'Точек': len(df_filtered[df_filtered['G'] == grain_size])
                                })
                        
                        if grain_results:
                            grain_df_results = pd.DataFrame(grain_results)
                            st.dataframe(grain_df_results)
                            
                            # Визуализация моделей для каждого зерна
                            st.write("**Визуализация моделей для каждого зерна:**")
                            
                            # Создаем сетку графиков
                            grains = sorted(model.grain_models.keys())
                            n_cols = 2
                            n_rows = (len(grains) + n_cols - 1) // n_cols
                            
                            for row in range(n_rows):
                                cols = st.columns(n_cols)
                                for col in range(n_cols):
                                    idx = row * n_cols + col
                                    if idx < len(grains):
                                        grain_size = grains[idx]
                                        with cols[col]:
                                            # Данные для этого зерна
                                            grain_data = df_filtered[df_filtered['G'] == grain_size]
                                            if len(grain_data) > 0:
                                                # Создаем график
                                                chart = alt.Chart(grain_data).mark_circle(size=50).encode(
                                                    x=alt.X('P_trunin:Q', title='P Трунина'),
                                                    y=alt.Y('d:Q', title='Диаметр (мкм²)'),
                                                    tooltip=['T', 't', 'd', 'P_trunin']
                                                ).properties(
                                                    width=250,
                                                    height=200,
                                                    title=f'Зерно {grain_size}'
                                                )
                                                
                                                # Добавляем линию тренда если есть хорошая модель
                                                grain_models = model.grain_models[grain_size]
                                                if grain_models and max([m['r2'] for m in grain_models.values()]) > 0.5:
                                                    # Создаем данные для линии
                                                    P_range = np.linspace(
                                                        grain_data['P_trunin'].min(), 
                                                        grain_data['P_trunin'].max(), 
                                                        100
                                                    )
                                                    best_r2 = max([m['r2'] for m in grain_models.values()])
                                                    best_model_name = [name for name, m in grain_models.items() if m['r2'] == best_r2][0]
                                                    best_model = grain_models[best_model_name]
                                                    d_pred = best_model['function'](P_range, *best_model['params'])
                                                    
                                                    trend_data = pd.DataFrame({
                                                        'P_trunin': P_range,
                                                        'd': d_pred
                                                    })
                                                    
                                                    trend_line = alt.Chart(trend_data).mark_line(color='red').encode(
                                                        x='P_trunin:Q',
                                                        y='d:Q'
                                                    )
                                                    
                                                    chart = chart + trend_line
                                                
                                                st.altair_chart(chart)
                        
                        # Сохраняем модели
                        st.session_state['model'] = model
                        st.session_state['training_data'] = df_filtered
                        
                    else:
                        st.warning("Недостаточно данных для построения моделей")
                        
                else:
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    st.error(f"Отсутствуют столбцы: {missing_cols}")
                    
            except Exception as e:
                st.error(f"Ошибка обработки данных: {str(e)}")
        else:
            st.info("Загрузите Excel файл с колонками: G, T, t, d")
    
    with tab2:
        st.header("Калькулятор температуры")
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            training_data = st.session_state.get('training_data', None)
            
            st.write("Введите параметры для расчета температуры:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                grain_size = st.selectbox("Номер зерна", options=[3, 4, 5, 6, 7, 8, 9, 10])
            with col2:
                time_hours = st.number_input("Время эксплуатации (ч)", min_value=1, value=5000, step=100)
            with col3:
                d_sigma = st.number_input("Диаметр сигма-фазы (мкм²)", min_value=0.1, value=2.0, step=0.1)
            
            # Выбор типа модели
            col1, col2 = st.columns(2)
            with col1:
                use_grain_specific = st.checkbox("Использовать модель для конкретного зерна", value=True)
            with col2:
                if use_grain_specific:
                    model_type = st.selectbox("Тип модели", options=['best', 'linear', 'power'])
            
            if st.button("🎯 Рассчитать температуру"):
                try:
                    if use_grain_specific and grain_size in model.grain_models:
                        temperature, used_model, r2 = model.predict_temperature_grain_specific(
                            grain_size, d_sigma, time_hours, model_type
                        )
                        st.success(f"**Рассчитанная температура:** {temperature:.1f} °C")
                        st.write(f"**Использована модель:** {used_model} для зерна {grain_size}")
                        st.write(f"**Качество модели R²:** {r2:.4f}")
                        
                        # Показываем доверительный интервал ±5°C
                        st.info(f"**Диапазон:** {temperature-5:.1f} - {temperature+5:.1f} °C")
                        
                    else:
                        # Используем лучшую глобальную модель
                        if model.models:
                            best_global = max(model.models.items(), key=lambda x: x[1]['r2'])
                            temperature = model.predict_temperature_global(best_global[0], d_sigma, time_hours, grain_size)
                            st.success(f"**Рассчитанная температура:** {temperature:.1f} °C")
                            st.write(f"**Использована глобальная модель:** {best_global[0]}")
                            st.write(f"**Качество модели R²:** {best_global[1]['r2']:.4f}")
                            st.info(f"**Диапазон:** {temperature-5:.1f} - {temperature+5:.1f} °C")
                        else:
                            st.error("Нет доступных моделей для расчета")
                    
                    # Дополнительная информация
                    with st.expander("Детали расчета"):
                        if training_data is not None:
                            similar_data = training_data[
                                (training_data['G'] == grain_size) & 
                                (abs(training_data['d'] - d_sigma) <= 0.5)
                            ]
                            if len(similar_data) > 0:
                                st.write("**Ближайшие экспериментальные точки:**")
                                st.dataframe(similar_data[['G', 'T', 't', 'd']])
                            
                except Exception as e:
                    st.error(f"Ошибка расчета: {str(e)}")
        else:
            st.warning("Сначала постройте модели во вкладке анализа данных")

if __name__ == "__main__":
    main()
