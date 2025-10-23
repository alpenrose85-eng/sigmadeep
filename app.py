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

class PhysicsBasedSigmaModel:
    def __init__(self):
        self.models = {}
        self.grain_models = {}
        
    def calculate_trunin_parameter(self, T_kelvin, time_hours):
        """Расчет параметра Трунина: P = T(logτ - 2logT + 26.3)"""
        return T_kelvin * (np.log10(time_hours) - 2 * np.log10(T_kelvin) + 26.3)
    
    def arrhenius_model(self, t, T, G, A, Q, n, p):
        """Модель на основе уравнения Аррениуса: d = A * t^n * exp(-Q/RT) * f(G)"""
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        R = 8.314  # Дж/(моль·К)
        return A * (t ** n) * np.exp(-Q / (R * (T + 273.15))) * fG
    
    def trunin_power_model(self, P, a, b, c):
        """Степенная модель с параметром Трунина: d = a * (P - P0)^b + c"""
        P0 = 18000  # Пороговое значение P, ниже которого фаза не выделяется (~550°C)
        if P <= P0:
            return c
        return a * ((P - P0) ** b) + c
    
    def linear_trunin_model(self, P, a, b):
        """Линейная модель: d = a * P + b"""
        return a * P + b
    
    def fit_physics_models(self, df):
        """Обучение физически обоснованных моделей"""
        # Рассчитываем параметр Трунина для всех точек
        T_kelvin = df['T'] + 273.15
        P_values = self.calculate_trunin_parameter(T_kelvin, df['t'])
        d_values = df['d'].values
        
        # Модель на основе Аррениуса (глобальная для всех зерен)
        try:
            def arrhenius_wrapper(X, A, Q, n, p):
                t, T, G = X
                result = np.zeros_like(t)
                for i in range(len(t)):
                    result[i] = self.arrhenius_model(t[i], T[i], G[i], A, Q, n, p)
                return result
            
            # Начальные приближения based on typical values for sigma phase
            initial_guess = [1.0, 200000, 0.3, 0.5]  # A, Q (Дж/моль), n, p
            bounds = ([0.001, 50000, 0.1, 0.1], [100, 500000, 1.0, 2.0])
            
            popt, pcov = curve_fit(
                arrhenius_wrapper,
                (df['t'].values, df['T'].values, df['G'].values),
                d_values,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            predictions = arrhenius_wrapper((df['t'].values, df['T'].values, df['G'].values), *popt)
            r2 = 1 - np.sum((d_values - predictions) ** 2) / np.sum((d_values - np.mean(d_values)) ** 2)
            rmse = np.sqrt(np.mean((d_values - predictions) ** 2))
            
            self.models['arrhenius'] = {
                'params': popt,
                'r2': r2,
                'rmse': rmse,
                'predictions': predictions,
                'Q_kJ_mol': popt[1] / 1000,  # Энергия активации в кДж/моль
                'formula': f"d = {popt[0]:.3f} × t^{popt[2]:.3f} × exp(-{popt[1]/1000:.0f}/(RT)) × (1/√a_v)^{popt[3]:.3f}"
            }
        except Exception as e:
            st.warning(f"Модель Аррениуса не сошлась: {str(e)}")
        
        # Модели с параметром Трунина
        models_config = {
            'trunin_power': {
                'function': self.trunin_power_model,
                'bounds': ([0.0001, 0.1, -1], [10, 3, 5]),
                'initial_guess': [0.01, 1.0, 0.0],
                'formula_template': "d = {a:.4f} × (P - 18000)^{b:.3f} + {c:.3f}"
            },
            'linear_trunin': {
                'function': self.linear_trunin_model,
                'bounds': ([-10, -10], [10, 10]),
                'initial_guess': [0.001, 1.0],
                'formula_template': "d = {a:.4f} × P + {b:.3f}"
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
                    maxfev=5000
                )
                
                predictions = config['function'](P_values, *popt)
                r2 = 1 - np.sum((d_values - predictions) ** 2) / np.sum((d_values - np.mean(d_values)) ** 2)
                rmse = np.sqrt(np.mean((d_values - predictions) ** 2))
                
                # Форматируем формулу
                formula = config['formula_template'].format(**{chr(97+i): param for i, param in enumerate(popt)})
                
                self.models[model_name] = {
                    'params': popt,
                    'r2': r2,
                    'rmse': rmse,
                    'predictions': predictions,
                    'formula': formula,
                    'function': config['function']
                }
                
            except Exception as e:
                st.warning(f"Модель {model_name} не сошлась: {str(e)}")
    
    def fit_grain_specific_physics_models(self, df):
        """Обучение физических моделей для каждого зерна"""
        for grain_size in sorted(df['G'].unique()):
            if grain_size < 3 or grain_size > 10:
                continue
                
            grain_data = df[df['G'] == grain_size].copy()
            if len(grain_data) < 3:
                continue
                
            T_kelvin = grain_data['T'] + 273.15
            P_values = self.calculate_trunin_parameter(T_kelvin, grain_data['t'])
            d_values = grain_data['d'].values
            
            grain_models = {}
            
            # Модель Аррениуса для конкретного зерна
            try:
                def arrhenius_grain(t, A, Q, n):
                    R = 8.314
                    return A * (t ** n) * np.exp(-Q / (R * (grain_data['T'].values + 273.15)))
                
                popt, _ = curve_fit(
                    arrhenius_grain,
                    grain_data['t'].values,
                    d_values,
                    p0=[1.0, 200000, 0.3],
                    bounds=([0.001, 50000, 0.1], [100, 500000, 1.0]),
                    maxfev=5000
                )
                
                predictions = arrhenius_grain(grain_data['t'].values, *popt)
                r2 = 1 - np.sum((d_values - predictions) ** 2) / np.sum((d_values - np.mean(d_values)) ** 2)
                
                grain_models['arrhenius'] = {
                    'params': popt,
                    'r2': r2,
                    'Q_kJ_mol': popt[1] / 1000,
                    'formula': f"d = {popt[0]:.3f} × t^{popt[2]:.3f} × exp(-{popt[1]/1000:.0f}/(RT))"
                }
            except:
                pass
            
            # Модель Трунина для конкретного зерна
            try:
                popt, _ = curve_fit(
                    self.trunin_power_model,
                    P_values,
                    d_values,
                    p0=[0.01, 1.0, 0.0],
                    bounds=([0.0001, 0.1, -1], [10, 3, 5]),
                    maxfev=5000
                )
                
                predictions = self.trunin_power_model(P_values, *popt)
                r2 = 1 - np.sum((d_values - predictions) ** 2) / np.sum((d_values - np.mean(d_values)) ** 2)
                
                formula = f"d = {popt[0]:.4f} × (P - 18000)^{popt[1]:.3f} + {popt[2]:.3f}"
                
                grain_models['trunin_power'] = {
                    'params': popt,
                    'r2': r2,
                    'formula': formula,
                    'function': self.trunin_power_model
                }
            except:
                pass
            
            if grain_models:
                self.grain_models[grain_size] = grain_models
    
    def predict_temperature_physics(self, model_name, d_sigma, time_hours, grain_size):
        """Предсказание температуры по физической модели"""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не обучена")
        
        model_info = self.models[model_name]
        
        from scipy.optimize import root_scalar
        
        if model_name == 'arrhenius':
            # Для модели Аррениуса
            A, Q, n, p = model_info['params']
            grain_info = grain_df[grain_df['G'] == grain_size].iloc[0]
            fG = grain_info['inv_sqrt_a_v'] ** p
            
            def equation(T_celsius):
                R = 8.314
                predicted_d = A * (time_hours ** n) * np.exp(-Q / (R * (T_celsius + 273.15))) * fG
                return predicted_d - d_sigma
            
            result = root_scalar(equation, bracket=[550, 900], method='brentq')
            if result.converged:
                return result.root, model_info
            else:
                raise ValueError("Не удалось найти решение")
        
        else:
            # Для моделей Трунина
            def equation(T_celsius):
                T_kelvin = T_celsius + 273.15
                P = self.calculate_trunin_parameter(T_kelvin, time_hours)
                predicted_d = model_info['function'](P, *model_info['params'])
                return predicted_d - d_sigma
            
            result = root_scalar(equation, bracket=[550, 900], method='brentq')
            if result.converged:
                return result.root, model_info
            else:
                raise ValueError("Не удалось найти решение")
    
    def predict_temperature_grain_specific(self, grain_size, d_sigma, time_hours, model_type='best'):
        """Предсказание температуры по модели для конкретного зерна"""
        if grain_size not in self.grain_models:
            raise ValueError(f"Нет модели для зерна {grain_size}")
        
        grain_models = self.grain_models[grain_size]
        
        if not grain_models:
            raise ValueError(f"Нет подходящих моделей для зерна {grain_size}")
        
        # Выбираем лучшую модель
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
        
        from scipy.optimize import root_scalar
        
        if model_name == 'arrhenius':
            # Модель Аррениуса для зерна
            A, Q, n = model_info['params']
            
            def equation(T_celsius):
                R = 8.314
                predicted_d = A * (time_hours ** n) * np.exp(-Q / (R * (T_celsius + 273.15)))
                return predicted_d - d_sigma
            
            result = root_scalar(equation, bracket=[550, 900], method='brentq')
            if result.converged:
                return result.root, model_name, model_info
            else:
                raise ValueError("Не удалось найти решение")
        
        else:
            # Модель Трунина для зерна
            def equation(T_celsius):
                T_kelvin = T_celsius + 273.15
                P = self.calculate_trunin_parameter(T_kelvin, time_hours)
                predicted_d = model_info['function'](P, *model_info['params'])
                return predicted_d - d_sigma
            
            result = root_scalar(equation, bracket=[550, 900], method='brentq')
            if result.converged:
                return result.root, model_name, model_info
            else:
                raise ValueError("Не удалось найти решение")

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
    tab1, tab2 = st.tabs(["📊 Физическое моделирование", "🧮 Калькулятор температуры"])
    
    with tab1:
        st.header("Физическое моделирование роста сигма-фазы")
        
        # Теория процесса
        with st.expander("📚 Физика процесса роста сигма-фазы"):
            st.markdown("""
            ### Физические основы моделирования
            
            **1. Уравнение Аррениуса:**
            ```
            d = A × tⁿ × exp(-Q/RT) × f(G)
            ```
            где:
            - `Q` - энергия активации процесса [Дж/моль]
            - `R` - газовая постоянная = 8.314 Дж/(моль·К)
            - `T` - температура в Кельвинах
            - `A` - предэкспоненциальный множитель
            - `n` - кинетический параметр
            - `f(G)` - функция влияния размера зерна
            
            **2. Параметр Трунина:**
            ```
            P = T × (logτ - 2logT + 26.3)
            ```
            Универсальный параметр, связывающий температуру и время.
            
            **Ожидаемые значения энергии активации:**
            - Диффузия хрома в аустените: ~240-280 кДж/моль
            - Рост интерметаллидных фаз: ~200-400 кДж/моль
            """)
        
        # Загрузка данных
        uploaded_file = st.file_uploader("Загрузите Excel файл с данными", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                df = read_excel_file(uploaded_file)
                
                if df is None:
                    st.stop()
                
                required_columns = ['G', 'T', 't', 'd']
                
                if all(col in df.columns for col in required_columns):
                    # Фильтруем данные
                    df_clean = df[
                        (df['G'].between(3, 10)) & 
                        (df['T'] >= 550) & 
                        (df['T'] <= 900) &
                        (df['d'] > 0)
                    ].copy()
                    
                    if len(df_clean) == 0:
                        st.error("Нет данных, удовлетворяющих критериям")
                        st.stop()
                    
                    # Добавляем индекс и параметр Трунина
                    df_clean = df_clean.reset_index(drop=True)
                    df_clean['point_id'] = df_clean.index
                    
                    T_kelvin = df_clean['T'] + 273.15
                    df_clean['P_trunin'] = self.calculate_trunin_parameter(T_kelvin, df_clean['t'])
                    
                    # Управление данными
                    st.subheader("1. Управление данными")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Всего точек", len(df_clean))
                    with col2:
                        st.metric("Номера зерен", ", ".join(map(str, sorted(df_clean['G'].unique()))))
                    with col3:
                        st.metric("Исключено", len(st.session_state.excluded_points))
                    
                    # Таблица данных
                    st.write("**Таблица данных:**")
                    for idx, row in df_clean.iterrows():
                        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 2, 1])
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
                            st.write(f"t={row['t']}ч, d={row['d']:.3f}")
                        with col6:
                            st.write(f"P={row['P_trunin']:.0f}")
                    
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
                    
                    # Визуализация
                    st.subheader("2. Визуализация данных")
                    
                    # Автоматическое масштабирование графика
                    P_min, P_max = df_filtered['P_trunin'].min(), df_filtered['P_trunin'].max()
                    d_min, d_max = df_filtered['d'].min(), df_filtered['d'].max()
                    
                    # Добавляем margins для лучшего отображения
                    P_range = [P_min - 0.1*(P_max-P_min), P_max + 0.1*(P_max-P_min)]
                    d_range = [max(0, d_min - 0.1*(d_max-d_min)), d_max + 0.1*(d_max-d_min)]
                    
                    chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
                        x=alt.X('P_trunin:Q', title='Параметр Трунина P', scale=alt.Scale(domain=P_range)),
                        y=alt.Y('d:Q', title='Диаметр сигма-фазы (мкм²)', scale=alt.Scale(domain=d_range)),
                        color=alt.Color('G:N', title='Номер зерна'),
                        tooltip=['G', 'T', 't', 'd', 'P_trunin']
                    ).properties(
                        width=800,
                        height=500,
                        title='Зависимость диаметра сигма-фазы от параметра Трунина'
                    )
                    
                    st.altair_chart(chart)
                    
                    # Построение моделей
                    st.subheader("3. Физические модели роста")
                    
                    if len(df_filtered) >= 4:
                        model = PhysicsBasedSigmaModel()
                        
                        with st.spinner("Обучение физических моделей..."):
                            model.fit_physics_models(df_filtered)
                            model.fit_grain_specific_physics_models(df_filtered)
                        
                        # Результаты глобальных моделей
                        st.write("**Глобальные физические модели:**")
                        if model.models:
                            physics_results = []
                            for model_name, model_info in model.models.items():
                                result_row = {
                                    'Модель': model_name,
                                    'R²': model_info['r2'],
                                    'RMSE': f"{model_info['rmse']:.3f}",
                                    'Формула': model_info['formula']
                                }
                                if 'Q_kJ_mol' in model_info:
                                    result_row['Q, кДж/моль'] = f"{model_info['Q_kJ_mol']:.0f}"
                                physics_results.append(result_row)
                            
                            physics_df = pd.DataFrame(physics_results)
                            st.dataframe(physics_df)
                            
                            # Анализ энергии активации
                            if 'arrhenius' in model.models:
                                Q_exp = model.models['arrhenius']['Q_kJ_mol']
                                st.info(f"""
                                **🔬 Экспериментальная энергия активации: {Q_exp:.0f} кДж/моль**
                                
                                *Сравнение с литературными данными:*
                                - Диффузия Cr в Fe: 240-280 кДж/моль
                                - Рост интерметаллидов: 200-400 кДж/моль
                                - Выделение карбидов: 150-250 кДж/моль
                                """)
                        else:
                            st.warning("Глобальные модели не сошлись")
                        
                        # Модели для отдельных зерен
                        st.write("**Модели для отдельных зерен:**")
                        grain_physics_results = []
                        for grain_size in sorted(model.grain_models.keys()):
                            grain_models = model.grain_models[grain_size]
                            if grain_models:
                                best_r2 = max([m['r2'] for m in grain_models.values()])
                                best_model_name = [name for name, m in grain_models.items() if m['r2'] == best_r2][0]
                                best_model = grain_models[best_model_name]
                                
                                result_row = {
                                    'Зерно': grain_size,
                                    'Лучшая модель': best_model_name,
                                    'R²': best_r2,
                                    'Формула': best_model['formula']
                                }
                                if 'Q_kJ_mol' in best_model:
                                    result_row['Q, кДж/моль'] = f"{best_model['Q_kJ_mol']:.0f}"
                                
                                grain_physics_results.append(result_row)
                        
                        if grain_physics_results:
                            grain_physics_df = pd.DataFrame(grain_physics_results)
                            st.dataframe(grain_physics_df)
                        
                        # Сохраняем модели
                        st.session_state['physics_model'] = model
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
        st.header("Калькулятор температуры эксплуатации")
        
        if 'physics_model' in st.session_state:
            model = st.session_state['physics_model']
            training_data = st.session_state.get('training_data', None)
            
            st.write("### Введите параметры для расчета температуры:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                grain_size = st.selectbox("Номер зерна", options=[3, 4, 5, 6, 7, 8, 9, 10])
            with col2:
                time_hours = st.number_input("Время эксплуатации (ч)", min_value=1, value=5000, step=100)
            with col3:
                d_sigma = st.number_input("Диаметр сигма-фазы (мкм²)", min_value=0.1, value=2.0, step=0.1)
            
            # Выбор модели
            col1, col2 = st.columns(2)
            with col1:
                use_grain_specific = st.checkbox("Использовать модель для конкретного зерна", value=True)
            with col2:
                if use_grain_specific:
                    model_type = st.selectbox("Тип модели", options=['best', 'arrhenius', 'trunin_power'])
                else:
                    available_models = list(model.models.keys())
                    global_model = st.selectbox("Глобальная модель", options=available_models)
            
            if st.button("🎯 Рассчитать температуру"):
                try:
                    if use_grain_specific and grain_size in model.grain_models:
                        temperature, used_model, model_info = model.predict_temperature_grain_specific(
                            grain_size, d_sigma, time_hours, model_type
                        )
                        
                        st.success(f"### Рассчитанная температура: {temperature:.1f} °C")
                        st.write(f"**Использована модель:** {used_model} для зерна {grain_size}")
                        st.write(f"**Формула модели:** {model_info['formula']}")
                        st.write(f"**Качество модели R²:** {model_info['r2']:.4f}")
                        
                        if 'Q_kJ_mol' in model_info:
                            st.write(f"**Энергия активации:** {model_info['Q_kJ_mol']:.0f} кДж/моль")
                        
                    else:
                        temperature, model_info = model.predict_temperature_physics(
                            global_model, d_sigma, time_hours, grain_size
                        )
                        
                        st.success(f"### Рассчитанная температура: {temperature:.1f} °C")
                        st.write(f"**Использована глобальная модель:** {global_model}")
                        st.write(f"**Формула модели:** {model_info['formula']}")
                        st.write(f"**Качество модели R²:** {model_info['r2']:.4f}")
                        
                        if 'Q_kJ_mol' in model_info:
                            st.write(f"**Энергия активации:** {model_info['Q_kJ_mol']:.0f} кДж/моль")
                    
                    # Доверительный интервал
                    st.info(f"**Температурный диапазон:** {temperature-5:.1f} - {temperature+5:.1f} °C")
                    
                    # Дополнительная информация
                    with st.expander("🔍 Физическая интерпретация"):
                        st.write(f"**Параметр Трунина для рассчитанных условий:**")
                        T_kelvin = temperature + 273.15
                        P = model.calculate_trunin_parameter(T_kelvin, time_hours)
                        st.write(f"P = {P:.0f}")
                        
                        if training_data is not None:
                            similar_data = training_data[
                                (training_data['G'] == grain_size) & 
                                (abs(training_data['d'] - d_sigma) <= 0.3)
                            ]
                            if len(similar_data) > 0:
                                st.write("**Ближайшие экспериментальные точки:**")
                                st.dataframe(similar_data[['G', 'T', 't', 'd']].round(3))
                            
                except Exception as e:
                    st.error(f"Ошибка расчета: {str(e)}")
        else:
            st.warning("Сначала постройте модели во вкладке физического моделирования")

if __name__ == "__main__":
    main()
