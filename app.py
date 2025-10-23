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

class AdvancedSigmaPhaseModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_r2 = -np.inf
        
    def calculate_trunin_parameter(self, T_kelvin, time_hours):
        """Расчет параметра Трунина: P = T(logτ - 2logT + 26.3)"""
        return T_kelvin * (np.log10(time_hours) - 2 * np.log10(T_kelvin) + 26.3)
    
    def model1_power_law(self, t, T, G, A, m, Q, p):
        """Степенная модель: d = A * t^m * exp(-Q/RT) * f(G)"""
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        return A * (t ** m) * np.exp(-Q / (8.314 * (T + 273.15))) * fG
    
    def model2_saturating_growth(self, t, T, G, d_max, k, n, Q, p):
        """Модель насыщающего роста: d = d_max * [1 - exp(-k * t^n * exp(-Q/RT))]"""
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        rate = k * np.exp(-Q / (8.314 * (T + 273.15))) * fG
        return d_max * (1 - np.exp(-rate * (t ** n)))
    
    def model3_modified_power(self, t, T, G, A, m, n, p):
        """Модифицированная степенная модель: d = A * t^m * T^n * f(G)"""
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        return A * (t ** m) * (T ** n) * fG
    
    def model4_trunin_parameter(self, t, T, G, A, m, p):
        """Модель с параметром Трунина: d = A * P^m * f(G)"""
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        T_kelvin = T + 273.15
        P = self.calculate_trunin_parameter(T_kelvin, t)
        return A * (P ** m) * fG
    
    def model5_combined(self, t, T, G, A, m, Q, n, p):
        """Комбинированная модель: d = A * t^m * exp(-Q/RT) * P^n * f(G)"""
        grain_info = grain_df[grain_df['G'] == G].iloc[0]
        fG = grain_info['inv_sqrt_a_v'] ** p
        T_kelvin = T + 273.15
        P = self.calculate_trunin_parameter(T_kelvin, t)
        return A * (t ** m) * np.exp(-Q / (8.314 * T_kelvin)) * (P ** n) * fG
    
    def fit_models(self, df):
        """Обучение всех моделей"""
        # Преобразуем данные в правильный формат
        data_points = []
        for idx, row in df.iterrows():
            data_points.append({
                't': row['t'],
                'T': row['T'], 
                'G': row['G'],
                'd': row['d']
            })
        
        t_data = np.array([p['t'] for p in data_points])
        T_data = np.array([p['T'] for p in data_points])
        G_data = np.array([p['G'] for p in data_points])
        d_data = np.array([p['d'] for p in data_points])
        
        models_config = {
            'model1_power_law': {
                'function': self._fit_model1,
                'description': 'd = A × t^m × exp(-Q/RT) × f(G)'
            },
            'model2_saturating_growth': {
                'function': self._fit_model2, 
                'description': 'd = d_max × [1 - exp(-k × t^n × exp(-Q/RT) × f(G))]'
            },
            'model3_modified_power': {
                'function': self._fit_model3,
                'description': 'd = A × t^m × T^n × f(G)'
            },
            'model4_trunin_parameter': {
                'function': self._fit_model4,
                'description': 'd = A × P^m × f(G)  (P = T(logτ - 2logT + 26.3))'
            },
            'model5_combined': {
                'function': self._fit_model5,
                'description': 'd = A × t^m × exp(-Q/RT) × P^n × f(G)'
            }
        }
        
        for model_name, config in models_config.items():
            try:
                result = config['function'](t_data, T_data, G_data, d_data)
                if result is not None:
                    params, predictions, r2, rmse, mae = result
                    self.models[model_name] = {
                        'params': params,
                        'r2': r2,
                        'predictions': predictions,
                        'rmse': rmse,
                        'mae': mae,
                        'description': config['description']
                    }
                    
                    if r2 > self.best_r2:
                        self.best_r2 = r2
                        self.best_model = model_name
                    
            except Exception as e:
                st.warning(f"Модель {model_name} не сошлась: {str(e)}")
    
    def _fit_model1(self, t_data, T_data, G_data, d_data):
        """Обучение модели 1"""
        try:
            # Создаем функцию для curve_fit
            def model_func(data, A, m, Q, p):
                t, T, G = data
                result = np.zeros_like(t)
                for i in range(len(t)):
                    result[i] = self.model1_power_law(t[i], T[i], G[i], A, m, Q, p)
                return result
            
            # Начальные guess и границы
            initial_guess = [1, 0.1, 10000, 0.5]
            bounds = ([0.1, 0.01, 1000, 0.1], [10, 1, 50000, 2])
            
            # Подгонка
            popt, pcov = curve_fit(
                model_func,
                (t_data, T_data, G_data),
                d_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            # Предсказания и метрики
            predictions = model_func((t_data, T_data, G_data), *popt)
            r2 = 1 - np.sum((d_data - predictions) ** 2) / np.sum((d_data - np.mean(d_data)) ** 2)
            rmse = np.sqrt(np.mean((d_data - predictions) ** 2))
            mae = np.mean(np.abs(d_data - predictions))
            
            return popt, predictions, r2, rmse, mae
            
        except Exception as e:
            raise Exception(f"Model1 error: {str(e)}")
    
    def _fit_model2(self, t_data, T_data, G_data, d_data):
        """Обучение модели 2"""
        try:
            def model_func(data, d_max, k, n, Q, p):
                t, T, G = data
                result = np.zeros_like(t)
                for i in range(len(t)):
                    result[i] = self.model2_saturating_growth(t[i], T[i], G[i], d_max, k, n, Q, p)
                return result
            
            initial_guess = [3, 1e-4, 0.5, 10000, 0.5]
            bounds = ([1, 1e-6, 0.1, 1000, 0.1], [10, 1e-2, 2, 50000, 2])
            
            popt, pcov = curve_fit(
                model_func,
                (t_data, T_data, G_data),
                d_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            predictions = model_func((t_data, T_data, G_data), *popt)
            r2 = 1 - np.sum((d_data - predictions) ** 2) / np.sum((d_data - np.mean(d_data)) ** 2)
            rmse = np.sqrt(np.mean((d_data - predictions) ** 2))
            mae = np.mean(np.abs(d_data - predictions))
            
            return popt, predictions, r2, rmse, mae
            
        except Exception as e:
            raise Exception(f"Model2 error: {str(e)}")
    
    def _fit_model3(self, t_data, T_data, G_data, d_data):
        """Обучение модели 3"""
        try:
            def model_func(data, A, m, n, p):
                t, T, G = data
                result = np.zeros_like(t)
                for i in range(len(t)):
                    result[i] = self.model3_modified_power(t[i], T[i], G[i], A, m, n, p)
                return result
            
            initial_guess = [1, 0.1, 1, 0.5]
            bounds = ([0.1, 0.01, 0.1, 0.1], [10, 1, 2, 2])
            
            popt, pcov = curve_fit(
                model_func,
                (t_data, T_data, G_data),
                d_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            predictions = model_func((t_data, T_data, G_data), *popt)
            r2 = 1 - np.sum((d_data - predictions) ** 2) / np.sum((d_data - np.mean(d_data)) ** 2)
            rmse = np.sqrt(np.mean((d_data - predictions) ** 2))
            mae = np.mean(np.abs(d_data - predictions))
            
            return popt, predictions, r2, rmse, mae
            
        except Exception as e:
            raise Exception(f"Model3 error: {str(e)}")
    
    def _fit_model4(self, t_data, T_data, G_data, d_data):
        """Обучение модели 4"""
        try:
            def model_func(data, A, m, p):
                t, T, G = data
                result = np.zeros_like(t)
                for i in range(len(t)):
                    result[i] = self.model4_trunin_parameter(t[i], T[i], G[i], A, m, p)
                return result
            
            initial_guess = [1, 1, 0.5]
            bounds = ([0.1, 0.1, 0.1], [10, 5, 2])
            
            popt, pcov = curve_fit(
                model_func,
                (t_data, T_data, G_data),
                d_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            predictions = model_func((t_data, T_data, G_data), *popt)
            r2 = 1 - np.sum((d_data - predictions) ** 2) / np.sum((d_data - np.mean(d_data)) ** 2)
            rmse = np.sqrt(np.mean((d_data - predictions) ** 2))
            mae = np.mean(np.abs(d_data - predictions))
            
            return popt, predictions, r2, rmse, mae
            
        except Exception as e:
            raise Exception(f"Model4 error: {str(e)}")
    
    def _fit_model5(self, t_data, T_data, G_data, d_data):
        """Обучение модели 5"""
        try:
            def model_func(data, A, m, Q, n, p):
                t, T, G = data
                result = np.zeros_like(t)
                for i in range(len(t)):
                    result[i] = self.model5_combined(t[i], T[i], G[i], A, m, Q, n, p)
                return result
            
            initial_guess = [1, 0.1, 10000, 1, 0.5]
            bounds = ([0.1, 0.01, 1000, 0.1, 0.1], [10, 1, 50000, 5, 2])
            
            popt, pcov = curve_fit(
                model_func,
                (t_data, T_data, G_data),
                d_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            predictions = model_func((t_data, T_data, G_data), *popt)
            r2 = 1 - np.sum((d_data - predictions) ** 2) / np.sum((d_data - np.mean(d_data)) ** 2)
            rmse = np.sqrt(np.mean((d_data - predictions) ** 2))
            mae = np.mean(np.abs(d_data - predictions))
            
            return popt, predictions, r2, rmse, mae
            
        except Exception as e:
            raise Exception(f"Model5 error: {str(e)}")
    
    def predict_temperature(self, model_name, d_sigma, time_hours, grain_size):
        """Предсказание температуры по выбранной модели"""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не обучена")
            
        params = self.models[model_name]['params']
        
        try:
            if model_name == 'model1_power_law':
                # d = A * t^m * exp(-Q/RT) * f(G)
                A, m, Q, p = params
                grain_info = grain_df[grain_df['G'] == grain_size].iloc[0]
                fG = grain_info['inv_sqrt_a_v'] ** p
                term = d_sigma / (A * (time_hours ** m) * fG)
                if term <= 0:
                    raise ValueError("Некорректные параметры для расчета")
                inv_T = -np.log(term) * 8.314 / Q
                T_kelvin = 1 / inv_T
                return T_kelvin - 273.15
                
            elif model_name == 'model3_modified_power':
                # d = A * t^m * T^n * f(G)
                A, m, n, p = params
                grain_info = grain_df[grain_df['G'] == grain_size].iloc[0]
                fG = grain_info['inv_sqrt_a_v'] ** p
                denominator = A * (time_hours ** m) * fG
                if denominator <= 0:
                    raise ValueError("Некорректные параметры для расчета")
                T = (d_sigma / denominator) ** (1/n)
                return T
                
            else:
                # Для сложных моделей используем численное решение
                from scipy.optimize import root_scalar
                
                def equation(T_celsius):
                    T_kelvin = T_celsius + 273.15
                    if model_name == 'model2_saturating_growth':
                        d_max, k, n, Q, p = params
                        grain_info = grain_df[grain_df['G'] == grain_size].iloc[0]
                        fG = grain_info['inv_sqrt_a_v'] ** p
                        rate = k * np.exp(-Q / (8.314 * T_kelvin)) * fG
                        return d_max * (1 - np.exp(-rate * (time_hours ** n))) - d_sigma
                    elif model_name == 'model4_trunin_parameter':
                        A, m, p = params
                        grain_info = grain_df[grain_df['G'] == grain_size].iloc[0]
                        fG = grain_info['inv_sqrt_a_v'] ** p
                        P = self.calculate_trunin_parameter(T_kelvin, time_hours)
                        return A * (P ** m) * fG - d_sigma
                    elif model_name == 'model5_combined':
                        A, m, Q, n, p = params
                        grain_info = grain_df[grain_df['G'] == grain_size].iloc[0]
                        fG = grain_info['inv_sqrt_a_v'] ** p
                        P = self.calculate_trunin_parameter(T_kelvin, time_hours)
                        return A * (time_hours ** m) * np.exp(-Q / (8.314 * T_kelvin)) * (P ** n) * fG - d_sigma
                
                result = root_scalar(equation, bracket=[500, 900], method='brentq')
                
                if result.converged:
                    return result.root
                else:
                    raise ValueError("Не удалось найти решение для температуры")
                
        except Exception as e:
            raise ValueError(f"Ошибка расчета температуры: {str(e)}")

def read_excel_file(uploaded_file):
    """Чтение Excel файла с обработкой различных форматов"""
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
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    
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
                    # Сохраняем оригинальные данные
                    st.session_state.original_data = df.copy()
                    
                    # Фильтруем некорректные температуры
                    df_clean = df[(df['T'] >= 500) & (df['T'] <= 900)].copy()
                    if len(df_clean) < len(df):
                        st.warning(f"Автоматически исключено {len(df) - len(df_clean)} точек с температурами вне диапазона 500-900°C")
                    
                    # Добавляем индекс для идентификации точек
                    df_clean = df_clean.reset_index(drop=True)
                    df_clean['point_id'] = df_clean.index
                    df_clean['excluded'] = df_clean['point_id'].isin(st.session_state.excluded_points)
                    
                    # Показываем загруженные данные
                    st.subheader("📋 Загруженные данные")
                    st.write(f"**Всего точек:** {len(df_clean)}")
                    
                    # Создаем редактируемую таблицу
                    st.write("**Таблица данных (отметьте точки для исключения):**")
                    
                    # Создаем копию для редактирования
                    edited_df = df_clean.copy()
                    
                    # Добавляем чекбоксы для исключения
                    for idx in edited_df.index:
                        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 2, 2, 2, 2, 2, 2])
                        with col1:
                            excluded = st.checkbox(
                                "Исключить", 
                                value=edited_df.loc[idx, 'excluded'],
                                key=f"exclude_{idx}"
                            )
                            if excluded and idx not in st.session_state.excluded_points:
                                st.session_state.excluded_points.add(idx)
                            elif not excluded and idx in st.session_state.excluded_points:
                                st.session_state.excluded_points.remove(idx)
                        
                        with col2:
                            st.write(f"**{idx}**")
                        with col3:
                            st.write(f"G = {edited_df.loc[idx, 'G']}")
                        with col4:
                            st.write(f"T = {edited_df.loc[idx, 'T']}°C")
                        with col5:
                            st.write(f"t = {edited_df.loc[idx, 't']}ч")
                        with col6:
                            st.write(f"d = {edited_df.loc[idx, 'd']}мкм²")
                        with col7:
                            if edited_df.loc[idx, 'excluded']:
                                st.error("❌ Исключена")
                            else:
                                st.success("✅ Включена")
                    
                    # Кнопки управления
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("🔄 Обновить данные"):
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Очистить все исключения"):
                            st.session_state.excluded_points = set()
                            st.rerun()
                    with col3:
                        if st.button("📊 Показать только включенные точки"):
                            # Временное отображение только включенных точек
                            pass
                    
                    # Фильтруем данные по исключенным точкам
                    df_filtered = df_clean[~df_clean['point_id'].isin(st.session_state.excluded_points)].copy()
                    
                    st.info(f"**Точек для анализа:** {len(df_filtered)} из {len(df_clean)}")
                    
                    # Визуализация данных
                    st.subheader("📈 Визуализация данных")
                    
                    if len(df_filtered) > 0:
                        # График зависимости от времени
                        chart_data = df_clean.copy()
                        chart_data['status'] = chart_data['point_id'].apply(
                            lambda x: 'Исключена' if x in st.session_state.excluded_points else 'Включена'
                        )
                        
                        time_chart = alt.Chart(chart_data).mark_circle(size=60).encode(
                            x=alt.X('t:Q', title='Время (ч)'),
                            y=alt.Y('d:Q', title='Диаметр (мкм²)'),
                            color=alt.Color('status:N', scale=alt.Scale(
                                domain=['Включена', 'Исключена'],
                                range=['blue', 'lightgray']
                            )),
                            tooltip=['point_id', 'G', 'T', 't', 'd', 'status'],
                            opacity=alt.condition(
                                alt.datum.status == 'Включена',
                                alt.value(1),
                                alt.value(0.3)
                            )
                        ).properties(
                            width=600,
                            height=400,
                            title='Зависимость диаметра от времени (синие точки - включены в анализ)'
                        ).facet(
                            column='G:N'
                        )
                        
                        st.altair_chart(time_chart)
                    
                    # Сравнение моделей
                    st.subheader("3. Сравнение физических моделей")
                    
                    if len(df_filtered) >= 4:
                        advanced_model = AdvancedSigmaPhaseModel()
                        with st.spinner("Обучение моделей..."):
                            advanced_model.fit_models(df_filtered)
                        
                        if advanced_model.models:
                            # Таблица сравнения моделей
                            comparison_data = []
                            for model_name, model_info in advanced_model.models.items():
                                comparison_data.append({
                                    'Модель': model_name,
                                    'Описание': model_info['description'],
                                    'R²': model_info['r2'],
                                    'RMSE': model_info['rmse'],
                                    'MAE': model_info['mae']
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            comparison_df = comparison_df.sort_values('R²', ascending=False)
                            
                            # Отображаем таблицу
                            st.dataframe(comparison_df)
                            
                            # Выбор лучшей модели
                            best_model_name = advanced_model.best_model
                            best_model_info = advanced_model.models[best_model_name]
                            
                            st.success(f"🎯 Лучшая модель: **{best_model_name}** (R² = {best_model_info['r2']:.4f})")
                            st.write(f"**Уравнение:** {best_model_info['description']}")
                            
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
                                'Время': df_filtered['t']
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
                            
                        else:
                            st.error("Ни одна из моделей не сошлась. Попробуйте изменить данные или исключить выбросы.")
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
            st.write(f"**Уравнение:** {model.models[best_model_name]['description']}")
            st.write(f"**Качество модели:** R² = {model.models[best_model_name]['r2']:.4f}")
            
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
                    temperature = model.predict_temperature(best_model_name, d_sigma, time_hours, grain_size)
                    
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
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при расчете: {str(e)}")
        else:
            st.warning("📊 Сначала обучите модель во вкладке 'Анализ данных'")

if __name__ == "__main__":
    main()
