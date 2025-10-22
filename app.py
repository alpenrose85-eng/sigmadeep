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

# Модифицированная модель KJMA с учетом температурных ограничений
def sigma_phase_model(params, G, T, t):
    """
    Модель образования сигма-фазы на основе уравнения KJMA 
    с учетом размера зерна и температурных ограничений
    """
    K0, a, b, n, T_sigma_min, T_sigma_max = params
    R = 8.314  # Универсальная газовая постоянная
    
    # Температурные ограничения (в Кельвинах)
    T_min = T_sigma_min + 273.15  # Минимальная температура образования сигма-фазы
    T_max = T_sigma_max + 273.15  # Максимальная температура растворения
    
    # Эффективная температура с учетом ограничений
    T_eff = np.where(T < T_min, T_min, T)
    T_eff = np.where(T_eff > T_max, T_max, T_eff)
    
    # Температурный фактор (сигмоида для плавного перехода)
    temp_factor = 1 / (1 + np.exp(-0.1 * (T - (T_min + 50)))) * 1 / (1 + np.exp(0.1 * (T - (T_max - 50))))
    
    Q = a + b * G
    K = K0 * np.exp(-Q / (R * T_eff)) * temp_factor
    
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
        self.model_version = "1.1"
        self.creation_date = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()
        
    def detect_outliers(self, data, method='iqr', contamination=0.1):
        """Обнаружение выбросов в данных"""
        features = data[['Номер_зерна', 'Температура_K', 'Время_ч', 'Сигма_фаза_процент']].values
        
        if method == 'iqr':
            # Применяем IQR к каждому параметру отдельно
            outlier_flags = np.zeros(len(data), dtype=bool)
            
            for i, col in enumerate(['Сигма_фаза_процент', 'Время_ч', 'Температура_K']):
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
            # Будем определять после первичной подгонки модели
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
            G = self.clean_data['Номер_зерна'].values
            T = self.clean_data['Температура_K'].values
            t = self.clean_data['Время_ч'].values
            sigma_exp = self.clean_data['Сигма_фаза_процент'].values / 100.0  # Конвертация % в доли
            
            # Начальные guess-значения параметров с температурными ограничениями
            initial_guess = [1e10, 200000, 10000, 1.0, 550.0, 900.0]  # [K0, a, b, n, T_min_C, T_max_C]
            
            # Границы параметров
            bounds = (
                [1e5, 100000, 0, 0.1, 500.0, 850.0],    # нижние границы
                [1e15, 500000, 50000, 4.0, 600.0, 950.0] # верхние границы
            )
            
            # Подгонка параметров
            self.params, _ = curve_fit(
                lambda x, K0, a, b, n, T_min, T_max: sigma_phase_model([K0, a, b, n, T_min, T_max], G, T, t),
                np.arange(len(G)), sigma_exp,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            # Расчет метрик качества
            sigma_pred = sigma_phase_model(self.params, G, T, t) * 100  # Обратно в проценты
            sigma_exp_percent = sigma_exp * 100
            self.R2 = r2_score(sigma_exp_percent, sigma_pred)
            self.rmse = np.sqrt(mean_squared_error(sigma_exp_percent, sigma_pred))
            
            # Если используется метод остатков, пересчитываем выбросы
            if remove_outliers and outlier_method == 'residual':
                residuals = np.abs(sigma_pred - sigma_exp_percent)
                residual_threshold = np.mean(residuals) + 2 * np.std(residuals)
                residual_outliers = residuals > residual_threshold
                
                if np.any(residual_outliers):
                    outlier_data_residual = self.clean_data[residual_outliers]
                    clean_data_residual = self.clean_data[~residual_outliers]
                    
                    # Переподгонка модели без выбросов по остаткам
                    G_clean = clean_data_residual['Номер_зерна'].values
                    T_clean = clean_data_residual['Температура_K'].values
                    t_clean = clean_data_residual['Время_ч'].values
                    sigma_exp_clean = clean_data_residual['Сигма_фаза_процент'].values / 100.0
                    
                    self.params, _ = curve_fit(
                        lambda x, K0, a, b, n, T_min, T_max: sigma_phase_model([K0, a, b, n, T_min, T_max], G_clean, T_clean, t_clean),
                        np.arange(len(G_clean)), sigma_exp_clean,
                        p0=self.params,
                        bounds=bounds,
                        maxfev=10000
                    )
                    
                    # Обновляем информацию о выбросах
                    self.outlier_info['outlier_data'] = outlier_data_residual
                    self.outlier_info['outlier_count'] = len(outlier_data_residual)
                    self.clean_data = clean_data_residual
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка при подгонке модели: {str(e)}")
            return False
    
    def predict_temperature(self, G, sigma_percent, t):
        """Предсказание температуры по известным параметрам"""
        if self.params is None:
            raise ValueError("Модель не обучена!")
        
        K0, a, b, n, T_sigma_min, T_sigma_max = self.params
        R = 8.314
        sigma = sigma_percent / 100.0  # Конвертация % в доли
        
        try:
            Q = a + b * G
            term = -np.log(1 - sigma) / (K0 * (t ** n))
            if term <= 0:
                return None
            
            T = -Q / (R * np.log(term))
            
            # Применяем температурные ограничения
            T_min_K = T_sigma_min + 273.15
            T_max_K = T_sigma_max + 273.15
            
            if T < T_min_K:
                return T_min_K - 273.15
            elif T > T_max_K:
                return T_max_K - 273.15
            else:
                return T - 273.15
                
        except:
            return None
    
    def plot_results_with_outliers(self, data):
        """Визуализация результатов с выделением выбросов"""
        if self.params is None:
            return None
        
        # Предсказанные значения для всех данных
        G_all = data['Номер_зерна'].values
        T_all = data['Температура_K'].values
        t_all = data['Время_ч'].values
        sigma_exp_all = data['Сигма_фаза_процент'].values
        sigma_pred_all = sigma_phase_model(self.params, G_all, T_all, t_all) * 100  # В проценты
        
        # Определяем, какие точки являются выбросами
        is_outlier = np.zeros(len(data), dtype=bool)
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
        
        fig.update_xaxes(title_text='Экспериментальные значения (%)', row=1, col=1)
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
        unique_temps = sorted(data['Температура_K'].unique())
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, temp in enumerate(unique_temps):
            if i >= len(colors):
                break
                
            temp_data = data[data['Температура_K'] == temp]
            temp_outliers = temp_data[temp_data.index.isin(outlier_indices)] if np.any(outlier_mask) else pd.DataFrame()
            temp_clean = temp_data[~temp_data.index.isin(outlier_indices)] if np.any(outlier_mask) else temp_data
            
            # Чистые данные
            if len(temp_clean) > 0:
                fig.add_trace(
                    go.Scatter(x=temp_clean['Время_ч'], y=temp_clean['Сигма_фаза_процент'],
                              mode='markers', name=f'Чистые {temp}K',
                              marker=dict(color=colors[i], size=8)),
                    row=2, col=1
                )
            
            # Выбросы
            if len(temp_outliers) > 0:
                fig.add_trace(
                    go.Scatter(x=temp_outliers['Время_ч'], y=temp_outliers['Сигма_фаза_процент'],
                              mode='markers', name=f'Выбросы {temp}K',
                              marker=dict(color=colors[i], size=10, symbol='x')),
                    row=2, col=1
                )
        
        fig.update_xaxes(title_text='Время (ч)', row=2, col=1)
        fig.update_yaxes(title_text='Сигма-фаза (%)', row=2, col=1)
        
        # График 4: Температурные зависимости
        unique_times = sorted(data['Время_ч'].unique())[:3]  # Первые 3 времени
        for i, time_val in enumerate(unique_times):
            if i >= len(colors):
                break
                
            time_data = data[data['Время_ч'] == time_val]
            time_outliers = time_data[time_data.index.isin(outlier_indices)] if np.any(outlier_mask) else pd.DataFrame()
            time_clean = time_data[~time_data.index.isin(outlier_indices)] if np.any(outlier_mask) else time_data
            
            # Чистые данные
            if len(time_clean) > 0:
                fig.add_trace(
                    go.Scatter(x=time_clean['Температура_K'] - 273.15, y=time_clean['Сигма_фаза_процент'],
                              mode='markers', name=f'Чистые {time_val}ч',
                              marker=dict(color=colors[i], size=8)),
                    row=2, col=2
                )
        
        fig.update_xaxes(title_text='Температура (°C)', row=2, col=2)
        fig.update_yaxes(title_text='Сигма-фаза (%)', row=2, col=2)
        
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
            'last_modified': self.last_modified
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
            except Exception as e:
                st.sidebar.error(f"Ошибка загрузки проекта: {str(e)}")
    
    # Настройки обработки выбросов
    st.sidebar.header("🎯 Настройки обработки выбросов")
    
    remove_outliers = st.sidebar.checkbox("Удалять выбросы", value=True)
    
    if remove_outliers:
        outlier_method = st.sidebar.selectbox(
            "Метод обнаружения выбросов",
            ['iqr', 'isolation_forest', 'residual'],
            format_func=lambda x: {
                'iqr': 'Межквартильный размах (IQR)',
                'isolation_forest': 'Isolation Forest', 
                'residual': 'По остаткам модели'
            }[x]
        )
        
        contamination = st.sidebar.slider(
            "Ожидаемая доля выбросов", 
            min_value=0.01, max_value=0.3, value=0.1, step=0.01
        )
    else:
        outlier_method = 'none'
        contamination = 0.1
    
    # Пример данных с содержанием сигма-фазы в процентах
    sample_data = pd.DataFrame({
        'Номер_зерна': [3, 3, 5, 5, 8, 8, 9, 9, 3, 5, 8],
        'Температура_C': [600, 650, 600, 700, 650, 700, 600, 700, 600, 650, 750],
        'Температура_K': [873, 923, 873, 973, 923, 973, 873, 973, 873, 923, 1023],
        'Время_ч': [2000, 4000, 4000, 2000, 6000, 4000, 8000, 6000, 2000, 4000, 4000],
        'Сигма_фаза_процент': [5.2, 12.5, 8.1, 15.3, 18.7, 25.1, 22.4, 35.2, 12.8, 25.6, 2.1]
    })
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    sample_csv = convert_df_to_csv(sample_data)
    
    st.sidebar.download_button(
        label="📥 Скачать пример данных (CSV)",
        data=sample_csv,
        file_name="sample_sigma_phase_data.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.sidebar.file_uploader(
        "Загрузите CSV файл с экспериментальными данными",
        type=['csv']
    )
    
    if uploaded_file is not None:
        st.session_state.current_data = pd.read_csv(uploaded_file)
    elif st.session_state.current_data is None:
        st.info("👈 Пожалуйста, загрузите CSV файл с данными или используйте пример данных")
        st.session_state.current_data = sample_data
    
    # Показ загруженных данных
    st.header("📊 Экспериментальные данные")
    
    # Редактирование данных
    edited_data = st.data_editor(
        st.session_state.current_data,
        num_rows="dynamic",
        use_container_width=True
    )
    
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
            with col4:
                st.metric("T_max (°C)", f"{T_sigma_max:.1f}")
            
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
                                        min_value=1.0, max_value=12.0, 
                                        value=5.0, step=0.1)
            with col2:
                sigma_input = st.number_input("Содержание сигма-фазы (%)", 
                                            min_value=0.0, max_value=50.0,
                                            value=10.0, step=0.1,
                                            help="От 0% до 50%")
            with col3:
                t_input = st.number_input("Время эксплуатации (ч)", 
                                        min_value=100, max_value=100000,
                                        value=4000, step=100)
            
            if st.button("🔍 Рассчитать температуру", key="calc_temp"):
                try:
                    T_celsius = analyzer.predict_temperature(G_input, sigma_input, t_input)
                    
                    if T_celsius is not None:
                        st.success(f"""
                        ### Результат расчета:
                        - **Температура эксплуатации:** {T_celsius:.1f}°C
                        - При номере зерна: {G_input}
                        - Содержании сигма-фазы: {sigma_input:.1f}%
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