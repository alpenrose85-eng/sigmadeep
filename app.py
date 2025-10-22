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

class AdvancedSigmaPhaseAnalyzer:
    def __init__(self):
        self.params = None
        self.R2 = None
        self.rmse = None
        self.mape = None
        self.model_type = None
        self.creation_date = datetime.now().isoformat()
        
    def fit_avrami_model(self, data):
        """Модель Аврами с насыщением"""
        try:
            G = data['G'].values
            T = data['T'].values + 273.15
            t = data['t'].values
            f_exp = data['f_exp (%)'].values
            
            # Начальные приближения
            # f_max, K0, Q, n, alpha
            initial_guess = [10.0, 1e10, 200000, 1.0, 0.1]
            bounds = (
                [1.0, 1e5, 100000, 0.1, -1.0],
                [20.0, 1e15, 400000, 3.0, 1.0]
            )
            
            def model(params, G, T, t):
                f_max, K0, Q, n, alpha = params
                R = 8.314
                
                # Влияние размера зерна
                grain_effect = 1 + alpha * (G - 8)  # Центрируем вокруг G=8
                
                # Кинетика
                K = K0 * np.exp(-Q / (R * T)) * grain_effect
                f_pred = f_max * (1 - np.exp(-K * (t ** n)))
                return f_pred
            
            self.params, _ = curve_fit(
                lambda x, f_max, K0, Q, n, alpha: model([f_max, K0, Q, n, alpha], G, T, t),
                np.arange(len(G)), f_exp,
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000
            )
            
            # Расчет метрик
            f_pred = model(self.params, G, T, t)
            self.R2 = r2_score(f_exp, f_pred)
            self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
            self.mape = np.mean(np.abs((f_exp - f_pred) / f_exp)) * 100
            self.model_type = "avrami_saturation"
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка в модели Аврами: {e}")
            return False
    
    def fit_power_law_model(self, data):
        """Степенная модель с температурной зависимостью"""
        try:
            G = data['G'].values
            T = data['T'].values + 273.15
            t = data['t'].values
            f_exp = data['f_exp (%)'].values
            
            # A, B, C, D, E - коэффициенты
            initial_guess = [1.0, 0.1, -10000, 0.5, 0.01]
            bounds = (
                [0.1, -1.0, -50000, 0.1, -0.1],
                [10.0, 1.0, -1000, 2.0, 0.1]
            )
            
            def model(params, G, T, t):
                A, B, C, D, E = params
                R = 8.314
                
                # Комбинированная модель
                temp_effect = np.exp(C / (R * T))
                time_effect = t ** D
                grain_effect = 1 + E * (G - 8)
                
                f_pred = A * temp_effect * time_effect * grain_effect + B
                return np.clip(f_pred, 0, 20)  # Ограничиваем разумными значениями
            
            self.params, _ = curve_fit(
                lambda x, A, B, C, D, E: model([A, B, C, D, E], G, T, t),
                np.arange(len(G)), f_exp,
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000
            )
            
            # Расчет метрик
            f_pred = model(self.params, G, T, t)
            self.R2 = r2_score(f_exp, f_pred)
            self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
            self.mape = np.mean(np.abs((f_exp - f_pred) / f_exp)) * 100
            self.model_type = "power_law"
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка в степенной модели: {e}")
            return False
    
    def fit_logistic_model(self, data):
        """Логистическая модель роста"""
        try:
            G = data['G'].values
            T = data['T'].values + 273.15
            t = data['t'].values
            f_exp = data['f_exp (%)'].values
            
            # f_max, k, t0, alpha, beta
            initial_guess = [8.0, 1e-4, 1000, 0.1, -10000]
            bounds = (
                [1.0, 1e-8, 100, -1.0, -50000],
                [15.0, 1e-2, 10000, 1.0, -1000]
            )
            
            def model(params, G, T, t):
                f_max, k, t0, alpha, beta = params
                R = 8.314
                
                # Температурная зависимость скорости
                temp_factor = np.exp(beta / (R * T))
                grain_factor = 1 + alpha * (G - 8)
                
                # Логистический рост
                rate = k * temp_factor * grain_factor
                f_pred = f_max / (1 + np.exp(-rate * (t - t0)))
                return f_pred
            
            self.params, _ = curve_fit(
                lambda x, f_max, k, t0, alpha, beta: model([f_max, k, t0, alpha, beta], G, T, t),
                np.arange(len(G)), f_exp,
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000
            )
            
            # Расчет метрик
            f_pred = model(self.params, G, T, t)
            self.R2 = r2_score(f_exp, f_pred)
            self.rmse = np.sqrt(mean_squared_error(f_exp, f_pred))
            self.mape = np.mean(np.abs((f_exp - f_pred) / f_exp)) * 100
            self.model_type = "logistic"
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка в логистической модели: {e}")
            return False
    
    def fit_ensemble_model(self, data):
        """Ансамблевая модель - комбинация нескольких подходов"""
        try:
            G = data['G'].values
            T = data['T'].values + 273.15
            t = data['t'].values
            f_exp = data['f_exp (%)'].values
            
            # Параметры для ансамбля
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
            
            return True
            
        except Exception as e:
            st.error(f"Ошибка в ансамблевой модели: {e}")
            return False
    
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
    
    def _predict_temperature_analytic(self, G, sigma, t):
        """Аналитическое решение (где возможно)"""
        if self.model_type == "avrami_saturation":
            f_max, K0, Q, n, alpha = self.params
            R = 8.314
            
            grain_effect = 1 + alpha * (G - 8)
            K_eff = K0 * grain_effect
            
            if sigma >= f_max or sigma <= 0:
                return None
            
            term = -np.log(1 - sigma / f_max) / (K_eff * (t ** n))
            if term <= 0:
                return None
            
            T_kelvin = -Q / (R * np.log(term))
            return T_kelvin - 273.15
        
        return self._predict_temperature_bisection(G, sigma, t)
    
    def _evaluate_model(self, G, T, t):
        """Вычисление модели для данных параметров"""
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
            
            # Аврами компонент
            grain_effect_avrami = 1 + alpha * (G - 8)
            K_avrami = K0 * np.exp(-Q / (R * T_kelvin)) * grain_effect_avrami
            f_avrami = f_max * (1 - np.exp(-K_avrami * (t ** n)))
            
            # Степенной компонент
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
        
        # Фильтрация бесконечных значений
        valid_mask = np.isfinite(relative_errors) & (f_exp > 0.1)
        f_exp_valid = f_exp[valid_mask]
        f_pred_valid = f_pred[valid_mask]
        residuals_valid = residuals[valid_mask]
        relative_errors_valid = relative_errors[valid_mask]
        
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

def main():
    # Инициализация сессии
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    
    # Боковая панель
    st.sidebar.header("🎯 Выбор модели")
    
    model_type = st.sidebar.selectbox(
        "Тип физической модели",
        ["avrami_saturation", "power_law", "logistic", "ensemble"],
        format_func=lambda x: {
            "avrami_saturation": "Аврами с насыщением",
            "power_law": "Степенная модель", 
            "logistic": "Логистический рост",
            "ensemble": "Ансамблевая модель"
        }[x],
        help="Выберите физическую модель, наиболее подходящую для ваших данных"
    )
    
    # Пример данных
    sample_data = pd.DataFrame({
        'G': [8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        'T': [600, 600, 600, 600, 650, 650, 650, 650, 600, 600, 600, 600, 650, 650, 650, 650, 600, 600, 600, 600, 650, 650, 650, 650, 700, 700],
        't': [2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000],
        'f_exp (%)': [1.76, 0.68, 0.94, 1.09, 0.67, 1.2, 1.48, 1.13, 0.87, 1.28, 2.83, 3.25, 1.88, 2.29, 3.25, 2.89, 1.261, 2.04, 2.38, 3.3, 3.2, 4.26, 5.069, 5.41, 3.3, 5.0]
    })
    
    # Основной интерфейс
    st.header("🔧 Улучшенное моделирование сигма-фазы")
    
    st.info("""
    **Проблема предыдущих моделей:** Слишком высокие ошибки предсказания (MAPE > 40%)
    
    **Новые подходы:**
    - Модель Аврами с ограничением насыщения
    - Степенная модель с температурной зависимостью  
    - Логистическая модель роста
    - Ансамблевая комбинированная модель
    """)
    
    # Используем примерные данные
    st.session_state.current_data = sample_data
    
    st.header("📊 Данные для анализа")
    st.dataframe(st.session_state.current_data, use_container_width=True)
    
    # Подбор модели
    st.header("🎯 Подбор параметров модели")
    
    if st.button("🚀 Запустить подбор параметров", use_container_width=True):
        analyzer = AdvancedSigmaPhaseAnalyzer()
        
        with st.spinner("Подбираем параметры модели..."):
            if model_type == "avrami_saturation":
                success = analyzer.fit_avrami_model(st.session_state.current_data)
            elif model_type == "power_law":
                success = analyzer.fit_power_law_model(st.session_state.current_data)
            elif model_type == "logistic":
                success = analyzer.fit_logistic_model(st.session_state.current_data)
            else:  # ensemble
                success = analyzer.fit_ensemble_model(st.session_state.current_data)
        
        if success:
            st.session_state.analyzer = analyzer
            validation_results = analyzer.calculate_validation_metrics(st.session_state.current_data)
            st.session_state.validation_results = validation_results
            
            st.success(f"✅ Модель успешно обучена! R² = {analyzer.R2:.4f}")
            
            # Показ параметров
            st.subheader("📈 Параметры модели")
            if analyzer.model_type == "avrami_saturation":
                f_max, K0, Q, n, alpha = analyzer.params
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("f_max", f"{f_max:.2f}%")
                with col2: st.metric("K₀", f"{K0:.2e}")
                with col3: st.metric("Q", f"{Q/1000:.1f} кДж/моль")
                with col4: st.metric("n", f"{n:.3f}")
                with col5: st.metric("α", f"{alpha:.3f}")
                
    # Валидация
    if st.session_state.validation_results is not None:
        st.header("📊 Результаты валидации")
        
        validation = st.session_state.validation_results
        metrics = validation['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("R²", f"{metrics['R2']:.4f}")
        with col2: st.metric("MAE", f"{metrics['MAE']:.3f}%")
        with col3: st.metric("RMSE", f"{metrics['RMSE']:.3f}%")
        with col4: st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        
        # Оценка качества
        if metrics['MAPE'] < 15:
            st.success("✅ Отличное качество модели!")
        elif metrics['MAPE'] < 25:
            st.warning("⚠️ Удовлетворительное качество модели")
        else:
            st.error("❌ Рекомендуется попробовать другую модель")
            
        # График
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=validation['data']['f_exp (%)'],
            y=validation['predictions'],
            mode='markers',
            name='Предсказания',
            marker=dict(size=10, color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=[0, 6], y=[0, 6],
            mode='lines',
            name='Идеально',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='Предсказание vs Эксперимент',
            xaxis_title='Экспериментальные значения (%)',
            yaxis_title='Расчетные значения (%)'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
