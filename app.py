import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for, send_file
from flask_session import Session
import tempfile
import math

# --- Импортируем необходимые модули для ИИ-прогноза ---
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf

# === Настройки Flask и сессии ===
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# === Русские подписи месяцев ===
MONTHS_RU = [
    "", "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
    "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"
]

GROUPS = {
    'year': 'Y',
    'quarter': 'Q',
    'month': 'M',
    'week': 'W',
    'day': 'D',
    'custom': None  # Для custom-периода берём group_by из запроса
}

def pretty_period_label(period, date):
    """Формирует подписи к периодам для графика и прогноза на русском языке."""
    if period == 'week':
        weeknum = int(date.strftime('%U'))
        return f"{date.year} / {weeknum} неделя"
    elif period == 'month':
        return f"{date.year} / {MONTHS_RU[date.month]}"
    elif period == 'quarter':
        return f"{date.year} / {((date.month-1)//3+1)} квартал"
    elif period == 'year':
        return f"{date.year} год"
    elif period == 'day':
        return date.strftime('%d.%m.%Y')
    else:
        return str(date)

def fix_types(obj):
    """Приводит все значения int64/float64 к int/float для корректного вывода в шаблонах."""
    if isinstance(obj, dict):
        return {k: fix_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj

def smart_ai_forecast(y, n_pred=3):
    """
    Строит прогноз временного ряда с помощью нейросети (Keras, LSTM или Dense).
    - y: список чисел (инцидентов по периодам)
    - n_pred: сколько точек вперёд предсказывать (обычно 3)
    Возвращает: массив из n_pred прогнозов (всегда >=0, без выбросов)
    """
    # Если данных меньше 8 — берём медиану последних 3-х точек
    if len(y) < 8:
        med = int(np.median(y[-3:])) if y else 0
        return [max(0, med)] * n_pred

    # Нормализация данных для обучения нейросети (диапазон [0,1])
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()

    # Создаём обучающие выборки: X — n_in точек, y — следующая точка
    n_in = min(6, len(y)-2)  # размер "окна" для прогноза
    X, y_train = [], []
    for i in range(len(y_scaled) - n_in):
        X.append(y_scaled[i:i + n_in])
        y_train.append(y_scaled[i + n_in])
    X = np.array(X)
    y_train = np.array(y_train)

    # Если данных больше 25 — используем LSTM, иначе Dense (чтобы работало и на малых массивах)
    if len(X) > 25:
        X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential()
        model.add(LSTM(12, input_shape=(n_in,1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_lstm, y_train, epochs=30, verbose=0)
    else:
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=n_in))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y_train, epochs=80, verbose=0)

    # Последние n_in точек — стартовое окно для прогноза
    last_window = y_scaled[-n_in:].reshape(1, -1)
    preds = []
    for _ in range(n_pred):
        if len(X) > 25:
            to_pred = last_window.reshape((1, n_in, 1))
        else:
            to_pred = last_window
        y_pred = model.predict(to_pred, verbose=0)[0, 0]
        # Ограничение: не может быть меньше 0
        y_pred = max(0, y_pred)
        # Делаем обратную нормализацию (возвращаем "реальные" значения)
        real_pred = int(round(scaler.inverse_transform([[y_pred]])[0, 0]))
        real_pred = max(0, real_pred)  # страховка от min/max
        preds.append(real_pred)
        # Добавляем прогноз к окну для следующего шага
        last_window = np.append(last_window[:, 1:], [[y_pred]], axis=1)
    return preds

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    # --- Загрузка/чтение файла из сессии ---
    if 'data' in session:
        df = pd.read_json(session['data'])
        filename = session.get('filename', 'файл.xlsx')
        columns = session.get('columns', list(df.columns))
    else:
        df = None
        filename = None
        columns = None

    # --- Загрузка нового файла ---
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if not file.filename:
            return render_template('dashboard.html', file_uploaded=False)
        tmp = tempfile.NamedTemporaryFile(delete=False)
        file.save(tmp.name)
        tmp.close()
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(tmp.name)
        else:
            df = pd.read_csv(tmp.name)
        os.unlink(tmp.name)
        filename = file.filename
        columns = list(df.columns)
        session['data'] = df.to_json()
        session['filename'] = filename
        session['columns'] = columns
        session['col_map'] = None
        return redirect(url_for('dashboard'))

    # --- Нет данных: только форма загрузки ---
    if df is None:
        return render_template('dashboard.html', file_uploaded=False)

    # --- Автоопределение или ручной выбор колонок ---
    col_map = session.get('col_map')
    if not col_map:
        guesses = {
            'date': next((c for c in columns if 'дат' in c.lower()), None),
            'service': next((c for c in columns if 'серв' in c.lower()), None),
            'reason': next((c for c in columns if 'причин' in c.lower()), None),
            'responsible': next((c for c in columns if 'ответ' in c.lower()), None)
        }
        if all(guesses.values()):
            col_map = guesses
            session['col_map'] = col_map
        else:
            return redirect(url_for('column_select'))

    # --- Приводим к стандартным именам столбцов ---
    df = df.rename(columns={
        col_map['date']: 'date',
        col_map['service']: 'service',
        col_map['reason']: 'reason',
        col_map['responsible']: 'responsible'
    })
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # --- Параметры фильтра и периода ---
    services = sorted(df['service'].unique())
    selected_service = request.args.get('service', 'all')
    selected_period = request.args.get('period', 'month')
    custom_start = request.args.get('start')
    custom_end = request.args.get('end')
    group_by = request.args.get('group_by', 'week')
    page = int(request.args.get('page', 1))

    # --- Фильтрация по сервису ---
    if selected_service != 'all':
        df = df[df['service'] == selected_service]

    # --- Группировка по периоду ---
    if selected_period == 'custom' and custom_start and custom_end:
        mask = (df['date'] >= custom_start) & (df['date'] <= custom_end)
        df_period = df.loc[mask]
        group_freq = GROUPS.get(group_by, 'W')
        if group_freq is None: group_freq = 'W'
        df_g = df_period.groupby(pd.Grouper(key='date', freq=group_freq)).size().reset_index(name='count')
        chart_labels = [pretty_period_label(group_by, d) for d in df_g['date']]
        period_for_chart = group_by
    else:
        # "Весь период" — история по месяцам по умолчанию
        group_freq = GROUPS.get(selected_period, 'M' if selected_period == 'all' else 'W')
        if group_freq is None: group_freq = 'M'
        df_g = df.groupby(pd.Grouper(key='date', freq=group_freq)).size().reset_index(name='count')
        period_for_chart = selected_period if selected_period != 'all' else 'month'
        chart_labels = [pretty_period_label(period_for_chart, d) for d in df_g['date']]

    # --- Реальный ИИ-прогноз через нейросеть ---
    show_forecast = len(df_g) > 2
    forecast_strs = []
    chart_forecast_labels = []
    chart_forecast = []
    if show_forecast:
        y_hist = [int(x) for x in df_g['count']]
        preds = smart_ai_forecast(y_hist, n_pred=3)
        last_dt = df_g['date'].max()
        # Строим подписи для будущих периодов (чтобы не было дубликатов, двигаем месяц/неделю и т.д.)
        for i, val in enumerate(preds):
            if period_for_chart == 'month':
                dt = last_dt + pd.DateOffset(months=i+1)
            elif period_for_chart == 'week':
                dt = last_dt + pd.DateOffset(weeks=i+1)
            elif period_for_chart == 'quarter':
                dt = last_dt + pd.DateOffset(months=(i+1)*3)
            elif period_for_chart == 'year':
                dt = last_dt + pd.DateOffset(years=i+1)
            elif period_for_chart == 'day':
                dt = last_dt + pd.DateOffset(days=i+1)
            else:
                dt = last_dt + pd.DateOffset(weeks=i+1)
            period_label = pretty_period_label(period_for_chart, dt)
            forecast_strs.append(f"{period_label} — {val} инцидентов")
            chart_forecast_labels.append(period_label)
        # В график — None на историю, только прогноз к прогнозу
        chart_forecast = [None] * len(y_hist) + preds

    # --- Строим "пироги", "бары", топы, рекомендации ---
    services_counts = df['service'].value_counts().head(5)
    services_pie = {
        'labels': list(services_counts.index),
        'values': [int(x) for x in services_counts.values]
    }
    reasons_counts = df['reason'].value_counts().head(5)
    reasons_bar = {
        'labels': list(reasons_counts.index),
        'values': [int(x) for x in reasons_counts.values]
    }
    responsibles = []
    for name, count in df['responsible'].value_counts().head(5).items():
        responsibles.append({'name': name, 'count': int(count)})

    recommendations = [
        "Проводите профилактику в периоды минимального числа инцидентов",
        "Следите за всплесками по конкретным причинам",
        "Назначайте ответственных согласно топ-5"
    ]

    # --- Пагинация (окно 7 кнопок) ---
    rows_per_page = 50
    total_rows = len(df)
    total_pages = max(1, math.ceil(total_rows / rows_per_page))
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    page_rows = df.iloc[start_idx:end_idx].copy()
    page_rows['date'] = page_rows['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    def pagination_window(current, max_page, window=7):
        half = window // 2
        if max_page <= window:
            return list(range(1, max_page + 1))
        elif current <= half + 1:
            return list(range(1, window)) + ["...", max_page]
        elif current >= max_page - half:
            return [1, "..."] + list(range(max_page - window + 2, max_page + 1))
        else:
            return [1, "..."] + list(range(current - half + 1, current + half)) + ["...", max_page]

    page_list = pagination_window(page, total_pages, 7)

    # --- Формируем объект данных для шаблона ---
    data = fix_types({
        'filename': filename,
        'rowcount': len(df),
        'services': services,
        'forecast': {
            'values': forecast_strs,
            'risk_service': services_pie['labels'][0] if services_pie['labels'] else 'Нет данных',
            'best_days': "Пн, Вт, Ср"
        },
        'responsibles': responsibles,
        'recommendations': recommendations,
        'services_pie': services_pie,
        'reasons_bar': reasons_bar,
        'table': page_rows.to_dict('records'),
        'total_pages': total_pages,
        'page': page,
        'total_rows': total_rows,
        'page_list': page_list,
        'chart': {
            'labels': chart_labels + chart_forecast_labels,
            'count': [int(x) for x in df_g['count']] + ([None] * len(chart_forecast_labels)),
            'forecast': [None] * len(df_g['count']) + (chart_forecast[-3:] if show_forecast else [])
        }
    })

    return render_template(
        'dashboard.html',
        file_uploaded=True,
        data=data,
        selected_service=selected_service,
        selected_period=selected_period,
        custom_start=custom_start,
        custom_end=custom_end,
        group_by=group_by,
        show_forecast=show_forecast,
        request=request
    )

@app.route('/column_select', methods=['GET', 'POST'])
def column_select():
    columns = session.get('columns', [])
    if request.method == 'POST':
        col_map = {
            'date': request.form['date_col'],
            'service': request.form['service_col'],
            'reason': request.form['reason_col'],
            'responsible': request.form['responsible_col']
        }
        session['col_map'] = col_map
        return redirect(url_for('dashboard'))
    preview = None
    if 'data' in session:
        df = pd.read_json(session['data'])
        preview = df.head(8).to_html(index=False)
    return render_template('column_select.html', columns=columns, col_map={}, preview=preview)

@app.route('/howto')
def howto():
    return render_template('howto.html')

@app.route('/export_csv')
def export_csv():
    if 'data' not in session:
        return redirect(url_for('dashboard'))
    df = pd.read_json(session['data'])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return send_file(tmp.name, as_attachment=True, download_name='incidents_export.csv')

if __name__ == "__main__":
    # Подавляем лишние логи TensorFlow для чистоты вывода
    tf.get_logger().setLevel('ERROR')
    app.run(debug=True)
