from prophet import Prophet
import os
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
from datetime import datetime
from collections import Counter
import re

def find_incidents(df):
    incident_pattern = re.compile(r'INT\d{7}', re.IGNORECASE)
    matches = set()
    for col in df.columns:
        matches.update(df[col].astype(str).str.extractall(r'(INT\d{7})')[0].dropna().unique())
    return sorted(matches)

def find_jira_links(df, jira_base_url='https://jira.bcs.ru/browse/'):
    jira_pattern = re.compile(r'[A-Z]{2,}-\d+')
    matches = set()
    for col in df.columns:
        matches.update(df[col].astype(str).str.extractall(r'([A-Z]{2,}-\d+)')[0].dropna().unique())
    return [f"{jira_base_url}{key}" for key in sorted(matches)]

app = Flask(__name__)
app.secret_key = 'your-strong-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_PERMANENT'] = False
Session(app)

PERIODS = {
    'week': 'Неделя',
    'month': 'Месяц',
    'quarter': 'Квартал',
    'year': 'Год',
    'custom': 'Свой период',
    'all': 'Все время'
}

def generate_forecast(monthly_df):
    if monthly_df.shape[0] < 2:
        return "Недостаточно данных для прогноза."

    df = monthly_df.rename(columns={"date": "ds", "count": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    if df["ds"].nunique() < 3:
        df = df.set_index("ds").resample("W").sum().reset_index()
        period_text = "неделю"
        freq = 'W'
    else:
        df = df.set_index("ds").resample("D").sum().reset_index()
        period_text = "день"
        freq = 'D'

    future_periods = 3

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=future_periods, freq=freq)
    forecast = model.predict(future)
    future_points = forecast.tail(future_periods)

    predicted_avg = int(round(future_points["yhat"].mean()))
    predicted_range = (
        int(round(future_points["yhat_lower"].min())),
        int(round(future_points["yhat_upper"].max()))
    )
    date_start = future_points["ds"].min().date()

    last_known = df.iloc[-1]["y"]
    delta = predicted_avg - last_known

    if delta > 5:
        trend = "ожидается рост"
    elif delta < -5:
        trend = "ожидается снижение"
    else:
        trend = "сохраняется стабильность"

    confidence_note = ""
    if df.shape[0] < 5:
        confidence_note = "⚠️ Прогноз построен на ограниченном числе точек. Используйте с осторожностью. "

    return (
        f"{confidence_note}"
        f"📅 С {date_start} ожидается в среднем ~{predicted_avg} инцидентов в {period_text} "
        f"(от {predicted_range[0]} до {predicted_range[1]}). "
        f"📊 По тренду: {trend}."
    )

def find_column(df, keywords):
    for col in df.columns:
        for keyword in keywords:
            if keyword in col.lower():
                return col
    return None

def extract_top_counts(series, top_n=5):
    values = series.dropna().astype(str).str.strip().str.lower()
    values = values[~values.isin(['', 'nan', 'нет', 'нет данных', '-', '—', 'тест', 'прочее'])]
    counts = Counter(values)
    return dict(counts.most_common(top_n))

def extract_combined_blame(df):
    blame_candidates = ['группа решателей', 'отв. за пр', 'подразделение пр', 'подразделение ор']
    incident_pattern = re.compile(r'INT\d{7}', re.IGNORECASE)
    blame_dict = {}

    for col in df.columns:
        if any(key in col.lower() for key in blame_candidates):
            blame_col = df[col].astype(str).str.strip().str.lower()
            for i, value in enumerate(blame_col):
                if value and value not in ['nan', '', 'нет', 'прочее']:
                    incident_row = ' '.join(df.iloc[i].astype(str))
                    incidents = incident_pattern.findall(incident_row)
                    if incidents:
                        if value not in blame_dict:
                            blame_dict[value] = set()
                        blame_dict[value].update(incidents)

    return {k: sorted(v) for k, v in blame_dict.items()}

def process_period(df, date_col, reason_col, period_key):
    latest_date = df[date_col].max()
    if period_key == 'week':
        df_filtered = df[df[date_col] >= latest_date - pd.Timedelta(weeks=1)]
        freq = 'D'
    elif period_key == 'month':
        df_filtered = df[df[date_col] >= latest_date - pd.DateOffset(months=1)]
        freq = 'D'
    elif period_key == 'quarter':
        df_filtered = df[df[date_col] >= latest_date - pd.DateOffset(months=3)]
        freq = 'W'
    elif period_key == 'year':
        df_filtered = df[df[date_col] >= latest_date - pd.DateOffset(years=1)]
        freq = 'M'
    elif period_key == 'custom':
        from_str = request.args.get('from')
        to_str = request.args.get('to')
        try:
            from_date = pd.to_datetime(from_str)
            to_date = pd.to_datetime(to_str)
            df_filtered = df[(df[date_col] >= from_date) & (df[date_col] <= to_date)]
        except:
            df_filtered = df.copy()
        freq = 'D'
    else:
        df_filtered = df.copy()
        freq = 'M'

    grouped = df_filtered.groupby(pd.Grouper(key=date_col, freq=freq)).size().sort_index()
    if grouped.empty:
        return None

    monthly_df = pd.DataFrame({
        "date": grouped.index,
        "count": grouped.values
    })

    forecast_text = generate_forecast(monthly_df)

    insights = []
    if not grouped.empty:
        max_date = grouped.idxmax().strftime('%Y-%m-%d')
        max_count = grouped.max()
        insights.append(f"📅 Пик инцидентов: {max_count} — {max_date}")

    reasons = extract_top_counts(df_filtered[reason_col]) if reason_col else {}
    blame = extract_combined_blame(df_filtered)

    return {
        'monthly': {
            'labels': [x.strftime('%Y-%m-%d') for x in grouped.index],
            'points': [int(v) for v in grouped.values]
        },
        'forecast': forecast_text,
        'insights': insights,
        'blame': blame,
        'reasons': reasons
    }

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    selected_period = request.args.get('period', 'month')

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        try:
            df = pd.read_excel(file)
            session['raw_data'] = df.to_json(orient='split', date_format='iso')
            return redirect(url_for('dashboard', period=selected_period))
        except Exception:
            return render_template("dashboard.html", data=None, periods=PERIODS, selected=selected_period)

    if 'raw_data' not in session:
        return render_template("dashboard.html", data=None, periods=PERIODS, selected=selected_period)

    try:
        df = pd.read_json(session['raw_data'], orient='split')
    except Exception:
        return render_template("dashboard.html", data=None, periods=PERIODS, selected=selected_period)

    date_col = find_column(df, ['дата', 'date'])
    reason_col = find_column(df, ['причин', 'ошибк'])

    if not date_col:
        return render_template("dashboard.html", data=None, periods=PERIODS, selected=selected_period)

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[date_col])

    processed = process_period(df, date_col, reason_col, selected_period)
    if not processed:
        return render_template("dashboard.html", data=None, periods=PERIODS, selected=selected_period)

    processed["incidents"] = find_incidents(df)
    processed["jira_links"] = find_jira_links(df)

    from_date = request.args.get("from")
    to_date = request.args.get("to")

    return render_template("dashboard.html", data=processed, periods=PERIODS,
                           selected=selected_period, from_date=from_date, to_date=to_date)

if __name__ == '__main__':
    app.run(debug=True)
