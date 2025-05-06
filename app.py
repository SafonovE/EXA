# -*- coding: utf-8 -*-


from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import pandas as pd
from prophet import Prophet
import re
from collections import Counter
from datetime import datetime

# -------------------------
# 1) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -------------------------

def find_column(df, keywords):
    """
    Ищем первую колонку, имя которой содержит любое слово из keywords.
    Нужно для автоматического определения:
      - колонки с датой
      - колонок с причинами
    """
    for col in df.columns:
        low = col.lower()
        for kw in keywords:
            if kw in low:
                return col
    return None

def find_incidents(df):
    """
    Ищем все коды инцидентов вида INT1234567 по всей табличке.
    Возвращаем отсортированный список уникальных инцидентов.
    """
    pat = re.compile(r'(INT\d{7})', re.IGNORECASE)
    found = set()
    for col in df.columns:
        hits = df[col].astype(str).str.extractall(pat)[0].dropna().unique()
        found.update(hits)
    return sorted(found)

def extract_combined_blame(df):
    """
    Блок «Виновники». Ищем в двух специальных колонках:
      - «Группа решателей»
      - «Отв. за ПР»
    Для каждого найденного имени:
      1) собираем текст всей строки (по метке idx)
      2) ищем в нём списком все коды инцидентов
      3) сохраняем в {имя: set(инцидентов)}
    """
    keys = ['группа решателей', 'отв. за пр']
    inc_pat = re.compile(r'INT\d{7}', re.IGNORECASE)
    result = {}
    for col in df.columns:
        if any(k in col.lower() for k in keys):
            for idx, val in df[col].astype(str).items():
                name = val.strip()
                if not name or name.lower() in {'nan','нет','прочее'}:
                    continue
                # Собираем весь текст строки по метке (loc), а не по позиции (iloc)
                row_text = ' '.join(df.loc[idx].astype(str))
                incs = inc_pat.findall(row_text)
                if incs:
                    result.setdefault(name, set()).update(incs)
    # Преобразуем множества во вложенные списки и сортируем
    return {n: sorted(v) for n,v in result.items()}

def extract_jira_links_with_incidents(df, base_url='https://jira.bcs.ru/browse/'):
    """
    Для блока «Jira-задачи»:
    1) находим ключи вида ABC-123
    2) для каждой задачи собираем список инцидентов из той же строки
    3) возвращаем список словарей:
       [ {'jira': 'ABC-123', 'url': 'https://...', 'incidents': ['INT0000001', ...]}, ... ]
    """
    jira_pat = re.compile(r'([A-Z]{2,}-\d+)')
    inc_pat  = re.compile(r'INT\d{7}', re.IGNORECASE)
    temp = {}  # {jira_key: set(incs)}
    for col in df.columns:
        # Ищем jira во всех колонках
        for idx, cell in df[col].astype(str).items():
            for jira in jira_pat.findall(cell):
                # строим полный URL
                url = f"{base_url}{jira}"
                # сразу ищем инциденты в той же строке
                row_text = ' '.join(df.loc[idx].astype(str))
                incs = inc_pat.findall(row_text)
                if incs:
                    temp.setdefault(jira, {'url': url, 'incs': set()})
                    temp[jira]['incs'].update(incs)
    # Приводим к списку словарей и спискам
    result = []
    for jira, info in temp.items():
        result.append({
            'jira': jira,
            'url': info['url'],
            'incidents': sorted(info['incs'])
        })
    # Сортируем по алфавиту jira-кода
    return sorted(result, key=lambda x: x['jira'])

# -------------------------
# 2) ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ
# -------------------------

def process_period(df, date_col, reason_cols,
                   period_key, from_date=None, to_date=None, group_by=None):
    """
    1) Фильтруем по выбранному периоду (week/month/quarter/year/custom/all)
    2) Группируем данные по нужной частоте (D/W/M/Q/Y)
    3) Считаем число инцидентов в каждом «бакете»
    4) Строим прогноз Prophet на столько же «бакетов» вперёд
    5) Собираем результаты для шаблона:
       - история / прогноз
       - инсайты
       - причины (текст + список инцидентов)
       - виновники
       - полный список инцидентов
       - jira-задачи с их инцидентами
    """
    latest = df[date_col].max()

    # 2.1) Определяем границы и шаг группировки
    if period_key == 'week':
        start, freq = latest - pd.Timedelta(days=7), 'D'
    elif period_key == 'month':
        start, freq = latest - pd.DateOffset(months=1), 'M'
    elif period_key == 'quarter':
        start, freq = latest - pd.DateOffset(months=3), 'Q'
    elif period_key == 'year':
        start, freq = latest - pd.DateOffset(years=1), 'Y'
    elif period_key == 'custom':
        try:
            start = pd.to_datetime(from_date)
            end   = pd.to_datetime(to_date)
        except:
            start = end = None
        freq = group_by if group_by in ('D','W','M','Q','Y') else 'D'
    else:  # 'all'
        start, freq = None, 'Y'

    # 2.2) Применяем фильтры по дате
    if period_key=='custom' and start and end:
        df_filt = df[(df[date_col]>=start)&(df[date_col]<=end)]
    elif start is not None:
        df_filt = df[df[date_col]>=start]
    else:
        df_filt = df.copy()

    # 2.3) Группируем по дате и считаем
    grouped = df_filt.groupby(pd.Grouper(key=date_col, freq=freq)).size().sort_index()
    if grouped.empty:
        return None
    history_df = pd.DataFrame({'date': grouped.index, 'count': grouped.values})

    # 2.4) Строим прогноз Prophet
    periods = len(history_df)
    model_df = history_df.rename(columns={'date':'ds','count':'y'})
    model_df['ds'] = pd.to_datetime(model_df['ds'])
    model_df = model_df.sort_values('ds')
    model = Prophet()
    model.fit(model_df)
    future   = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future).tail(periods)

    # Собираем метки и точки прогноза
    fc_labels = [d.strftime('%Y-%m-%d') for d in forecast['ds']]
    fc_points = [int(round(v)) for v in forecast['yhat']]

    # Текстовый прогноз, понятными словами
    unit_map = {'D':'дней','W':'недель','M':'месяцев','Q':'кварталов','Y':'лет'}
    unit = unit_map.get(freq, 'точек')
    avg  = int(round(forecast['yhat'].mean()))
    fc_text = f"Прогноз на следующие {periods} {unit}: в среднем ~{avg} инцидентов."

    # Инсайты: когда был пик
    peak_date = grouped.idxmax().strftime('%Y-%m-%d')
    peak_cnt  = int(grouped.max())
    insights  = [f"Пик: {peak_cnt} инцидентов ({peak_date})"]

    # ---------------------
    # 2.5) ДЕТАЛЬНЫЕ ПРИЧИНЫ
    # ---------------------
    # Собираем текст причин из всех reason_cols, склеивая через " / "
    reasons_series = df_filt[reason_cols].fillna('').astype(str).agg(' / '.join, axis=1)
    reason_map = {}  # {текст причины: set(incIDs)}
    inc_pat     = re.compile(r'INT\d{7}', re.IGNORECASE)
    for idx, text in reasons_series.items():
        reason = text.strip()
        if not reason:
            continue
        # ищем инциденты в той же строке
        row_txt = ' '.join(df_filt.loc[idx].astype(str))
        incs = inc_pat.findall(row_txt)
        if incs:
            reason_map.setdefault(reason, set()).update(incs)
    # Переводим множества в списки
    reasons = {r: sorted(v) for r,v in reason_map.items()}

    # ---------------------
    # 2.6) ВИНОВНИКИ
    # ---------------------
    blame = extract_combined_blame(df_filt)

    # ---------------------
    # 2.7) СПИСОК ИНЦИДЕНТОВ
    # ---------------------
    incidents = find_incidents(df_filt)

    # ---------------------
    # 2.8) JIRA-ЗАДАЧИ + СВЯЗАННЫЕ ИНЦИДЕНТЫ
    # ---------------------
    jira_with_incs = extract_jira_links_with_incidents(df_filt)

    # ---------------------
    # 2.9) Собираем всё в итоговый словарь
    # ---------------------
    return {
        'monthly': {
            'labels': [d.strftime('%Y-%m-%d') for d in history_df['date']],
            'points': [int(v) for v in history_df['count']]
        },
        'forecast_labels':  fc_labels,
        'forecast_points':  fc_points,
        'forecast':         fc_text,
        'insights':         insights,
        'reasons':          reasons,       # {текст: [inc1,inc2,...]}
        'blame':            blame,         # {имя: [inc1,inc2,...]}
        'incidents':        incidents,     # [inc1, inc2, ...]
        'jira':             jira_with_incs # [{'jira':..., 'url':..., 'incidents':[...]}]
    }

# -------------------------
# 3) НАСТРОЙКА FLASK
# -------------------------
app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SESSION_TYPE']      = 'filesystem'
app.config['SESSION_FILE_DIR']  = './.flask_session/'
app.config['SESSION_PERMANENT'] = False
Session(app)

# Словарь «ключ»:«название кнопки»
PERIODS = {
    'week':    'Неделя',
    'month':   'Месяц',
    'quarter': 'Квартал',
    'year':    'Год',
    'custom':  'Свой период',
    'all':     'Все время'
}

# -------------------------
# 4) ГЛАВНЫЙ МАРШРУТ "/"
# -------------------------

@app.route('/', methods=['GET','POST'])
def dashboard():
    # читаем параметры из URL
    period    = request.args.get('period', 'month')
    from_date = request.args.get('from_date')
    to_date   = request.args.get('to_date')
    group_by  = request.args.get('group_by', 'D')

    # --- если пришёл POST с файлом, сразу сохраняем и редиректим на GET ---
    if request.method=='POST' and 'file' in request.files:
        f = request.files['file']
        try:
            df = pd.read_excel(f)
            session['raw_data'] = df.to_json(orient='split', date_format='iso')
            return redirect(url_for('dashboard',
                                    period=period,
                                    from_date=from_date,
                                    to_date=to_date,
                                    group_by=group_by))
        except:
            # при ошибке чтения — всё равно покажем форму без дашборда
            return render_template('dashboard.html',
                                   data=None,
                                   periods=PERIODS,
                                   selected=period,
                                   from_date=from_date,
                                   to_date=to_date,
                                   group_by=group_by)

    # --- если в сессии нет данных, показываем только форму загрузки ---
    if 'raw_data' not in session:
        return render_template('dashboard.html',
                               data=None,
                               periods=PERIODS,
                               selected=period,
                               from_date=from_date,
                               to_date=to_date,
                               group_by=group_by)

    # --- восстанавливаем DataFrame из сессии ---
    try:
        df = pd.read_json(session['raw_data'], orient='split')
    except:
        return render_template('dashboard.html',
                               data=None,
                               periods=PERIODS,
                               selected=period,
                               from_date=from_date,
                               to_date=to_date,
                               group_by=group_by)

    # --- автоматический поиск нужных колонок ---
    date_col      = find_column(df, ['дата','date'])
    reason_cands  = ['причина','категория шаблона','тип инцидента']
    reason_cols   = [c for c in df.columns if any(k in c.lower() for k in reason_cands)]
    if not date_col or not reason_cols:
        # если не нашли дату или причины — нечего строить
        return render_template('dashboard.html',
                               data=None,
                               periods=PERIODS,
                               selected=period,
                               from_date=from_date,
                               to_date=to_date,
                               group_by=group_by)

    # --- приводим колонки с датами к реальному типу datetime ---
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    df = df.dropna(subset=[date_col])  # убираем строки без даты

    # --- обрабатываем весь датасет под выбранный период и группировку ---
    processed = process_period(df,
                               date_col,
                               reason_cols,
                               period,
                               from_date,
                               to_date,
                               group_by)
    if not processed:
        # если по выбранным фильтрам ничего не осталось
        return render_template('dashboard.html',
                               data=None,
                               periods=PERIODS,
                               selected=period,
                               from_date=from_date,
                               to_date=to_date,
                               group_by=group_by)

    # --- отрисовываем dashboard.html с готовыми данными ---
    return render_template('dashboard.html',
                           data=processed,
                           periods=PERIODS,
                           selected=period,
                           from_date=from_date,
                           to_date=to_date,
                           group_by=group_by)

if __name__=='__main__':
    # debug=True — для локальной разработки,
    # в продакшене выключите!
    app.run(debug=True)
