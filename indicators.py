def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

def atr(df, n=14):
    data = df.copy()
    high = data['High']
    low = data['Low']
    close = data['Close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr

def vmbo(df):
    mv_avg_p = 9
    atr_p = 66

    mv_avg = wwma(df, mv_avg_p)
    norm_atr = atr(df, atr_p)

    close_vs_ma = df['Close'] - mv_avg
    vbc_vs_ma = close_vs_ma / norm_atr

    low_vs_ma = df['Low'] - mv_avg
    vbl_vs_ma = low_vs_ma / norm_atr

    high_vs_ma = df['High'] / mv_avg
    vbh_vs_ma = high_vs_ma / norm_atr

    return vbc_vs_ma, vbl_vs_ma, vbh_vs_ma