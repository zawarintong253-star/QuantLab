import streamlit as st
import pandas as pd
import numpy as np
import baostock as bs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==========================================
# 1. 页面配置与黑金 CSS (Black Gold UI)
# ==========================================
st.set_page_config(
    page_title="Leo Quant Lab V2.0",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入深度定制 CSS：实现“黑金私募”质感
st.markdown("""
<style>
    /* 全局背景纯黑 */
    .stApp {
        background-color: #000000;
        color: #E0E0E0;
    }
    
    /* 侧边栏背景深灰 */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #333;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        background-color: #1A1A1A;
        color: #D4AF37;
        border: 1px solid #333;
    }
    .stNumberInput > div > div > input {
        background-color: #1A1A1A;
        color: #D4AF37;
    }
    
    /* 下拉框样式 */
    .stSelectbox > div > div {
        background-color: #1A1A1A;
        color: #D4AF37;
    }
    
    /* 按钮样式：黑底金边 */
    .stButton > button {
        background-color: #000000;
        color: #D4AF37;
        border: 1px solid #D4AF37;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #D4AF37;
        color: #000000;
    }
    
    /* 关键指标 Metric 卡片样式 */
    div[data-testid="metric-container"] {
        background-color: #111111;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0,0,0,0.8);
    }
    label[data-testid="stMetricLabel"] {
        color: #888888 !important;
        font-size: 0.8rem;
    }
    div[data-testid="stMetricValue"] {
        color: #D4AF37 !important; /* 金黄色数值 */
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    div[data-testid="stMetricDelta"] {
        color: #aaa !important;
    }
    
    /* 隐藏 Streamlit 默认顶部红线与菜单 */
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 法律声明红色警示框 */
    .legal-warning {
        color: #ff4444;
        font-weight: bold;
        border: 1px solid #ff4444;
        padding: 15px;
        background-color: #220000;
        border-radius: 5px;
        margin-top: 20px;
        font-size: 13px;
        text-align: center;
    }
    
    /* 标题金字 */
    h1, h2, h3 {
        color: #D4AF37 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 分割线颜色 */
    hr {
        border-color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 授权墙逻辑 (Gatekeeper)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login_wall():
    with st.sidebar:
        st.title("🏛️ Leo Quant Lab")
        st.caption("Professional Edition V2.0")
        st.markdown("---")
        
        st.write("### 🔒 终端授权")
        pwd = st.text_input("Access Code", type="password", placeholder="请输入授权码")
        
        # 法律免责声明 (强制红色)
        st.markdown("""
        <div class="legal-warning">
        ⛔ 法律免责声明：<br><br>
        本系统仅供量化策略逻辑研究与教学使用。<br>
        系统生成的所有数据、信号均不构成任何投资建议。<br>
        股市有风险，入市需谨慎，风险自担。
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("解锁系统"):
            if pwd == "LEO666":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ 授权码错误")
        
    # 主界面遮罩
    st.markdown("""
    <div style='text-align: center; padding-top: 150px;'>
        <h1 style='color: #333 !important; font-size: 60px;'>SYSTEM LOCKED</h1>
        <p style='color: #666; font-size: 20px;'>PLEASE AUTHENTICATE VIA SIDEBAR</p>
        <p style='color: #444;'>Leo Quant Research Lab © 2024</p>
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.logged_in:
    login_wall()
    st.stop()  # 阻断后续代码执行

# ==========================================
# 3. 数据引擎 (BaoStock Engine)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data_bs(symbol, start_date, end_date):
    """BaoStock 数据获取与清洗"""
    try:
        bs.login()
        
        # 自动补全代码后缀逻辑
        code = str(symbol).strip()
        bs_code = ""
        # 如果已经包含后缀
        if code.startswith(('sh.', 'sz.', 'bj.')):
            bs_code = code
        else:
            # 智能推断
            if code.startswith(('6', '5', '9')): prefix = 'sh.'
            elif code.startswith(('0', '3')): prefix = 'sz.'
            elif code.startswith(('8', '4')): prefix = 'bj.'
            else: prefix = 'sh.' # 默认沪市
            bs_code = f"{prefix}{code}"

        # 获取日线
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3" # 不复权，保持价格直观
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        bs.logout()
        
        if not data_list:
            return None, "无数据"
            
        df = pd.DataFrame(data_list, columns=["date", "open", "high", "low", "close", "volume"])
        df['date'] = pd.to_datetime(df['date'])
        
        # 强制转浮点
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
            
        df = df.set_index('date').sort_index()
        # 获取名称简单处理
        name = bs_code
        return df, name
    except Exception as e:
        return None, str(e)

# ==========================================
# 4. 策略逻辑库 (Strategy Library)
# ==========================================
STRATEGY_MAP = {
    "MA Trend Filter (均线趋势过滤)": "MA_Filter",
    "Dual MA Cross (双均线交叉)": "MA_Cross",
    "RSI Mean Reversion (RSI超跌反转)": "RSI_Reversion",
    "Donchian Channel (唐奇安通道)": "Donchian_Breakout",
    "Bollinger Squeeze (布林带收口)": "Bollinger_Squeeze",
    "Grid Trading (网格波动套利)": "Grid_Trading",
    "BIAS Reversion (乖离率反转)": "BIAS_Reversion"
}

def get_strategy_doc(code, p):
    """ 生成策略说明书 """
    if code == "MA_Filter":
        return f"**买入**：收盘 > {p['ma_long']}日均线\n**卖出**：收盘 < {p['ma_long']}日均线"
    if code == "MA_Cross":
        return f"**买入**：{p['ma_short']}日快线金叉{p['ma_long']}日慢线\n**卖出**：快线死叉慢线"
    if code == "RSI_Reversion":
        return f"**买入**：RSI < {p['lower_bound']} (超跌)\n**卖出**：RSI > {p['upper_bound']} (超买)"
    if code == "Donchian_Breakout":
        return f"**买入**：突破过去{p['channel_period']}日最高价\n**卖出**：跌破过去{p['channel_period']//2}日最低价"
    if code == "Bollinger_Squeeze":
        return f"**买入**：突破布林上轨 (压力位: {p['std_dev']}倍标准差)\n**卖出**：跌回中轨"
    if code == "Grid_Trading":
        return f"**锚点**：基于初始价格\n**逻辑**：每跌{p['grid_size']:.1%}加仓，每涨{p['grid_size']:.1%}减仓"
    if code == "BIAS_Reversion":
        return f"**买入**：乖离率 < -{p['bias_th']}%\n**卖出**：乖离率 > {p['bias_th']}%"
    return "暂无说明"

def run_strategy_logic(df, code, p):
    df = df.copy()
    df['Signal'] = 0
    
    if code == "MA_Filter":
        df['MA'] = df['close'].rolling(p['ma_long']).mean()
        df.loc[df['close'] > df['MA'], 'Signal'] = 1
        
    elif code == "MA_Cross":
        df['MA_S'] = df['close'].rolling(p['ma_short']).mean()
        df['MA_L'] = df['close'].rolling(p['ma_long']).mean()
        df.loc[df['MA_S'] > df['MA_L'], 'Signal'] = 1
        
    elif code == "RSI_Reversion":
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(p['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        sig = np.zeros(len(df)); pos = 0
        rsi_vals = df['RSI'].values
        for i in range(1, len(df)):
            if rsi_vals[i] < p['lower_bound']: pos = 1
            elif rsi_vals[i] > p['upper_bound']: pos = 0
            sig[i] = pos
        df['Signal'] = sig
        
    elif code == "Donchian_Breakout":
        win = p['channel_period']
        df['Up'] = df['high'].rolling(win).max().shift(1)
        df['Dn'] = df['low'].rolling(int(win//2)).min().shift(1)
        sig = np.zeros(len(df)); pos = 0
        closes = df['close'].values
        ups = df['Up'].values
        dns = df['Dn'].values
        for i in range(1, len(df)):
            if closes[i] > ups[i]: pos = 1
            elif closes[i] < dns[i]: pos = 0
            sig[i] = pos
        df['Signal'] = sig

    elif code == "Bollinger_Squeeze":
        df['MA20'] = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['Up'] = df['MA20'] + p['std_dev'] * std
        df.loc[df['close'] > df['Up'], 'Signal'] = 1
        df.loc[df['close'] < df['MA20'], 'Signal'] = 0
        df['Signal'] = df['Signal'].fillna(method='ffill')

    elif code == "Grid_Trading":
        grid = p['grid_size']
        sig = np.zeros(len(df)); last_p = df['close'].iloc[0]; pos = 0
        closes = df['close'].values
        for i in range(1, len(df)):
            if closes[i] <= last_p * (1 - grid): pos = 1; last_p = closes[i]
            elif closes[i] >= last_p * (1 + grid): pos = 0; last_p = closes[i]
            sig[i] = pos
        df['Signal'] = sig

    elif code == "BIAS_Reversion":
        ma = df['close'].rolling(20).mean()
        bias = (df['close'] - ma) / ma * 100
        df['BIAS'] = bias
        df.loc[bias < -p['bias_th'], 'Signal'] = 1
        df.loc[bias > p['bias_th'], 'Signal'] = 0
        df['Signal'] = df['Signal'].fillna(method='ffill')

    df['Signal'] = df['Signal'].fillna(0)
    return df

# ==========================================
# 5. 回测与指标内核 (Backtest Engine)
# ==========================================
def run_backtest_core(df, initial_capital, commission_rate):
    # 基础涨跌幅
    df['Pct_Change'] = df['close'].pct_change().fillna(0)
    
    # 策略收益 (T+1)
    df['Strategy_Ret'] = df['Signal'].shift(1) * df['Pct_Change']
    
    # 扣费 (信号变动)
    df['Trade_Flag'] = df['Signal'].diff().abs().fillna(0)
    df['Cost'] = df['Trade_Flag'] * commission_rate
    
    # 净值
    df['Net_Ret'] = df['Strategy_Ret'] - df['Cost']
    df['Equity'] = (1 + df['Net_Ret']).cumprod() * initial_capital
    df['Benchmark'] = (1 + df['Pct_Change']).cumprod() * initial_capital
    
    return df

def calc_4x4_metrics(df, initial_capital, risk_free_rate=0.02):
    """计算专业 4x4 指标矩阵"""
    try:
        days = (df.index[-1] - df.index[0]).days
        years = max(days / 365, 1/365)
        
        total_ret = (df['Equity'].iloc[-1] / initial_capital) - 1
        ann_ret = (1 + total_ret) ** (1 / years) - 1
        
        # Alpha/Beta
        strat_daily = df['Net_Ret'].fillna(0)
        bench_daily = df['Pct_Change'].fillna(0)
        if bench_daily.var() != 0:
            cov = np.cov(strat_daily, bench_daily)[0, 1]
            beta = cov / bench_daily.var()
            bench_ann = (df['Benchmark'].iloc[-1] / initial_capital) ** (1 / years) - 1
            alpha = ann_ret - (risk_free_rate + beta * (bench_ann - risk_free_rate))
        else:
            beta, alpha = 0, 0
            
        # 风险
        vol = strat_daily.std() * np.sqrt(250)
        sharpe = (ann_ret - risk_free_rate) / vol if vol > 0 else 0
        
        roll_max = df['Equity'].cummax()
        max_dd = ((df['Equity'] - roll_max) / roll_max).min()
        
        downside = strat_daily[strat_daily < 0]
        sortino = (ann_ret - risk_free_rate) / (downside.std() * np.sqrt(250)) if not downside.empty else 0
        
        # 交易统计
        trade_count = int(df['Trade_Flag'].sum() / 2)
        total_fees = df['Cost'].sum()
        
        # 盈亏比
        df['trade_id'] = (df['Signal'].diff() != 0).cumsum()
        trade_rets = df[df['Signal'] == 1].groupby('trade_id')['Net_Ret'].sum()
        if len(trade_rets) > 0:
            win_rate = len(trade_rets[trade_rets > 0]) / len(trade_rets)
            avg_win = trade_rets[trade_rets > 0].mean() if not trade_rets[trade_rets > 0].empty else 0
            avg_loss = abs(trade_rets[trade_rets <= 0].mean()) if not trade_rets[trade_rets <= 0].empty else 1e-6
            pl_ratio = avg_win / avg_loss
        else:
            win_rate, pl_ratio = 0, 0

        # 基准信息
        bench_ann = (df['Benchmark'].iloc[-1] / initial_capital) ** (1/years) - 1
        excess = ann_ret - bench_ann
        active = strat_daily - bench_daily
        te = active.std() * np.sqrt(250)
        ir = excess / te if te > 0 else 0
        
        return {
            "Total_Ret": total_ret, "Ann_Ret": ann_ret, "Alpha": alpha, "Beta": beta,
            "Sharpe": sharpe, "Sortino": sortino, "Max_DD": max_dd, "Vol": vol,
            "Win_Rate": win_rate, "PL_Ratio": pl_ratio, "Trade_Count": trade_count, "Fees": total_fees,
            "Bench_Ann": bench_ann, "Excess": excess, "IR": ir, "Final_Eq": df['Equity'].iloc[-1]
        }
    except Exception:
        return None

# ==========================================
# 6. 主界面 (Main UI)
# ==========================================
def main_interface():
    # --- 侧边栏 ---
    with st.sidebar:
        st.header("⚙️ 实验控制台")
        
        # 资产
        symbol = st.text_input("股票代码", "600519", help="输入代码如 600519")
        
        # 资金
        c1, c2 = st.columns(2)
        initial_cap = c1.number_input("初始资金", 10000, 10000000, 100000, step=10000)
        comm_rate = c2.number_input("佣金费率", 0.0001, 0.0050, 0.0003, format="%.4f", step=0.0001)
        
        st.divider()
        
        # 策略
        st.subheader("策略模型")
        strat_name = st.selectbox("选择核心算法", list(STRATEGY_MAP.keys()))
        strat_code = STRATEGY_MAP[strat_name]
        
        # 动态参数
        p = {}
        if strat_code == "MA_Filter":
            p['ma_long'] = st.slider("均线周期", 10, 250, 20)
        elif strat_code == "MA_Cross":
            p['ma_short'] = st.slider("快线周期", 3, 60, 5)
            p['ma_long'] = st.slider("慢线周期", 10, 120, 20)
        elif strat_code == "RSI_Reversion":
            p['rsi_period'] = st.slider("RSI周期", 6, 24, 14)
            p['lower_bound'] = st.slider("买入阈值", 10, 40, 30)
            p['upper_bound'] = st.slider("卖出阈值", 60, 90, 70)
        elif strat_code == "Donchian_Breakout":
            p['channel_period'] = st.slider("通道周期", 10, 60, 20)
        elif strat_code == "Grid_Trading":
            p['grid_size'] = st.slider("网格密度", 0.01, 0.15, 0.05, step=0.01)
        elif strat_code == "Bollinger_Squeeze":
            p['std_dev'] = st.slider("带宽倍数", 1.0, 3.0, 2.0, step=0.1)
        elif strat_code == "BIAS_Reversion":
            p['bias_th'] = st.slider("乖离阈值", 3.0, 15.0, 6.0)
            
        with st.expander("策略逻辑说明"):
            st.markdown(get_strategy_doc(strat_code, p))
            
        st.divider()
        
        # 时间
        st.subheader("回测区间")
        start_date = st.date_input("开始", datetime.now() - timedelta(days=365*2))
        end_date = st.date_input("结束", datetime.now())

    # --- 主内容区 ---
    st.title(f"📈 资产净值与策略审计: {symbol}")
    
    # 获取数据
    with st.spinner("正在从交易所获取清洗数据..."):
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")
        df_raw, name = fetch_data_bs(symbol, s_str, e_str)
        
    if df_raw is not None:
        # 1. 策略运算
        df_res = run_strategy_logic(df_raw, strat_code, p)
        # 2. 回测运算
        df_res = run_backtest_core(df_res, initial_cap, comm_rate)
        # 3. 指标运算
        m = calc_4x4_metrics(df_res, initial_cap)
        
        if m:
            st.markdown("### 📊 专业指标审计矩阵")
            # 4x4 矩阵渲染
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("策略总收益", f"{m['Total_Ret']*100:.2f}%")
            c2.metric("年化收益率", f"{m['Ann_Ret']*100:.2f}%", help="CAGR")
            c3.metric("Alpha (α)", f"{m['Alpha']:.3f}")
            c4.metric("Beta (β)", f"{m['Beta']:.3f}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("夏普比率", f"{m['Sharpe']:.3f}")
            c2.metric("索提诺比率", f"{m['Sortino']:.3f}")
            c3.metric("最大回撤", f"{m['Max_DD']*100:.2f}%", delta_color="inverse")
            c4.metric("波动率", f"{m['Vol']*100:.2f}%")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("交易胜率", f"{m['Win_Rate']*100:.1f}%")
            c2.metric("盈亏比", f"{m['PL_Ratio']:.2f}")
            c3.metric("交易次数", f"{m['Trade_Count']}")
            c4.metric("手续费", f"¥{m['Fees']:.1f}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("基准年化", f"{m['Bench_Ann']*100:.2f}%")
            c2.metric("超额收益", f"{m['Excess']*100:.2f}%")
            c3.metric("信息比率", f"{m['IR']:.3f}")
            c4.metric("期末总资产", f"¥{m['Final_Eq']:,.0f}")
            
            # --- Plotly 可视化 (黑金主题) ---
            st.markdown("### 📉 净值走势与交易点位")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            
            # 1. 主图：净值
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Benchmark'], name="基准", line=dict(color='#555', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_res.index, y=df_res['Equity'], name="策略", line=dict(color='#D4AF37', width=2)), row=1, col=1)
            
            # 买卖点
            trades = df_res['Signal'].diff()
            buys = df_res[trades == 1]
            sells = df_res[trades == -1]
            
            fig.add_trace(go.Scatter(x=buys.index, y=buys['Equity'], mode='markers', name='买入', 
                                   marker=dict(symbol='triangle-up', size=10, color='#FF3333', line=dict(width=1, color='white'))), row=1, col=1)
            fig.add_trace(go.Scatter(x=sells.index, y=sells['Equity'], mode='markers', name='卖出', 
                                   marker=dict(symbol='triangle-down', size=10, color='#00CC66', line=dict(width=1, color='white'))), row=1, col=1)
            
            # 2. 副图：回撤
            dd = (df_res['Equity'] - df_res['Equity'].cummax()) / df_res['Equity'].cummax()
            fig.add_trace(go.Scatter(x=df_res.index, y=dd, name='回撤', fill='tozeroy', 
                                   line=dict(color='#cc3333', width=1), fillcolor='rgba(204, 51, 51, 0.3)'), row=2, col=1)
            
            # 样式
            fig.update_layout(
                paper_bgcolor='#000000',
                plot_bgcolor='#111111',
                xaxis=dict(showgrid=True, gridcolor='#333', tickfont=dict(color='#888')),
                yaxis=dict(showgrid=True, gridcolor='#333', tickfont=dict(color='#888')),
                yaxis2=dict(showgrid=True, gridcolor='#333', tickfont=dict(color='#888')),
                legend=dict(font=dict(color='#EEE'), bgcolor='rgba(0,0,0,0)'),
                height=650,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.warning(f"未获取到代码 {symbol} 的数据，请检查拼写或日期范围。")

# 执行
if st.session_state.logged_in:
    main_interface()