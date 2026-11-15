"""물 컵을 옮길 때 어떤 움직임이 더 안정적일까?  사다리꼴 vs S-curve vs 5차 다항식
실행:  pip install -r requirements.txt  →  streamlit run app.py
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ═════════════════════ 계산 (물리 모델) ═════════════════════
G = 9.81
DT = 0.002
TG = np.arange(0.0, 5.0, DT)          # 시뮬레이션 시간 (최대 5초)
ZETA = 0.01                            # 물의 출렁임 감쇠비 (작음)
CUP_H = 0.10                           # 컵 높이 [m]
KINDS = {"trap": "사다리꼴", "scurve": "S-curve", "quintic": "5차 다항식"}
COLORS = {"trap": "#E4572E", "scurve": "#2E86AB", "quintic": "#3BB273"}


def accel(kind, T, ta_r, tj_r, D, t):
    """컵(로봇)의 가속도 a(t).  T는 (N,1), t는 (1,S) 배열."""
    if kind == "quintic":
        tau = np.clip(t / T, 0, 1)
        return np.where(t < T, D / T**2 * (60 * tau - 180 * tau**2 + 120 * tau**3), 0.0)
    ta = ta_r * T
    tj = np.maximum(tj_r * ta, 1e-9) if kind == "scurve" else np.full_like(ta, 1e-9)
    A = D / (T - ta) / (ta - tj)

    def g(s):
        return A * (np.clip(s / tj, 0, 1) - np.clip((s - (ta - tj)) / tj, 0, 1))

    return g(t) - g(t - (T - ta))


def peaks(kind, T, ta_r, tj_r, D):
    """최대 가속도, 최대 저크 (해석식)"""
    if kind == "quintic":
        return 10 / np.sqrt(3) * D / T**2, 60 * D / T**3
    ta = ta_r * T
    tj = max(tj_r * ta, 1e-9) if kind == "scurve" else 1e-9
    A = D / (T - ta) / (ta - tj)
    return A, (A / tj if kind == "scurve" else np.inf)


def slosh_freq(R, h):
    """원통 컵 물 출렁임 1차 모드 고유진동수 [Hz]"""
    xi = 1.841
    return np.sqrt(G * xi / R * np.tanh(xi * h / R)) / (2 * np.pi)


def simulate(kind, T, ta_r, tj_r, D, R, h):
    """T: (N,) 배열. 반환: 벽면 수면 높이 변화 eta (N,S) [m], 기울기 s (N,S) = ω²q/g"""
    T = np.atleast_1d(np.asarray(T, float))
    a = accel(kind, T[:, None], ta_r, tj_r, D, TG[None, :])
    aT = np.ascontiguousarray(a.T)
    w = 2 * np.pi * slosh_freq(R, h)
    q = np.zeros(len(T)); qd = np.zeros(len(T))
    slope = np.empty_like(aT)
    for i in range(len(TG)):
        qd += (-aT[i] - w * w * q - 2 * ZETA * w * qd) * DT   # 등가 진자(스프링-질량) 모델
        q += qd * DT
        slope[i] = w * w * q / G                              # 수면 기울기
    slope = slope.T
    return a, slope, R * slope


@st.cache_data(show_spinner=False)
def sweep(D, ta_r, tj_r, R, h):
    Ts = np.round(np.arange(0.6, 3.001, 0.05), 2)
    out = {}
    for k in KINDS:
        _, _, eta = simulate(k, Ts, ta_r, tj_r, D, R, h)
        out[k] = np.abs(eta).max(axis=1)
    return Ts, out


# ═════════════════════ 화면 ═════════════════════
st.set_page_config(page_title="물 컵 운반 시뮬레이터", page_icon="🥤", layout="wide")
st.title("🥤 물 컵을 옮길 때, 어떤 움직임이 더 안정적일까?")
st.caption("같은 거리를 같은 시간에 옮겨도, **가속도가 변하는 방식(저크)** 에 따라 물이 출렁이는 정도가 달라져요.")

c1, c2, c3 = st.columns(3)
D = c1.slider("이동 거리 [m]", 0.3, 2.0, 1.0, 0.1)
T = c2.slider("이동 시간 [초]", 0.6, 3.0, 1.5, 0.1)
fill = c3.slider("컵에 담긴 물의 양 [%]", 40, 95, 80, 5)
with st.expander("⚙️ 더 설정하기 (선택)"):
    d1, d2, d3 = st.columns(3)
    Rcm = d1.slider("컵 반지름 [cm]", 2.0, 6.0, 4.0, 0.5)
    ta_r = d2.slider("가속하는 시간 비율 (ta / T)", 0.10, 0.50, 0.30, 0.05,
                     help="사다리꼴과 S-curve에 적용돼요. 0.5이면 등속 구간 없이 가속→감속만 해요.")
    tj_r = d3.slider("S-curve 부드러움 (tj / ta)", 0.1, 0.5, 0.3, 0.1,
                     help="클수록 가속도가 더 천천히 변해요.")

R = Rcm / 100
h = CUP_H * fill / 100
free = CUP_H - h                      # 물이 넘치지 않는 여유 높이
fn = slosh_freq(R, h)
st.write(f"컵 높이 10 cm, 물 높이 **{h * 100:.1f} cm** → 물이 넘치기까지 여유 **{free * 1000:.0f} mm**,  "
         f"물의 출렁임 고유진동수 **{fn:.2f} Hz** (한 번 출렁이는 데 {1 / fn:.2f}초)")

ts = np.array([T])
sim = {k: simulate(k, ts, ta_r, tj_r, D, R, h) for k in KINDS}
stat = {}
for k in KINDS:
    a, slope, eta = sim[k]
    amax, jmax = peaks(k, T, ta_r, tj_r, D)
    after = TG > T
    stat[k] = dict(amax=amax, jmax=jmax, emax=np.abs(eta[0]).max(),
                   eres=np.abs(eta[0][after]).max(), spill=np.abs(eta[0]).max() > free)

# ── 결과 카드
st.subheader("결과")
cols = st.columns(3)
best = min(KINDS, key=lambda k: stat[k]["emax"])
for col, k in zip(cols, KINDS):
    s = stat[k]
    with col:
        st.markdown(f"<h4 style='color:{COLORS[k]};margin-bottom:0'>{KINDS[k]}</h4>", unsafe_allow_html=True)
        st.metric("물이 가장 많이 출렁인 높이", f"{s['emax'] * 1000:.1f} mm",
                  delta=f"여유 {free * 1000:.0f} mm 중", delta_color="off")
        st.write(("💦 **넘침!**" if s["spill"] else "✅ 안 넘침") + f"  |  도착 후 출렁임 {s['eres'] * 1000:.1f} mm")
        jtxt = "∞ (이론상)" if np.isinf(s["jmax"]) else f"{s['jmax']:.1f} m/s³"
        st.caption(f"최대 가속도 {s['amax']:.2f} m/s²  ·  최대 저크 {jtxt}")
st.success(f"이 조건에서는 **{KINDS[best]}** 가 물을 가장 적게 출렁이게 했어요.")

# ── 컵 애니메이션
st.subheader("🎬 컵 속 물 모습")
st.caption("컵 안에서 본 물 표면이에요 (출렁임이 잘 보이도록 그린 모습). 물이 컵 위로 넘치면 빨간색으로 바뀌어요.")
tf = np.arange(0, T + 1.5, 0.04)
idx = np.round(tf / DT).astype(int)
Wcm, Hcm = R * 100, CUP_H * 100
centers = {k: i * (2 * Wcm + 4) for i, k in enumerate(KINDS)}


def water_traces(fi):
    out = []
    for k in KINDS:
        sl = sim[k][1][0][idx[fi]]
        yl, yr = (h - R * sl) * 100, (h + R * sl) * 100        # 왼쪽 벽, 오른쪽 벽의 수면 높이 [cm]
        x0 = centers[k]
        over = max(yl, yr) > Hcm
        out.append(go.Scatter(x=[x0 - Wcm, x0 - Wcm, x0 + Wcm, x0 + Wcm, x0 - Wcm],
                              y=[0, yl, yr, 0, 0], mode="lines", fill="toself", showlegend=False,
                              line=dict(color="#d62728" if over else "#1f77b4", width=2),
                              fillcolor="rgba(214,39,40,0.5)" if over else "rgba(31,119,180,0.5)"))
    return out


fig = go.Figure(data=water_traces(0), frames=[
    go.Frame(data=water_traces(i), traces=[0, 1, 2], name=str(i)) for i in range(len(tf))])
for k in KINDS:
    x0 = centers[k]
    fig.add_trace(go.Scatter(x=[x0 - Wcm, x0 - Wcm, x0 + Wcm, x0 + Wcm], y=[Hcm, 0, 0, Hcm], mode="lines",
                             line=dict(color="black", width=4), showlegend=False))
    fig.add_annotation(x=x0, y=Hcm + 2.2, text=f"<b>{KINDS[k]}</b>", showarrow=False,
                       font=dict(color=COLORS[k], size=16))
fig.update_layout(
    height=380, margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False, range=[-1, Hcm + 5]),
    updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=-0.05, direction="left", buttons=[
        dict(label="▶ 재생", method="animate",
             args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
        dict(label="⏸ 정지", method="animate",
             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
    sliders=[dict(x=0.15, len=0.85, y=-0.02, currentvalue=dict(prefix="시간 [초]: "),
                  steps=[dict(method="animate", label=f"{tf[i]:.1f}",
                              args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                         for i in range(len(tf))])])
st.plotly_chart(fig)

# ── 시간 그래프
st.subheader("📈 시간에 따른 변화")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=("컵의 가속도 [m/s²]", "벽면에서 물 높이의 변화 [mm]  (점선 = 넘치는 높이)"))
tmask = TG <= T + 1.5
for k in KINDS:
    a, _, eta = sim[k]
    fig.add_trace(go.Scatter(x=TG[tmask], y=a[0][tmask], name=KINDS[k],
                             line=dict(color=COLORS[k], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=TG[tmask], y=eta[0][tmask] * 1000, name=KINDS[k], showlegend=False,
                             line=dict(color=COLORS[k], width=2)), row=2, col=1)
for s in (free * 1000, -free * 1000):
    fig.add_hline(y=s, row=2, col=1, line_dash="dot", line_color="red")
fig.add_vline(x=T, line_dash="dash", line_color="gray", annotation_text="도착")
fig.update_xaxes(title_text="시간 [초]", row=2, col=1)
fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig)
st.caption("사다리꼴은 가속도가 계단처럼 바뀌어서(저크 ∞) 출렁임이 크게 일어나기 쉬워요. "
           "S-curve와 5차 다항식은 가속도가 부드럽게 바뀌어 출렁임이 덜하지만, 같은 가속 시간이면 S-curve는 "
           "최대 가속도가 더 커질 수 있어요. 또 가속 시간이 물의 출렁임 주기의 정수배에 맞으면 사다리꼴도 "
           "도착 후 출렁임이 거의 사라져요. 설정을 바꿔 가며 확인해 보세요!")

# ── 얼마나 빨리 옮길 수 있나
st.subheader("⏱️ 얼마나 빨리 옮길 수 있을까?")
Ts, mx = sweep(D, ta_r, tj_r, R, h)
fig = go.Figure()
for k in KINDS:
    fig.add_trace(go.Scatter(x=Ts, y=mx[k] * 1000, name=KINDS[k], line=dict(color=COLORS[k], width=2.5)))
fig.add_hline(y=free * 1000, line_dash="dot", line_color="red", annotation_text="넘치는 높이")
fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="이동 시간 [초]",
                  yaxis_title="최대 출렁임 높이 [mm]", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig)
cols = st.columns(3)
for col, k in zip(cols, KINDS):
    unsafe = np.where(mx[k] > free)[0]
    if len(unsafe) == 0:
        txt = f"{Ts[0]:.2f} 초 이상"
    elif unsafe[-1] == len(Ts) - 1:
        txt = "3초로도 넘침"
    else:
        txt = f"{Ts[unsafe[-1] + 1]:.2f} 초 이상"
    col.metric(f"{KINDS[k]}: 안 넘치게 옮기려면", txt)
st.caption("곡선이 낮을수록 같은 시간에 물이 덜 출렁이고, 같은 곡선이라도 점선 아래에 있어야 물이 안 넘쳐요. "
           "이 시뮬레이션은 단순화한 모델이라 실제 컵의 값과는 차이가 있을 수 있어요.")

with st.expander("📖 어떻게 계산했나요?"):
    st.markdown("물의 출렁임을 **스프링에 매달린 질량**(등가 진자)처럼 단순화했어요. 컵이 가속도 a(t)로 움직일 때,")
    st.latex(r"\ddot q = -a(t) - \omega^2 q - 2\zeta\omega\dot q,\qquad \text{벽면 수면 높이 변화} = R\,\frac{\omega^2 q}{g}")
    st.markdown("- 출렁임 고유진동수 ω는 컵 반지름과 물 높이로 계산한 원통 용기의 1차 슬로싱 모드 값이에요.\n"
                "- 벽면 수면 높이 변화가 컵의 여유 높이를 넘으면 물이 넘친다고 봤어요.\n"
                "- 5차 다항식: s = 10τ³ − 15τ⁴ + 6τ⁵, 사다리꼴/S-curve: 가속 시간 ta를 같은 방식으로 맞춰 비교했어요.")
