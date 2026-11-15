"""저크(Jerk) 시각화 앱 - 사다리꼴 / S-curve / 5차 다항식 궤적 비교
실행:  pip install -r requirements.txt  →  streamlit run app.py
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="저크(Jerk) 이해하기", page_icon="📈", layout="wide")

DT = 0.0005  # 시간 간격 (s)
COLORS = {"사다리꼴": "#E4572E", "S-curve": "#2E86AB", "5차 다항식": "#3BB273"}


# ───────────────────────── 궤적 계산 ─────────────────────────
def cumint(f, t):
    """사다리꼴 공식 누적 적분"""
    return np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(t))])


def quintic(T, D, t):
    """5차 다항식: s = 10τ³ − 15τ⁴ + 6τ⁵ (τ = t/T)"""
    tau = np.clip(t / T, 0, 1)
    x = D * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    v = D / T * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
    a = D / T**2 * (60 * tau - 180 * tau**2 + 120 * tau**3)
    j = D / T**3 * (60 - 360 * tau + 360 * tau**2)
    after = t > T
    v[after] = a[after] = j[after] = 0.0
    return dict(x=x, v=v, a=a, j=j, impulses=None)


def scurve(T, D, ta_ratio, tj_ratio, t):
    """대칭 S-curve. tj_ratio = 0 이면 사다리꼴.
    ta: 가속 시간, tj: 가속도가 0 → 최대로 올라가는 시간(저크 구간)"""
    ta = ta_ratio * T
    tj = tj_ratio * ta
    vp = D / (T - ta)          # 최고 속도 (가속·감속 구간이 대칭이므로 사다리꼴과 동일)
    A = vp / (ta - tj)         # 최대 가속도
    if tj < 1e-12:             # 사다리꼴: 가속도가 계단 → 저크는 임펄스(∞)
        a = np.where(t < ta, A, np.where(t < T - ta, 0.0, np.where(t <= T, -A, 0.0)))
        j = None
        impulses = [0.0, ta, T - ta, T]
        v = np.interp(t, [0, ta, T - ta, T], [0, vp, vp, 0])  # 속도는 정확한 조각 선형
        return dict(x=cumint(v, t), v=v, a=a, j=j, impulses=impulses)
    else:
        tk = [0, tj, ta - tj, ta, T - ta, T - ta + tj, T - tj, T]
        ak = [0, A, A, 0, 0, -A, -A, 0]
        a = np.interp(t, tk, ak)
        j = np.gradient(a, t)
        impulses = None
    v = cumint(a, t)
    x = cumint(v, t)
    return dict(x=x, v=v, a=a, j=j, impulses=impulses)


@st.cache_data
def simulate_tip(u, fn, zeta):
    """끝단(질량-스프링-댐퍼)이 기준 위치 u(t)를 따라가는 모델: y'' = ω²(u − y) − 2ζω y'"""
    w = 2 * np.pi * fn
    y = yd = 0.0
    out = np.empty_like(u)
    for i in range(len(u)):
        yd += (w * w * (u[i] - y) - 2 * zeta * w * yd) * DT
        y += yd * DT
        out[i] = y
    return out


# ───────────────────────── 그리기 ─────────────────────────
def stack_plot(profiles, t, T):
    titles = ("위치 x  [m]", "속도 v = dx/dt  [m/s]",
              "가속도 a = dv/dt  [m/s²]", "저크 j = da/dt  [m/s³]")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, subplot_titles=titles)
    for name, p in profiles.items():
        c = COLORS[name]
        for row, key in enumerate(["x", "v", "a"], start=1):
            fig.add_trace(go.Scatter(x=t, y=p[key], name=name, legendgroup=name,
                                     line=dict(color=c, width=2.5),
                                     showlegend=(row == 1)), row=row, col=1)
        if p["j"] is not None:
            fig.add_trace(go.Scatter(x=t, y=p["j"], name=name, legendgroup=name,
                                     line=dict(color=c, width=2.5),
                                     showlegend=False), row=4, col=1)
        else:  # 임펄스(이론상 무한대)
            for ti in p["impulses"]:
                fig.add_vline(x=ti, row=4, col=1, line_dash="dot",
                              line_color=c, line_width=2)
            fig.add_annotation(x=T / 2, y=0, row=4, col=1, showarrow=False,
                               text=f"{name}: 점선 위치에서 저크 = ∞ (임펄스)",
                               font=dict(color=c))
    fig.update_xaxes(title_text="시간 t [s]", row=4, col=1)
    fig.update_layout(height=820, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.07))
    return fig


# ───────────────────────── 사이드바 ─────────────────────────
st.sidebar.header("⚙️ 이동 조건")
D = st.sidebar.slider("이동 거리 Δ [m]", 0.1, 2.0, 1.0, 0.1)
T = st.sidebar.slider("이동 시간 T [s]", 0.5, 4.0, 2.0, 0.1)
st.sidebar.header("사다리꼴 / S-curve 설정")
ta_ratio = st.sidebar.slider("가속 시간 비율 (ta / T)", 0.10, 0.50, 0.25, 0.05,
                             help="0.5이면 등속 구간이 없는 삼각형 속도 프로파일")
tj_ratio = st.sidebar.slider("S-curve 저크 구간 비율 (tj / ta)", 0.05, 0.50, 0.50, 0.05,
                             help="클수록 가속도 변화가 더 부드러워집니다. 0.5이면 가속도가 삼각형 모양")

t = np.arange(0, T + 1.5 + DT / 2, DT)  # 도착 후 1.5초 더 (진동 관찰용)
trap = scurve(T, D, ta_ratio, 0.0, t)
sc = scurve(T, D, ta_ratio, tj_ratio, t)
q5 = quintic(T, D, t)
ALL = {"사다리꼴": trap, "S-curve": sc, "5차 다항식": q5}

st.title("📈 저크(Jerk)란? — 로봇 팔의 부드러운 움직임")
st.caption("위치 → 속도 → 가속도 → **저크**로 미분하며 사다리꼴, S-curve, 5차 다항식 궤적을 비교해 봅니다.")

tab1, tab2, tab3 = st.tabs(["① 저크 개념", "② 궤적 비교", "③ 진동 실험"])

# ───────────────────────── 탭 1 ─────────────────────────
with tab1:
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.subheader("미분 사슬")
        st.latex(r"x \xrightarrow{\ d/dt\ } v \xrightarrow{\ d/dt\ } a \xrightarrow{\ d/dt\ } j")
        st.markdown(
            """
- **속도** : 위치가 변하는 빠르기
- **가속도** : 속도가 변하는 빠르기
- **저크** : **가속도가 변하는 빠르기** (단위 m/s³)

버스가 출발할 때 몸이 **덜컥** 하는 느낌이 저크예요.
F = ma 이므로, 저크는 곧 **힘이 얼마나 갑자기 변하는가**를 뜻하고,
저크가 크면 기계에 충격과 **진동**이 생겨요.
"""
        )
        choice = st.radio("궤적 선택", list(ALL.keys()), index=0)
        st.info("사다리꼴은 가속도가 계단처럼 점프하므로, 점프하는 순간 저크가 이론상 무한대입니다."
                if choice == "사다리꼴" else
                "가속도가 서서히 변해서 저크가 유한한 값으로 유지됩니다.")
    with c2:
        st.plotly_chart(stack_plot({choice: ALL[choice]}, t, T), use_container_width=True)

# ───────────────────────── 탭 2 ─────────────────────────
with tab2:
    st.subheader("같은 거리, 같은 시간, 다른 모양")
    st.plotly_chart(stack_plot(ALL, t, T), use_container_width=True)

    def fmt_j(p):
        return "∞ (이론상)" if p["j"] is None else f"{np.max(np.abs(p['j'])):.2f}"

    rows = {
        "궤적": list(ALL.keys()),
        "최고 속도 [m/s]": [f"{np.max(p['v']):.3f}" for p in ALL.values()],
        "최대 가속도 [m/s²]": [f"{np.max(np.abs(p['a'])):.3f}" for p in ALL.values()],
        "최대 저크 [m/s³]": [fmt_j(p) for p in ALL.values()],
    }
    st.table(rows)
    st.caption("S-curve는 사다리꼴과 최고 속도가 같고, 가속도 변화만 부드럽게 만든 형태입니다. "
               "저크를 줄이는 대신 같은 이동 시간 안에서 최대 가속도가 조금 커집니다.")

    with st.expander("📐 5차 다항식 수식 확인 (보고서 Ⅲ-(2))"):
        st.latex(r"s(\tau)=10\tau^3-15\tau^4+6\tau^5,\quad \tau=t/T")
        st.latex(r"v=\frac{\Delta}{T}(30\tau^2-60\tau^3+30\tau^4),\;\;"
                 r"a=\frac{\Delta}{T^2}(60\tau-180\tau^2+120\tau^3),\;\;"
                 r"j=\frac{\Delta}{T^3}(60-360\tau+360\tau^2)")
        st.markdown(
            f"- 최고 속도 이론값 **1.875·Δ/T = {1.875*D/T:.3f}** → 그래프 값 **{np.max(q5['v']):.3f}**\n"
            f"- 최대 가속도 이론값 **(10/√3)·Δ/T² = {10/np.sqrt(3)*D/T**2:.3f}** → 그래프 값 **{np.max(np.abs(q5['a'])):.3f}**\n"
            f"- 출발 순간 가속도 a(0) = **{q5['a'][0]:.3f}** (0이라 부드럽게 출발), "
            f"출발 순간 저크 j(0) = 60Δ/T³ = **{q5['j'][0]:.2f}** (무한대는 아니지만 0도 아님)"
        )

# ───────────────────────── 탭 3 ─────────────────────────
with tab3:
    st.subheader("로봇 끝단은 얼마나 흔들릴까?")
    st.markdown("로봇 팔 끝을 **질량-스프링-댐퍼**로 단순화하고, 각 궤적을 기준 위치로 주었을 때 "
                "끝단이 기준 위치에서 얼마나 벗어나는지(진동)를 계산합니다.")
    c1, c2 = st.columns(2)
    fn = c1.slider("로봇의 고유진동수 fn [Hz]", 0.5, 10.0, 3.0, 0.1)
    zeta = c2.slider("감쇠비 ζ", 0.005, 0.2, 0.02, 0.005)

    Tn = 1 / fn
    ta = ta_ratio * T
    st.write(f"고유진동 주기 Tn = 1/fn = **{Tn:.3f} s**,  가속 시간 ta = **{ta:.3f} s**  "
             f"→  ta / Tn = **{ta / Tn:.2f}**")
    st.caption("참고: 사다리꼴에서는 가속 시간이 진동 주기의 정수배(ta/Tn = 1, 2, 3 …)일 때 "
               "가속 시작·종료 시점의 저크가 만든 진동이 서로 상쇄되어 잔류 진동이 줄어듭니다 "
               "(강한솔 외, 2017, 로봇학회논문지).")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("끝단 위치 y (점선: 기준 위치 u)", "추종 오차 y − u  [m]"))
    resid = {}
    for name, p in ALL.items():
        y = simulate_tip(p["x"], fn, zeta)
        err = y - p["x"]
        resid[name] = np.max(np.abs(err[t > T]))
        c = COLORS[name]
        fig.add_trace(go.Scatter(x=t, y=y, name=name, line=dict(color=c, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=p["x"], name=name + " 기준", showlegend=False,
                                 line=dict(color=c, width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=err, name=name, showlegend=False,
                                 line=dict(color=c, width=2)), row=2, col=1)
    fig.add_vline(x=T, line_dash="dash", line_color="gray")
    fig.add_annotation(x=T, y=1, yref="paper", text="도착", showarrow=False, yshift=10)
    fig.update_xaxes(title_text="시간 t [s]", row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**도착 후 잔류 진동 크기 (최대 오차)**")
    cols = st.columns(3)
    for col, (name, r) in zip(cols, resid.items()):
        col.metric(name, f"{r * 1000:.3f} mm")
    best = min(resid, key=resid.get)
    st.success(f"이 조건에서는 **{best}** 궤적의 잔류 진동이 가장 작아요. "
               "고유진동수와 가속 시간을 바꿔 보면 순위가 달라지는 것도 확인해 보세요!")
