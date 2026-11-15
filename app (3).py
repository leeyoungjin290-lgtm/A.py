"""AI(강화학습)가 찾아낸 '물건을 가장 안정적으로 옮기는 움직임' 비교 시뮬레이터
사다리꼴  vs  S-curve  vs  5차 다항식

실행:  pip install -r requirements.txt  →  streamlit run app.py
(sim_core.py 와 같은 폴더에 두세요)
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sim_core import (DT, G, KINDS, SETTLE, TGRID, TMAX, accel, cumint, evaluate,
                      grid_best, peak_accel_jerk, train)

st.set_page_config(page_title="AI 물건 운반 시뮬레이터", page_icon="🤖", layout="wide")

COLORS = {"trap": "#E4572E", "scurve": "#2E86AB", "quintic": "#3BB273"}
ORDER = list(KINDS)

PRESETS = {
    "물 담긴 컵": dict(fn=2.5, zeta=0.02, mu=0.40, zlim=8.0),
    "유리잔 (미끄러운 쟁반)": dict(fn=4.0, zeta=0.03, mu=0.12, zlim=4.0),
    "단단한 부품 (볼트)": dict(fn=6.0, zeta=0.05, mu=0.25, zlim=3.0),
}

# ───────────────────────── 사이드바 ─────────────────────────
st.sidebar.header("📦 운반 조건")
preset = st.sidebar.selectbox("운반할 물체", list(PRESETS))
P = PRESETS[preset]
D = st.sidebar.slider("이동 거리 [m]", 0.5, 2.0, 1.0, 0.1)
fn = st.sidebar.slider("물체 고정부 고유진동수 [Hz]", 1.0, 8.0, P["fn"], 0.1, key=f"fn_{preset}",
                       help="물체가 그리퍼/쟁반에 얼마나 '말랑하게' 붙어 있는지. 낮을수록 잘 흔들려요.")
zeta = st.sidebar.slider("감쇠비 ζ", 0.005, 0.2, P["zeta"], 0.005, key=f"ze_{preset}")
mu = st.sidebar.slider("마찰계수 μ", 0.05, 0.8, P["mu"], 0.01, key=f"mu_{preset}",
                       help="가속도가 μg 를 넘으면 물체가 미끄러져요.")
zlim = st.sidebar.slider("허용 흔들림 [mm]", 1.0, 15.0, P["zlim"], 0.5, key=f"zl_{preset}",
                         help="이보다 많이 흔들리면 불합격(물이 넘침, 유리 파손 등).") / 1000
lam = st.sidebar.slider("속도 중요도 λ", 0.0, 3.0, 1.0, 0.1,
                        help="클수록 빨리 옮기는 것을 중요하게 생각해요. 0이면 안정성만 봅니다.")
st.sidebar.header("🤖 학습 설정")
iters = st.sidebar.slider("학습 반복 횟수", 30, 120, 70, 10)
batch = st.sidebar.slider("한 번에 시도하는 움직임 수", 12, 48, 24, 4)
seed = st.sidebar.number_input("랜덤 시드", 0, 9999, 1)

scen = dict(fn=fn, zeta=zeta, mu=mu)
sig = (preset, D, fn, zeta, mu, zlim, lam, iters, batch, int(seed))

st.title("🤖 AI가 고른 '가장 안정적인 운반 움직임'")
st.caption("강화학습(정책경사)으로 사다리꼴 · S-curve · 5차 다항식 각각의 **최적 파라미터**를 학습시키고, "
           "물건을 옮길 때 어느 쪽이 더 안정적인지 비교합니다.")


# ───────────────────────── 헬퍼 ─────────────────────────
def nominal_metrics(kind, r):
    m = evaluate(kind, [r["T"]], [r["ta"]], [r["tj"]], D, fn, zeta, mu, lam, zlim, keep=True)
    amax, jmax = peak_accel_jerk(kind, np.array([r["T"]]), np.array([r["ta"]]), np.array([r["tj"]]), D)
    return dict(cost=float(m["cost"][0]), peak=float(m["peak"][0]), slip=float(m["slip"][0]),
                rrms=float(m["rrms"][0]), rmax=float(m["rmax"][0]), ok=bool(m["success"][0]),
                amax=float(amax[0]), jmax=float(jmax[0]), a=m["a"][0], Z=m["Z"], P=m["P"], SL=m["SL"])


def run_training():
    res = {}
    bar = st.progress(0.0, text="학습 준비 중…")
    for n, kind in enumerate(ORDER):
        def cb(f, n=n, kind=kind):
            bar.progress((n + f) / (len(ORDER) + 1), text=f"🤖 {KINDS[kind]} 궤적 학습 중… ({int(f * 100)}%)")
        r = train(kind, D, scen, lam, zlim, iters=iters, batch=batch, seed=int(seed), progress=cb)
        res[kind] = r
    bar.progress(len(ORDER) / (len(ORDER) + 1), text="격자 탐색으로 결과 검증 중…")
    for kind in ORDER:
        res[kind]["grid"] = grid_best(kind, D, scen, lam, zlim)
        res[kind]["nom"] = nominal_metrics(kind, res[kind])
    bar.empty()
    return res


@st.cache_data(show_spinner=False)
def fn_sweep(kind, T, ta, tj, D, fn, zeta, mu, lam, zlim):
    rat = np.linspace(0.5, 1.5, 51)
    m = evaluate(kind, np.full(51, T), np.full(51, ta), np.full(51, tj), D, fn * rat, zeta, mu, lam, zlim)
    return rat, m["cost"], m["rmax"]


@st.cache_data(show_spinner=False)
def random_trials(kind, T, ta, tj, D, fn, zeta, mu, lam, zlim, n, var, seed):
    rng = np.random.default_rng(seed)      # 모든 궤적이 똑같은 시험 조건을 받도록 시드 고정
    f = fn * rng.uniform(1 - var, 1 + var, n)
    z = zeta * rng.uniform(1 - var, 1 + var, n)
    u = mu * rng.uniform(1 - var, 1 + var, n)
    m = evaluate(kind, np.full(n, T), np.full(n, ta), np.full(n, tj), D, f, z, u, lam, zlim)
    return m["cost"], m["success"]


def build_animation(res, exag):
    n = len(ORDER)
    tf_end = max(res[k]["T"] for k in ORDER) + SETTLE
    tf = np.arange(0, tf_end, 0.05)
    tfine = np.arange(0, TMAX + SETTLE, 0.001)
    data = {}
    for k in ORDER:
        r = res[k]
        af = accel(k, np.float64(r["T"]), np.float64(r["ta"]), np.float64(r["tj"]), D, tfine)
        x = cumint(cumint(af, tfine), tfine)
        x = x * D / x[-1]
        ic = np.minimum(np.round(tf / DT).astype(int), len(TGRID) - 1)
        iff = np.minimum(np.round(tf / 0.001).astype(int), len(tfine) - 1)
        nom = r["nom"]
        data[k] = dict(x=x[iff], off=nom["Z"][0][ic] + nom["P"][0][ic], sl=nom["SL"][0][ic])

    def lane_traces(fi):
        out = []
        for i, k in enumerate(ORDER):
            y = n - 1 - i
            xt = data[k]["x"][fi]
            out.append(go.Scatter(x=[xt - 0.09, xt + 0.09], y=[y, y], mode="lines",
                                  line=dict(color="#555", width=8), showlegend=False))
            out.append(go.Scatter(x=[xt + exag * data[k]["off"][fi]], y=[y + 0.14], mode="markers",
                                  marker=dict(size=20, line=dict(color="black", width=1),
                                              color="red" if data[k]["sl"][fi] else COLORS[k]),
                                  showlegend=False))
        return out

    fig = go.Figure(data=lane_traces(0), frames=[
        go.Frame(data=lane_traces(i), traces=list(range(2 * n)), name=str(i)) for i in range(len(tf))])
    for i, k in enumerate(ORDER):
        fig.add_annotation(x=-0.2, y=n - 1 - i + 0.38, xanchor="left", showarrow=False,
                           text=f"<b>{KINDS[k]}</b> (T={res[k]['T']:.2f}s)", font=dict(color=COLORS[k], size=14))
    fig.add_vline(x=D, line_dash="dot", line_color="gray", annotation_text="목표")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-0.2, D + 0.3], title="위치 [m]"),
        yaxis=dict(range=[-0.5, n - 0.3], visible=False),
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=-0.25, direction="left", buttons=[
            dict(label="▶ 재생", method="animate",
                 args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True,
                                  transition=dict(duration=0))]),
            dict(label="⏸ 정지", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.15, len=0.85, y=-0.1, currentvalue=dict(prefix="시간 [s]: "),
                      steps=[dict(method="animate", label=f"{tf[i]:.1f}",
                                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for i in range(len(tf))])])
    return fig


res = st.session_state.get("res")
stale = res is not None and st.session_state.get("sig") != sig

tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI 학습", "🎬 운반 시뮬레이션", "🛡️ 안정성 시험", "📖 모델 설명"])

# ───────────────────────── 탭 1 : 학습 ─────────────────────────
with tab1:
    st.write(f"선택한 물체: **{preset}**  |  이동 거리 {D} m  |  fn {fn} Hz, ζ {zeta}, μ {mu}  |  "
             f"허용 흔들림 {zlim * 1000:.1f} mm  |  λ {lam}")
    if st.button("🚀 학습 시작", type="primary"):
        res = run_training()
        st.session_state["res"] = res
        st.session_state["sig"] = sig
        stale = False
    if res is None:
        st.info("왼쪽에서 조건을 고른 뒤 **학습 시작**을 눌러 주세요. (약 10~20초)")
    else:
        if stale:
            st.warning("⚠️ 조건이 바뀌었어요. 아래 결과는 이전 조건에서 학습한 것이니 **학습 시작**을 다시 눌러 주세요.")

        fig = go.Figure()
        for k in ORDER:
            fig.add_trace(go.Scatter(y=res[k]["hist_eval"], name=KINDS[k], mode="lines",
                                     line=dict(color=COLORS[k], width=3)))
            fig.add_hline(y=res[k]["grid"], line_dash="dot", line_color=COLORS[k], opacity=0.6)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                          title="AI의 학습 곡선 (낮을수록 좋음, 점선 = 격자탐색으로 구한 최고 점수)",
                          xaxis_title="학습 반복", yaxis_title="비용 = −보상", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig)

        rows = {k2: [] for k2 in ["궤적", "이동시간 T [s]", "가속시간 ta/T", "저크구간 tj/ta", "최대 가속도 [m/s²]",
                                  "최대 저크 [m/s³]", "최대 흔들림 [mm]", "미끄러짐 [mm]", "도착 후 잔류진동 [mm]",
                                  "비용", "합격", "격자탐색 대비"]}
        for k in ORDER:
            r, n_ = res[k], res[k]["nom"]
            rows["궤적"].append(KINDS[k])
            rows["이동시간 T [s]"].append(f"{r['T']:.2f}")
            rows["가속시간 ta/T"].append("—" if k == "quintic" else f"{r['ta']:.2f}")
            rows["저크구간 tj/ta"].append(f"{r['tj']:.2f}" if k == "scurve" else "—")
            rows["최대 가속도 [m/s²]"].append(f"{n_['amax']:.2f}")
            rows["최대 저크 [m/s³]"].append("∞ (이론상)" if np.isinf(n_["jmax"]) else f"{n_['jmax']:.1f}")
            rows["최대 흔들림 [mm]"].append(f"{n_['peak'] * 1000:.2f}")
            rows["미끄러짐 [mm]"].append(f"{n_['slip'] * 1000:.2f}")
            rows["도착 후 잔류진동 [mm]"].append(f"{n_['rmax'] * 1000:.2f}")
            rows["비용"].append(f"{n_['cost']:.3f}")
            rows["합격"].append("✅" if n_["ok"] else "❌")
            rows["격자탐색 대비"].append(f"+{(n_['cost'] / r['grid'] - 1) * 100:.1f}%")
        st.table(rows)
        best = min(ORDER, key=lambda k: res[k]["nom"]["cost"])
        st.success(f"학습한 조건에서는 **{KINDS[best]}** 궤적의 비용이 가장 낮았어요. "
                   "하지만 이건 *이 조건에서의 점수*일 뿐이에요. 조건이 조금 달라져도 안정적인지는 "
                   "**🛡️ 안정성 시험** 탭에서 확인해 보세요!")
        st.caption("'격자탐색 대비'는 AI가 찾은 값이 무작정 모든 조합을 다 시험해 본 최고 점수보다 몇 % 나쁜지를 뜻해요. "
                   "숫자가 작을수록 AI가 최적값에 가깝게 학습한 거예요.")

# ───────────────────────── 탭 2 : 시뮬레이션 ─────────────────────────
with tab2:
    if res is None:
        st.info("먼저 **🤖 AI 학습** 탭에서 학습을 실행해 주세요.")
    else:
        if stale:
            st.warning("⚠️ 이전 조건의 학습 결과를 보여 주고 있어요.")
        exag = st.slider("물체 움직임 확대 배율", 1, 200, 40,
                         help="실제 흔들림은 mm 단위라 눈에 안 보여서 확대해서 그려요. 빨간색 = 미끄러지는 중")
        st.plotly_chart(build_animation(res, exag))
        st.caption("▶ 재생을 누르면 세 궤적이 동시에 출발해요. 회색 막대 = 쟁반(그리퍼), 동그라미 = 물체.")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("쟁반 가속도 a(t) [m/s²]", "쟁반에 대한 물체의 상대 변위 z(t) [mm]"))
        for k in ORDER:
            n_ = res[k]["nom"]
            fig.add_trace(go.Scatter(x=TGRID, y=n_["a"], name=KINDS[k], line=dict(color=COLORS[k], width=2.5)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=TGRID, y=n_["Z"][0] * 1000, name=KINDS[k], showlegend=False,
                                     line=dict(color=COLORS[k], width=2)), row=2, col=1)
        zmax_mm = mu * G / (2 * np.pi * fn) ** 2 * 1000
        for s in (zmax_mm, -zmax_mm):
            fig.add_hline(y=s, row=2, col=1, line_dash="dot", line_color="gray")
        fig.add_annotation(x=TMAX + SETTLE, y=zmax_mm, row=2, col=1, text="미끄러짐 한계", showarrow=False,
                           xanchor="right", yshift=10)
        fig.update_xaxes(title_text="시간 t [s]", row=2, col=1)
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)
        st.caption("가속도가 갑자기 바뀌는(저크가 큰) 궤적일수록, 도착한 뒤에도 z(t)가 오래 떨려요. "
                   "z(t)가 점선(미끄러짐 한계)에 닿으면 물체가 미끄러져요.")

# ───────────────────────── 탭 3 : 안정성 시험 ─────────────────────────
with tab3:
    if res is None:
        st.info("먼저 **🤖 AI 학습** 탭에서 학습을 실행해 주세요.")
    else:
        if stale:
            st.warning("⚠️ 이전 조건의 학습 결과를 보여 주고 있어요.")
        st.subheader("① 물체의 고유진동수가 달라지면?")
        st.write("AI는 *하나의 조건*에서 학습했어요. 실제로는 물체 무게나 잡는 힘이 바뀌어 고유진동수가 달라질 수 있죠. "
                 "학습한 궤적을 그대로 두고 고유진동수만 50%~150%로 바꿔 볼게요.")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("도착 후 잔류진동 [mm]", "종합 비용 (낮을수록 좋음)"))
        for k in ORDER:
            r = res[k]
            rat, cost, rmax = fn_sweep(k, r["T"], r["ta"], r["tj"], D, fn, zeta, mu, lam, zlim)
            fig.add_trace(go.Scatter(x=rat * 100, y=rmax * 1000, name=KINDS[k],
                                     line=dict(color=COLORS[k], width=2.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=rat * 100, y=cost, name=KINDS[k], showlegend=False,
                                     line=dict(color=COLORS[k], width=2.5)), row=1, col=2)
        fig.add_vline(x=100, line_dash="dot", line_color="gray")
        fig.update_xaxes(title_text="학습 때 대비 고유진동수 [%]")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig)
        st.caption("학습한 조건(100%)에서만 진동이 0에 가깝게 '맞춰진' 궤적은 조건이 조금만 어긋나도 진동이 커져요 "
                   "(공진 조건에 딱 맞춘 셈). 곡선이 완만할수록 조건 변화에 강한 궤적이에요.")

        st.subheader("② 무작위 조건 시험")
        c1, c2 = st.columns(2)
        n_tr = c1.slider("시험 횟수", 100, 500, 300, 50)
        var = c2.slider("조건 변동 폭 (±%)", 10, 50, 30, 5) / 100
        stats = {}
        for k in ORDER:
            r = res[k]
            cost, ok = random_trials(k, r["T"], r["ta"], r["tj"], D, fn, zeta, mu, lam, zlim, n_tr, var, 123)
            stats[k] = dict(cost=cost, ok=ok)
        c1, c2 = st.columns(2)
        fig = go.Figure(go.Bar(x=[KINDS[k] for k in ORDER], y=[stats[k]["ok"].mean() * 100 for k in ORDER],
                               marker_color=[COLORS[k] for k in ORDER],
                               text=[f"{stats[k]['ok'].mean() * 100:.0f}%" for k in ORDER], textposition="outside"))
        fig.update_layout(title="합격률 (미끄러짐·흔들림·잔류진동 기준)", yaxis=dict(range=[0, 110], title="%"),
                          height=340, margin=dict(l=10, r=10, t=50, b=10))
        c1.plotly_chart(fig)
        fig = go.Figure()
        for k in ORDER:
            fig.add_trace(go.Box(y=stats[k]["cost"], name=KINDS[k], marker_color=COLORS[k], boxmean=True))
        fig.update_layout(title="비용 분포 (낮고 좁을수록 안정적)", height=340, showlegend=False,
                          margin=dict(l=10, r=10, t=50, b=10))
        c2.plotly_chart(fig)

        rank = sorted(ORDER, key=lambda k: (-stats[k]["ok"].mean(), stats[k]["cost"].mean()))
        top = rank[0]
        st.success(
            f"이 시험에서는 **{KINDS[top]}** 궤적이 합격률 {stats[top]['ok'].mean() * 100:.0f}%, "
            f"평균 비용 {stats[top]['cost'].mean():.2f}로 가장 안정적이었어요.")
        st.caption("⚠️ 결과는 물체·조건·시드에 따라 달라져요. 물체를 바꾸거나 변동 폭을 키워서 순위가 바뀌는지 "
                   "직접 확인해 보세요. 이 시뮬레이션은 단순화한 모델이라 실제 로봇의 측정 결과는 아니에요.")

# ───────────────────────── 탭 4 : 모델 설명 ─────────────────────────
with tab4:
    st.subheader("물리 모델")
    st.markdown("쟁반(그리퍼)이 정해진 궤적으로 움직이고, 그 위의 물체는 **탄성 패드**로 붙어 있다고 단순화했어요. "
                "z는 쟁반에 대한 물체의 상대 변위예요.")
    st.latex(r"\ddot z = -a_t(t) - \omega^2 z - 2\zeta\omega\,\dot z,\qquad \omega = 2\pi f_n")
    st.latex(r"|z| > z_{max} = \frac{\mu g}{\omega^2}\ \Rightarrow\ \text{물체가 미끄러짐 (미끄러진 거리 누적)}")
    st.markdown("가속도 a_t가 크면 z가 커지고, 저크가 크면(가속도가 갑자기 바뀌면) 도착 후에도 z가 떨려요.")

    st.subheader("강화학습 (REINFORCE, 정책경사)")
    st.markdown(
        """
- **행동(action)** : 궤적 파라미터 — 이동시간 T, 가속시간 비율 ta/T, 저크구간 비율 tj/ta  
  (사다리꼴 = T, ta / S-curve = T, ta, tj / 5차 다항식 = T)
- **환경(environment)** : 위의 물체 운반 시뮬레이터. 한 번 옮기는 것이 1 에피소드(1-step)예요.
- **정책(policy)** : 파라미터 공간의 가우시안 분포. 평균이 'AI가 생각하는 최선', 표준편차가 '탐험 정도'예요.
- **학습** : 매 반복마다 여러 움직임을 시도 → 보상이 평균보다 높았던 방향으로 정책 평균을 이동 → 탐험 폭을 점점 줄임.
"""
    )
    st.latex(r"\text{비용} = \frac{z_{peak}}{z_{lim}} + 4\,\frac{z_{rms,\ 도착\ 후}}{z_{lim}} + 10\,\frac{\text{미끄러짐}}{z_{lim}} + \lambda T,"
             r"\qquad \text{보상} = -\text{비용}")
    st.markdown(
        """
- 흔들림, 도착 후 잔류진동, 미끄러짐이 작을수록, 그리고 빠를수록(λ가 클수록 더 중요) 보상이 커요.
- 세 궤적 모두 **같은 보상, 같은 학습 방법**으로 각자 최선을 찾게 해서 공정하게 비교해요.
- 격자 탐색(가능한 조합을 모두 시험)으로 구한 최고 점수와 비교해 AI가 제대로 학습했는지 검증해요.

**한계** : 1-step 정책경사(= 컨텍스트 없는 밴딧 문제)라서, 로봇이 매 순간 저크를 직접 고르는 '완전한' 강화학습
(PPO, DDPG 등)보다는 단순해요. 대신 numpy만으로 몇 초 안에 학습되고, 결과를 해석하기 쉬워요.
"""
    )
