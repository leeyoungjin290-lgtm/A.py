"""물체 운반 시뮬레이션 + 강화학습(정책경사) 핵심 로직 (numpy 전용)

■ 물리 모델 (단순화한 스틱-슬립 모델)
   쟁반(로봇 그리퍼)이 궤적 a_t(t)로 움직이고, 그 위의 물체는 탄성 패드(고유진동수 fn, 감쇠비 ζ)로
   쟁반에 붙어 있다.  z = 쟁반에 대한 물체의 상대 변위.
        z'' = -a_t - ω² z - 2ζω z'
   패드가 낼 수 있는 힘은 마찰 한계 μg 까지이므로 |z| > μg/ω² 가 되면 물체가 미끄러진다.
   (미끄러진 거리를 slip 으로 누적)

■ 강화학습 (REINFORCE, 가우시안 정책)
   행동  = 궤적 파라미터(이동시간 T, 가속시간 비율 ta/T, 저크구간 비율 tj/ta)
   보상  = -(흔들림 + 잔류진동 + 미끄러짐 + λ·시간)   ← 한 번 옮기면 1 에피소드(1-step)
"""
import numpy as np

G = 9.81
DT = 0.003
TMIN, TMAX = 0.8, 3.0
SETTLE = 1.0
TGRID = np.arange(0.0, TMAX + SETTLE, DT)

KINDS = {"trap": "사다리꼴", "scurve": "S-curve", "quintic": "5차 다항식"}
DIM = {"trap": 2, "scurve": 3, "quintic": 1}
TA_LO, TA_HI = 0.08, 0.50
TJ_LO, TJ_HI = 0.05, 0.50


# ───────────────────────── 궤적 (가속도) ─────────────────────────
def accel(kind, T, ta_r, tj_r, D, t):
    """쟁반 가속도 a_t(t).  T, ta_r, tj_r 는 t 와 브로드캐스트 가능한 배열."""
    if kind == "quintic":
        tau = np.clip(t / T, 0.0, 1.0)
        a = D / T**2 * (60 * tau - 180 * tau**2 + 120 * tau**3)
        return np.where(t < T, a, 0.0)
    ta = ta_r * T
    tj = np.maximum(tj_r * ta, 1e-9) if kind == "scurve" else np.full_like(ta, 1e-9)
    vp = D / (T - ta)
    A = vp / (ta - tj)

    def g(s):
        return A * (np.clip(s / tj, 0, 1) - np.clip((s - (ta - tj)) / tj, 0, 1))

    return g(t) - g(t - (T - ta))


def peak_accel_jerk(kind, T, ta_r, tj_r, D):
    """최대 가속도, 최대 저크(해석식).  사다리꼴 저크는 이론상 ∞"""
    if kind == "quintic":
        return 10 / np.sqrt(3) * D / T**2, 60 * D / T**3
    ta = ta_r * T
    tj = np.maximum(tj_r * ta, 1e-9) if kind == "scurve" else np.full_like(ta, 1e-9)
    A = D / (T - ta) / (ta - tj)
    return A, (A / tj if kind == "scurve" else np.full_like(A, np.inf))


# ───────────────────────── 물체 시뮬레이션 ─────────────────────────
def object_sim(a_t, T, fn, zeta, mu_f, keep=False):
    """a_t: (N, S) 쟁반 가속도.  반환: dict of (N,) 지표 (+ keep 이면 시계열)"""
    N, S = a_t.shape
    aT = np.ascontiguousarray(a_t.T)
    w = 2 * np.pi * fn
    w2 = w * w
    cd = 2 * zeta * w
    zmax = mu_f * G / w2
    z = np.zeros(N); zd = np.zeros(N); p = np.zeros(N)
    peak = np.zeros(N); slip = np.zeros(N)
    ssq = np.zeros(N); cnt = np.zeros(N); rmax = np.zeros(N)
    if keep:
        Z = np.empty((N, S)); P = np.empty((N, S)); SL = np.zeros((N, S), dtype=bool)
    for i in range(S):
        zd += (-aT[i] - w2 * z - cd * zd) * DT
        z += zd * DT
        over = np.abs(z) > zmax
        if over.any():
            d = np.abs(z) - zmax
            sg = np.sign(z)
            slip += np.where(over, d, 0.0)
            p += np.where(over, sg * d, 0.0)
            z = np.where(over, sg * zmax, z)
            zd = np.where(over, 0.0, zd)
        np.maximum(peak, np.abs(z), out=peak)
        ti = TGRID[i]
        win = (ti > T) & (ti <= T + SETTLE)
        az = np.abs(z)
        ssq += np.where(win, z * z, 0.0)
        cnt += win
        np.maximum(rmax, np.where(win, az, 0.0), out=rmax)
        if keep:
            Z[:, i] = z; P[:, i] = p; SL[:, i] = over
    out = dict(peak=peak, slip=slip, rrms=np.sqrt(ssq / np.maximum(cnt, 1)), rmax=rmax)
    if keep:
        out.update(Z=Z, P=P, SL=SL)
    return out


def evaluate(kind, T, ta, tj, D, fn, zeta, mu_f, lam, z_lim, keep=False):
    """파라미터 N개를 한꺼번에 평가. 모두 (N,) 배열(또는 스칼라)."""
    T = np.atleast_1d(np.asarray(T, float))
    N = len(T)
    bc = lambda x: np.broadcast_to(np.asarray(x, float), (N,)).copy()
    ta, tj, fn, zeta, mu_f = map(bc, (ta, tj, fn, zeta, mu_f))
    a = accel(kind, T[:, None], ta[:, None], tj[:, None], D, TGRID[None, :])
    m = object_sim(a, T, fn, zeta, mu_f, keep=keep)
    cost = (m["peak"] / z_lim + 4.0 * m["rrms"] / z_lim
            + 10.0 * m["slip"] / z_lim + lam * T)
    m["success"] = (m["slip"] <= 0.5e-3) & (m["peak"] <= z_lim) & (m["rmax"] <= 0.3 * z_lim)
    m["cost"] = cost
    m["a"] = a
    return m


# ───────────────────────── 강화학습 ─────────────────────────
def to_params(theta, kind):
    """비제한 공간 θ → 실제 파라미터 (시그모이드로 범위 제한)"""
    s = 1.0 / (1.0 + np.exp(-theta))
    T = TMIN + (TMAX - TMIN) * s[:, 0]
    ta = TA_LO + (TA_HI - TA_LO) * s[:, 1] if kind != "quintic" else np.full(len(s), 0.25)
    tj = TJ_LO + (TJ_HI - TJ_LO) * s[:, 2] if kind == "scurve" else np.zeros(len(s))
    return T, ta, tj


def train(kind, D, scen, lam, z_lim, iters=60, batch=24, seed=0, lr=1.5, progress=None):
    """REINFORCE(가우시안 정책).  scen = dict(fn, zeta, mu)"""
    rng = np.random.default_rng(seed)
    d = DIM[kind]
    mean = np.zeros(d)
    hist_eval, hist_batch = [], []
    for k in range(iters):
        sigma = max(0.15, 1.0 * 0.97**k)
        eps = rng.standard_normal((batch, d))
        theta = np.vstack([mean, mean + sigma * eps])   # 0번 행: 현재 정책의 평균 행동(평가용)
        T, ta, tj = to_params(theta, kind)
        m = evaluate(kind, T, ta, tj, D, scen["fn"], scen["zeta"], scen["mu"], lam, z_lim)
        r = -m["cost"]
        hist_eval.append(m["cost"][0]); hist_batch.append(m["cost"][1:].mean())
        adv = (r[1:] - r[1:].mean()) / (r[1:].std() + 1e-8)
        mean = mean + lr * sigma * (adv[:, None] * eps).mean(axis=0)
        if progress:
            progress((k + 1) / iters)
    T, ta, tj = to_params(mean[None, :], kind)
    return dict(kind=kind, theta=mean, T=float(T[0]), ta=float(ta[0]), tj=float(tj[0]),
                hist_eval=np.array(hist_eval), hist_batch=np.array(hist_batch))


def grid_best(kind, D, scen, lam, z_lim):
    """격자 탐색으로 구한 최고 점수(AI 결과 검증용)"""
    if kind == "quintic":
        g = [np.linspace(TMIN, TMAX, 60), [0.25], [0.0]]
    elif kind == "trap":
        g = [np.linspace(TMIN, TMAX, 30), np.linspace(TA_LO, TA_HI, 30), [0.0]]
    else:
        g = [np.linspace(TMIN, TMAX, 14), np.linspace(TA_LO, TA_HI, 14), np.linspace(TJ_LO, TJ_HI, 14)]
    T, ta, tj = [x.ravel() for x in np.meshgrid(*g, indexing="ij")]
    best = np.inf
    for s in range(0, len(T), 800):
        sl = slice(s, s + 800)
        m = evaluate(kind, T[sl], ta[sl], tj[sl], D, scen["fn"], scen["zeta"], scen["mu"], lam, z_lim)
        best = min(best, m["cost"].min())
    return best


def cumint(f, t):
    """사다리꼴 공식 누적 적분 (시각화용)"""
    return np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(t))])
