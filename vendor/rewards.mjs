// EchoChat 核心机制实现（论文 NeurIPS'26, arXiv:2610.04826）
// 严格对齐论文公式：(3) 门控 (4) 级联推理奖励 (5) 响应奖励 (6) 总奖励 (7)(8) SDCA 每段组内相对优势

const EPS = 1e-3; // 组内标准差下限，避免 σ→0 爆炸（论文 Eq.8 的 ε）

export function assert(cond, msg) { if (!cond) throw new Error(msg); }

/** 5 分制 -> [0,1]，并列做区间校验 */
export function norm(s5) {
  assert(typeof s5 === 'number' && Number.isFinite(s5), `score must be a finite number, got ${s5}`);
  assert(s5 >= 0 && s5 <= 5, `score out of rubric range [0,5]: ${s5}`);
  return s5 / 5;
}

/**
 * Eq.(3)(4)：级联门控推理奖励
 * @param {number[]} s 三阶段原始分 [感知, 心智状态推理, 支持策略]，各 0..5
 * @param {{tau?:number, alpha?:number[]}} o tau 门控阈值；alpha 三段权重且和为 1
 */
export function reasoningReward(s, o = {}) {
  const tau = o.tau ?? 0.6;
  const alpha = o.alpha ?? [0.4, 0.3, 0.3];
  assert(Array.isArray(s) && s.length === 3, 'reasoningReward needs 3 stage scores');
  assert(Math.abs(alpha.reduce((a, b) => a + b, 0) - 1) < 1e-9, 'alpha must sum to 1');
  assert(tau >= 0 && tau <= 1, 'tau must be in [0,1]');
  const sn = s.map(norm);
  const g1 = sn[0] >= tau ? 1 : 0;
  const g2 = sn[1] >= tau ? 1 : 0;
  const contrib = [alpha[0] * sn[0], alpha[1] * sn[1] * g1, alpha[2] * sn[2] * g1 * g2];
  return { Rreas: contrib[0] + contrib[1] + contrib[2], g: [g1, g2], norm: sn, contrib, tau, alpha };
}

/** Eq.(5)：响应级奖励 = 内容共情(0..5 归一) + 音频情绪对齐(0/1) */
export function responseReward(x, o = {}) {
  const wc = o.omegaC ?? 0.7, wa = o.omegaA ?? 0.3;
  assert(Math.abs(wc + wa - 1) < 1e-9, 'omegaC+omegaA must be 1');
  const rc = norm(x.content);
  const ra = x.audio ? 1 : 0;
  return { rresp: wc * rc + wa * ra, rcontent: rc, raudio: ra, rfmt: x.format === false ? 0 : 1 };
}

/** Eq.(6)：轨迹总奖励 */
export function totalReward(x, w = {}) {
  const ww = { reas: w.reas ?? 0.5, resp: w.resp ?? 0.4, fmt: w.fmt ?? 0.1 };
  assert(Math.abs(ww.reas + ww.resp + ww.fmt - 1) < 1e-9, 'weights must sum to 1');
  return ww.reas * x.Rreas + ww.resp * x.rresp + ww.fmt * x.rfmt;
}

/** Eq.(7)：把一次 rollout 拆成四段标量奖励 [r1,r2,r3,rR]（前三段复用门控后的级联分量） */
export function spanRewards(roll, o = {}) {
  const rr = reasoningReward(roll.s, o);
  const rp = responseReward(roll.resp, o);
  const wa = o.omegaResp ?? 0.6, wf = o.omegaFmt ?? 0.4;
  assert(Math.abs(wa + wf - 1) < 1e-9, 'omegaResp+omegaFmt must be 1');
  return { spans: [...rr.contrib, wa * rp.rresp + wf * rp.rfmt], rr, rp };
}

/** Eq.(8)：SDCA —— 每个 span 在组内独立算相对优势 A_k=(r_k-μ_k)/(σ_k+ε) */
export function sdcaAdvantages(rollouts, o = {}) {
  assert(rollouts.length >= 1, 'need at least one rollout');
  const rows = rollouts.map((r) => ({ id: r.id, ...spanRewards(r, o) }));
  const out = rows.map((row) => ({ id: row.id, A: [0, 0, 0, 0], spans: row.spans }));
  for (let k = 0; k < 4; k++) {
    const xs = rows.map((r) => r.spans[k]);
    const mu = xs.reduce((a, b) => a + b, 0) / xs.length;
    const sd0 = Math.sqrt(xs.reduce((a, b) => a + (b - mu) ** 2, 0) / xs.length);
    const sd = Math.max(sd0, 0); // 论文用 σ+ε 作分母
    const denom = sd + (o.epsilon ?? EPS);
    out.forEach((row, i) => { row.A[k] = sd === 0 ? 0 : (xs[i] - mu) / denom; });
  }
  // 每段 token 数 -> 逐 token 优势（这里按段长度展开，等价于论文"把 A_k 赋给该段所有 token"）
  return { rows: out, perToken: (lens) => out.map((r) => lens.flatMap((n, k) => Array(n).fill(r.A[k]))) };
}

/** 轨迹级回退：段边界解析失败时用总奖励的组内相对优势（论文第 3.2.3 节 safeguard） */
export function fallbackAdvantage(rollouts, o = {}) {
  const R = rollouts.map((r) => {
    const rr = reasoningReward(r.s, o), rp = responseReward(r.resp, o);
    return totalReward({ Rreas: rr.Rreas, rresp: rp.rresp, rfmt: rp.rfmt }, o.w);
  });
  const mu = R.reduce((a, b) => a + b, 0) / R.length;
  const sd = Math.sqrt(R.reduce((a, b) => a + (b - mu) ** 2, 0) / R.length);
  return R.map((x) => (sd === 0 ? 0 : (x - mu) / (sd + (o.epsilon ?? EPS))));
}

/**
 * 错误传播诊断：定位"该重做哪一步"。
 * 注意（实测踩到的坑）：不能直接用 SDCA 的 A_k 找最小——门控会把上游失守时的下游分值清零，
 * 于是 A_2/A_3 一起变负，argmin 会指到"心智状态推理"而不是真正的根因。根因要看**未加门**的原始分。
 */
export function diagnose(rollouts, o = {}) {
  const tau = o.tau ?? 0.6;
  const { rows } = sdcaAdvantages(rollouts, o);
  const names = ['感知', '心智状态推理', '支持策略', '回应'];
  return rows.map((r, i) => {
    const rr = reasoningReward(rollouts[i].s, o);
    const g = rr.g;
    const raw = rr.norm;
    let rootCause, repair;
    if (!g[0]) {
      rootCause = '感知';
      repair = '重做感知：先确认情绪/强度/韵律线索，再谈推断与回应（Eq.4 已把下游清零，改下游无效）';
    } else if (!g[1]) {
      rootCause = '心智状态推理';
      repair = '感知已达标，重做心智状态推断（潜台词/未说出口的需要），策略项此时不计分';
    } else {
      const min = Math.min(...raw);
      const ties = [0, 1, 2].filter((k) => Math.abs(raw[k] - min) < 1e-9).map((k) => names[k]);
      if (min >= 1) { rootCause = null; repair = '全段满分，无需返工'; }
      else { rootCause = ties.join('/'); repair = `各段均过门控，按绝对分最低的一步优化「${ties.join(' / ')}」`; }
    }
    const worstByA = names[r.A.indexOf(Math.min(...r.A))];
    return {
      id: r.id, stageScores: rollouts[i].s, norm: raw.map((v) => +v.toFixed(4)),
      gates: g, gatedSpans: r.spans.map((v) => +v.toFixed(4)), A: r.A.map((v) => +v.toFixed(4)),
      rootCause, repair, gateFailure: !g[0] || !g[1], weakestByAdvantage: worstByA,
      notes: !g[0]
        ? ['感知未过门控：下游两项在本轮不计分（抑制"表面漂亮"的补偿路径）']
        : (!g[1] ? ['心智状态推理未过门控：策略项不计分'] : []),
    };
  });
}
