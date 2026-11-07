// echochat 插件：把 echochat-lite 的门控 + 步骤级信用分配接成 DSH 工具。
// 关键点：打分走引擎自己的 ctx.llm（app 账号），**不需要任何 API key**。
import { reasoningReward, responseReward, totalReward, sdcaAdvantages, diagnose } from './vendor/rewards.mjs'

export const name = 'echochat'
export const inject = ['tools', 'llm']

export const config = {
  provider: 'deepseek-account',
  model: 'deepseek-flash',
  tau: 0.6,
  maxTokens: 900,
  timeoutMs: 90_000,
}

const RUBRIC = `你是共情对话评审。按 EchoChat 的四段结构评审候选回应。

输入包含：用户的话（可能附带声学/语气线索）、一个候选回应。
你要做两件事：
1) 为这个候选回应写出三段推理轨迹（不是评价，是"这条回应背后隐含的前提"）：
   - perception：它假设用户处于什么情绪、强度多少、依据哪些线索（文字或声学）
   - mental_state：它假设用户的潜台词与真实需要是什么
   - strategy：它采用的支撑策略，以及回避了什么
2) 给分段分数（0-5，整数或 .5）：
   - s[0] 感知：轨迹的感知是否贴合给定线索（把字面当事实、忽略声学冲突线索 → 低分）
   - s[1] 心智状态：是否抓住潜台词与真实需要（被字面"我没事/别管了"骗过 → 低分）
   - s[2] 策略：支撑方式是否匹配需要、是否留出对方空间（说教或空泛鼓励 → 低分）
   - content：回应内容本身的共情质量（空洞的暖话 → 低分）
   - audio：1 表示语气与用户状态匹配，0 表示不匹配（纯文本场景按文字语气判断）

只输出一个 JSON 对象，不要代码块、不要解释：
{"trace":{"perception":"","mental_state":"","strategy":""},"scores":{"s":[0,0,0],"content":0,"audio":1}}`

/** 从模型输出里抠出第一个完整 JSON 对象（容忍代码块与前后废话）。 */
export function extractJson(text) {
  if (typeof text !== 'string') return undefined
  const stripped = text.replace(/```(?:json)?/gi, '')
  const start = stripped.indexOf('{')
  if (start < 0) return undefined
  let depth = 0, inStr = false, esc = false
  for (let i = start; i < stripped.length; i++) {
    const c = stripped[i]
    if (inStr) {
      if (esc) esc = false
      else if (c === '\\') esc = true
      else if (c === '"') inStr = false
      continue
    }
    if (c === '"') inStr = true
    else if (c === '{') depth++
    else if (c === '}') { depth--; if (depth === 0) { try { return JSON.parse(stripped.slice(start, i + 1)) } catch { return undefined } } }
  }
  return undefined
}

/** 路由兜底：想要的 provider 不在注册表里时，优先挑名字带 deepseek 的，其次第一个。 */
export function pickProvider({ wanted, available }) {
  const ids = (available ?? []).map((p) => (typeof p === 'string' ? p : p && p.id)).filter(Boolean)
  if (!ids.length) return wanted
  if (wanted && ids.includes(wanted)) return wanted
  return ids.find((id) => /deepseek/i.test(id)) ?? ids[0] ?? wanted
}

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x))
const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : undefined)

/** 把 judge 原始输出规范成 rollouts 需要的形状；缺失即判该候选"评审失败"。 */
export function normalizeJudge(raw) {
  const j = raw ?? {}
  const sc = j.scores ?? {}
  const s = Array.isArray(sc.s) ? sc.s.map(num) : []
  if (s.length !== 3 || s.some((x) => x === undefined)) return { ok: false, reason: 'scores.s 缺失或非三个数' }
  const content = num(sc.content)
  if (content === undefined) return { ok: false, reason: 'scores.content 缺失' }
  const audio = num(sc.audio) === 1 ? 1 : 0
  const t = j.trace ?? {}
  return {
    ok: true,
    scores: { s: s.map((x) => clamp(x, 0, 5)), content: clamp(content, 0, 5), audio, format: true },
    trace: {
      perception: String(t.perception ?? ''),
      mental_state: String(t.mental_state ?? ''),
      strategy: String(t.strategy ?? ''),
    },
  }
}

export function buildJudgePrompt(user, candidate) {
  return [
    RUBRIC,
    '',
    '--- 输入 ---',
    `用户：${user}`,
    `候选回应：${candidate}`,
  ].join('\n')
}

/** 逐字收集流式文本（tool-call/reasoning 分片忽略）。 */
export async function collectText(stream) {
  let out = ''
  let failure
  for await (const chunk of stream) {
    if (chunk?.type === 'text-delta') out += chunk.text ?? ''
    else if (chunk?.type === 'finish' && chunk.kind === 'error') {
      failure = chunk.failure?.message ?? chunk.failure?.code ?? 'llm stream failed'
    }
  }
  if (failure) throw new Error(failure)
  return out
}

/**
 * 纯逻辑主体：给定 llmCall(provider, model, prompt) -> Promise<string>，跑完整流程。
 * 单独抽出来是为了让 L1 测试能注入假 judge，不需要真模型。
 */
export async function scoreCandidates({ user, candidates, tau = 0.6, llmCall, provider, model, onNote }) {
  const note = onNote ?? (() => {})
  const judged = []
  for (let i = 0; i < candidates.length; i++) {
    const cand = candidates[i]
    let entry = { id: `C${i + 1}`, candidate: cand }
    try {
      let text = await llmCall(provider, model, buildJudgePrompt(user, cand))
      let norm = normalizeJudge(extractJson(text))
      if (!norm.ok) {
        note(`C${i + 1} 首次输出不合规（${norm.reason}），重试一次`)
        text = await llmCall(provider, model, `${buildJudgePrompt(user, cand)}\n\n只输出 JSON，不要任何其他字符。`)
        norm = normalizeJudge(extractJson(text))
      }
      if (!norm.ok) { entry.judgeError = norm.reason }
      else { entry.scores = norm.scores; entry.trace = norm.trace }
    } catch (e) {
      entry.judgeError = e?.message ?? String(e)
    }
    judged.push(entry)
  }
  const usable = judged.filter((j) => j.scores)
  if (usable.length === 0) return { judged, table: [], winner: undefined, skipped: judged.length }
  if (usable.length === 1) {
    const only = usable[0]
    const rr = reasoningReward(only.scores.s, { tau })
    const rp = responseReward(only.scores)
    return {
      judged,
      table: [{ id: only.id, s: only.scores.s, Rreas: rr.Rreas, gates: rr.g, rresp: rp.rresp,
                Rtotal: totalReward({ Rreas: rr.Rreas, rresp: rp.rresp, rfmt: rp.rfmt }), A: [0, 0, 0, 0],
                rootCause: rr.g[0] && rr.g[1] ? '单候选，无组内对照' : (rr.g[0] ? '心智状态推理' : '感知'),
                repair: '只有一个可用候选，无法做组内相对比较；再给一条对照才能定位' }],
      winner: only.id, single: true,
    }
  }
  const list = usable.map((u) => ({ id: u.id, s: u.scores.s, resp: u.scores }))
  const { rows } = sdcaAdvantages(list, { tau })
  const diag = diagnose(list, { tau })
  const table = usable.map((u, i) => {
    const rr = reasoningReward(u.scores.s, { tau })
    const rp = responseReward(u.scores)
    return {
      id: u.id, s: u.scores.s, Rreas: rr.Rreas, gates: rr.g, rresp: rp.rresp,
      Rtotal: totalReward({ Rreas: rr.Rreas, rresp: rp.rresp, rfmt: rp.rfmt }),
      A: rows[i].A, rootCause: diag[i].rootCause, gateFailure: diag[i].gateFailure, repair: diag[i].repair,
    }
  })
  const winner = [...table].sort((a, b) => b.Rtotal - a.Rtotal)[0].id
  return { judged, table, winner }
}

export function renderReport({ table, judged, winner, tau }) {
  const lines = []
  lines.push(`τ=${tau}　候选 ${judged.length} 个，可用 ${table.length} 个`)
  lines.push('')
  lines.push('id  [感知 推理 策略]  R_reas  门控  R_resp  R_total | A(4段)              | 最弱环节')
  for (const t of table) {
    const fmt = (x, w = 5, d = 2) => x.toFixed(d).padStart(w)
    lines.push(`${t.id.padEnd(3)} ${t.s.map((x) => String(x).padStart(4)).join(' ')} | ${fmt(t.Rreas, 6, 3)} ${t.gates.join('').padStart(5)} ${fmt(t.rresp, 6, 3)} ${fmt(t.Rtotal, 7, 3)} | ${t.A.map((x) => fmt(x)).join(' ')} | ${t.rootCause ? (t.gateFailure ? '✗ ' : '') + t.rootCause : '全达标'}`)
  }
  lines.push('')
  lines.push(`胜出：${winner}`)
  for (const t of table) lines.push(`  ${t.id}：${t.repair}`)
  const bad = judged.filter((j) => j.judgeError)
  if (bad.length) {
    lines.push('')
    lines.push('评审失败（未参与排序，不要当成低分）：')
    for (const b of bad) lines.push(`  ${b.id}：${b.judgeError}`)
  }
  return lines.join('\n')
}

export function apply(ctx, cfg = {}) {
  const c = { ...config, ...cfg }
  const off = ctx.tools.register({
    name: 'echo_judge',
    description:
      '用 EchoChat 的级联门控 + 步骤级信用（SDCA）给一组共情回应候选排序并定位该重做哪一步。'
      + '给 2-4 条候选回应，工具会用引擎自己的模型逐条写三段推理轨迹并打分，然后算出：级联奖励、门控是否通过、'
      + '每段在组内的相对优势、以及"最弱环节"。适合要回一条有潜台词的对话（逞强/阴阳怪气/文字与语气冲突）时用。'
      + ' [risk: read (read-only)]',
    parameters: {
      type: 'object',
      properties: {
        user: { type: 'string', description: '用户的原话，附带声学/语气线索（如：发紧、语速快、句尾上扬）' },
        candidates: {
          type: 'array', items: { type: 'string' }, minItems: 1, maxItems: 4,
          description: '候选回应文本，2-4 条时才能做组内相对比较',
        },
        scenario: { type: 'string', description: '可选场景说明（模态冲突/自嘲/逞强/情绪漂移等）' },
        tau: { type: 'number', description: '门控阈值，默认 0.6（来自论文）；越高越保守' },
        provider: { type: 'string', description: '可选覆盖 LLM provider' },
        model: { type: 'string', description: '可选覆盖模型 id' },
      },
      required: ['user', 'candidates'],
    },
    timeoutMs: c.timeoutMs,
    isConcurrencySafe: () => true,
    output: { schema: {}, render: (_a, v) => v.content },
    async execute(args) {
      const resolveTarget = async (a) => {
        const wanted = a.provider ?? c.provider
        let provider = wanted
        let model = a.model ?? c.model
        try {
          const available = typeof ctx.llm.listProviders === 'function' ? ctx.llm.listProviders() : []
          provider = pickProvider({ wanted, available })
          if (!a.model && provider !== wanted && typeof ctx.llm.listModels === 'function') {
            const ms = await ctx.llm.listModels(provider)
            const first = Array.isArray(ms) && ms.length ? (typeof ms[0] === 'string' ? ms[0] : ms[0] && ms[0].id) : undefined
            if (first) model = first
          }
        } catch { /* 探测失败就退回配置值 */ }
        return { provider, model }
      }
      const candidates = (args.candidates ?? []).map((x) => String(x)).filter((x) => x.trim())
      if (candidates.length === 0) throw new Error('candidates 不能为空')
      const tau = num(args.tau) ?? c.tau
      const { provider, model } = await resolveTarget(args)
      const notes = []
      const llmCall = async (prov, mod, prompt) => {
        const controller = new AbortController()
        const timer = setTimeout(() => controller.abort(), c.timeoutMs - 5000)
        try {
          const stream = ctx.llm.stream({
            provider: prov,
            model: mod,
            maxTokens: c.maxTokens,
            messages: [{ role: 'user', content: [{ type: 'text', text: prompt }] }],
            signal: controller.signal,
          })
          return await collectText(stream)
        } finally { clearTimeout(timer) }
      }
      const res = await scoreCandidates({
        user: args.scenario ? `${args.user}\n（场景：${args.scenario}）` : args.user,
        candidates, tau, llmCall, provider, model, onNote: (m) => notes.push(m),
      })
      const report = renderReport({ ...res, tau })
      const head = `provider=${provider} model=${model}`
      const body = [head, '', report, notes.length ? `\n注：${notes.join('；')}` : ''].filter(Boolean).join('\n')
      return { content: [{ type: 'text', text: body }], winner: res.winner, table: res.table }
    },
  })
  if (typeof ctx.effect === 'function' && typeof off === 'function') ctx.effect(() => off)
}
