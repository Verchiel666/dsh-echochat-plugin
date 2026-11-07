// L1：纯逻辑层。用假 judge 注入，不需要真模型。
import assert from 'node:assert/strict'
import { salvageScores, aggregateJudges, pickProvider, extractJson, normalizeJudge, buildJudgePrompt, collectText, scoreCandidates, renderReport } from '../index.mjs'

let pass = 0, fail = 0
const t = async (name, fn) => {
  try { await fn(); pass++; console.log('PASS  ' + name) }
  catch (e) { fail++; console.log('FAIL  ' + name + '\n      ' + e.message) }
}

const J = (s, content = 4, audio = 1) => JSON.stringify({
  trace: { perception: 'p', mental_state: 'm', strategy: 's' },
  scores: { s, content, audio },
})

await t('extractJson：裸 JSON', () => {
  assert.deepEqual(extractJson('{"a":1}'), { a: 1 })
})
await t('extractJson：代码块 + 前后废话', () => {
  assert.deepEqual(extractJson('说明文字\n```json\n{"a":{"b":2}}\n```\n结尾'), { a: { b: 2 } })
})
await t('extractJson：字符串里的花括号不干扰配对', () => {
  assert.deepEqual(extractJson('{"t":"含 } 和 { 的文本","n":3}'), { t: '含 } 和 { 的文本', n: 3 })
})
await t('extractJson：垃圾输入返回 undefined', () => {
  assert.equal(extractJson('完全没有 JSON'), undefined)
  assert.equal(extractJson('{"a":1'), undefined)
})
await t('normalizeJudge：合法输出被接受', () => {
  const n = normalizeJudge(JSON.parse(J([4, 3, 5], 4, 1)))
  assert.equal(n.ok, true)
  assert.deepEqual(n.scores.s, [4, 3, 5])
  assert.equal(n.scores.audio, 1)
})
await t('normalizeJudge：越界分数被夹到 [0,5]', () => {
  const n = normalizeJudge(JSON.parse(J([9, -1, 3.5], 7, 0)))
  assert.deepEqual(n.scores.s, [5, 0, 3.5])
  assert.equal(n.scores.content, 5)
})
await t('normalizeJudge：缺 s 判失败', () => {
  assert.equal(normalizeJudge({ scores: { content: 4 } }).ok, false)
})
await t('normalizeJudge：audio 写成字符串 "1" 按 1（宽容）', () => {
  assert.equal(normalizeJudge(JSON.parse(J([4, 4, 4], 4, '1'))).scores.audio, 1)
})
await t('normalizeJudge：audio 写 true 按 1、写瞎话按 0', () => {
  assert.equal(normalizeJudge({ scores: { s: [4, 4, 4], content: 4, audio: true } }).scores.audio, 1)
  assert.equal(normalizeJudge({ scores: { s: [4, 4, 4], content: 4, audio: '应该算匹配吧' } }).scores.audio, 0)
})
await t('buildJudgePrompt：含线索与候选', () => {
  const p = buildJudgePrompt('我没事（声音发紧）', '一切都会好起来的')
  assert.ok(p.includes('我没事（声音发紧）'))
  assert.ok(p.includes('一切都会好起来的'))
  assert.ok(p.includes('perception'))
})
await t('collectText：只拼 text-delta', async () => {
  async function* gen() {
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text: 'abc' }
    yield { type: 'reasoning-delta', index: 1, text: '忽略我' }
    yield { type: 'text-delta', index: 0, text: 'def' }
    yield { type: 'finish', kind: 'stop' }
  }
  assert.equal(await collectText(gen()), 'abcdef')
})
await t('collectText：finish error 抛错', async () => {
  async function* gen() { yield { type: 'finish', kind: 'error', failure: { code: 'EMPTY_RESPONSE' } } }
  await assert.rejects(() => collectText(gen()), /EMPTY_RESPONSE/)
})

// ---- 端到端逻辑（假 judge）----
const answers = {
  'A-温暖空': J([2, 4, 4], 4),
  'B-依据对但空': J([4, 3, 3], 3),
  'C-依据对且贴': J([4, 4, 5], 5),
}
const fakeLlm = async (_p, _m, prompt) => {
  for (const k of Object.keys(answers)) if (prompt.includes(k)) return answers[k]
  throw new Error('unexpected prompt')
}
const base = { user: '没事，你别管了（发紧/快/尾音上扬）', candidates: ['A-温暖空', 'B-依据对但空', 'C-依据对且贴'], llmCall: fakeLlm }

await t('scoreCandidates：最优胜出且门控生效', async () => {
  const r = await scoreCandidates({ ...base, tau: 0.6 })
  assert.equal(r.winner, 'C3')
  const a = r.table.find((x) => x.id === 'C1')
  assert.deepEqual(a.gates, [0, 1])
  assert.ok(a.Rreas < 0.3, '感知失守时级联奖励应被压低')
})
await t('scoreCandidates：感知失守者响应分最高但总分垫底', async () => {
  const r = await scoreCandidates({ ...base, tau: 0.6 })
  const sorted = [...r.table].sort((x, y) => x.Rtotal - y.Rtotal)
  const a = r.table.find((x) => x.id === 'C1')
  const c = r.table.find((x) => x.id === 'C3')
  assert.ok(a.rresp > r.table.find((x) => x.id === 'C2').rresp, 'A 的响应奖励应高于 B')
  assert.equal(sorted[0].id, 'C1', 'A 总分垫底')
  assert.ok(c.Rtotal > a.Rtotal)
})
await t('scoreCandidates：评审失败的候选不参与排序', async () => {
  const bad = async (_p, _m, prompt) => (prompt.includes('A-温暖空') ? '这不是 JSON' : fakeLlm(_p, _m, prompt))
  const r = await scoreCandidates({ ...base, llmCall: bad })
  assert.equal(r.table.length, 2)
  assert.ok(r.judged.find((j) => j.id === 'C1').judgeError)
})
await t('scoreCandidates：非法 JSON 会重试一次成功', async () => {
  let n = 0
  const firstBad = async (_p, _m, prompt) => { if (prompt.includes('只输出 JSON')) return fakeLlm(_p, _m, prompt); n++; return '第一次胡写' }
  const r = await scoreCandidates({ ...base, candidates: ['A-温暖空', 'C-依据对且贴'], llmCall: firstBad })
  assert.equal(r.table.length, 2)
  assert.ok(n >= 1)
})
await t('scoreCandidates：只有一个可用候选时标记 single', async () => {
  const r = await scoreCandidates({ ...base, candidates: ['C-依据对且贴'], llmCall: fakeLlm })
  assert.equal(r.single, true)
})
await t('renderReport：含胜出与表格', async () => {
  const r = await scoreCandidates({ ...base, tau: 0.6 })
  const txt = renderReport({ ...r, tau: 0.6 })
  assert.ok(txt.includes('胜出：C3'))
  assert.ok(txt.includes('R_reas'))
  assert.ok(txt.includes('τ=0.6'))
})

await t('pickProvider：命中配置的 provider', () => {
  assert.equal(pickProvider({ wanted: 'deepseek-account', available: [{ id: 'other' }, { id: 'deepseek-account' }] }), 'deepseek-account')
})
await t('pickProvider：不在列表时挑 deepseek 路由', () => {
  assert.equal(pickProvider({ wanted: 'deepseek-account', available: [{ id: 'openai' }, { id: 'deepseek-other' }] }), 'deepseek-other')
})
await t('pickProvider：没有 deepseek 就用第一个', () => {
  assert.equal(pickProvider({ wanted: 'deepseek-account', available: [{ id: 'openai' }] }), 'openai')
})
await t('pickProvider：注册表为空时保留原值', () => {
  assert.equal(pickProvider({ wanted: 'deepseek-account', available: [] }), 'deepseek-account')
})

await t('normalizeJudge：扁平键（perception/mental_state/strategy）', () => {
  const n = normalizeJudge({ scores: { perception: 5, mental_state: 4.5, strategy: 4.5, content: 4, audio: 1 } })
  assert.equal(n.ok, true)
  assert.deepEqual(n.scores.s, [5, 4.5, 4.5])
})
await t('normalizeJudge：s 是逗号字符串', () => {
  const n = normalizeJudge({ scores: { s: '4, 4.5, 5', content: '4' } })
  assert.deepEqual(n.scores.s, [4, 4.5, 5])
  assert.equal(n.scores.content, 4)
})
await t('normalizeJudge：分数写成字符串 + audio 布尔', () => {
  const n = normalizeJudge({ scores: { s: ['5', '5', '5'], content: '5', audio: true } })
  assert.deepEqual(n.scores.s, [5, 5, 5])
  assert.equal(n.scores.audio, 1)
})
await t('normalizeJudge：完全没有 scores 外层包装也能读', () => {
  const n = normalizeJudge({ perception: 2, mental_state: 4, strategy: 4, content: 4, audio: 0 })
  assert.deepEqual(n.scores.s, [2, 4, 4])
})
await t('normalizeJudge：真不合规时错误信息带原始输出片段', () => {
  const n = normalizeJudge({ foo: 1 }, '模型说了一堆废话 scores={} 完')
  assert.equal(n.ok, false)
  assert.ok(n.reason.includes('原始输出'), n.reason)
  assert.ok(n.reason.includes('模型说了一堆废话'), n.reason)
})

await t('aggregateJudges：取中位数（偶数样本取中间两数均值）', () => {
  const mk = (s, content = 4) => ({ scores: { s, content, audio: 1 }, trace: { perception: 'p' } })
  const agg = aggregateJudges([mk([2, 2, 2]), mk([4, 4, 4])])
  assert.deepEqual(agg.scores.s, [3, 3, 3])
  const agg2 = aggregateJudges([mk([1, 1, 1]), mk([5, 5, 5]), mk([5, 5, 5])])
  assert.deepEqual(agg2.scores.s, [5, 5, 5])
})
await t('aggregateJudges：audio 取多数', () => {
  const mk = (audio) => ({ scores: { s: [4, 4, 4], content: 4, audio }, trace: {} })
  assert.equal(aggregateJudges([mk(1), mk(0), mk(1)]).scores.audio, 1)
  assert.equal(aggregateJudges([mk(0), mk(1), mk(0)]).scores.audio, 0)
})
await t('samples=2：两次采样不一致时按中位数落定', async () => {
  let n = 0
  const wobble = async () => { n++; return n % 2 ? J([2, 2, 2], 4) : J([4, 4, 4], 4) }
  const r = await scoreCandidates({ user: 'u', candidates: ['x', 'y'], samples: 2, llmCall: wobble })
  assert.deepEqual(r.table[0].s, [3, 3, 3])
  assert.equal(r.table[0].samples, 2)
})
await t('近分并列：不宣布胜负，改为警示', async () => {
  const same = async () => J([5, 5, 5], 5)
  const r = await scoreCandidates({ user: 'u', candidates: ['a', 'b'], llmCall: same, tieMargin: 0.05 })
  assert.equal(r.tied, true)
  const txt = renderReport({ ...r, tau: 0.6 })
  assert.ok(txt.includes('⚠ 前两名差距'), txt)
  assert.ok(!txt.includes('胜出：'), '并列时不该出现"胜出"')
})
await t('差距够大时正常宣布胜出并给领先值', async () => {
  const r = await scoreCandidates({ ...base, tau: 0.6, tieMargin: 0.05 })
  const txt = renderReport({ ...r, tau: 0.6 })
  assert.ok(txt.includes('胜出：C3（领先第二名'), txt)
})

await t('salvageScores：从截断的 JSON 里抢救分数', () => {
  const trunc = '{"scores":{"s":[4,3.5,5],"content":4,"audio":1},"trace":{"perception":"它假设用户只是普通担心…'
  assert.equal(extractJson(trunc), undefined, '前提：这不是合法 JSON')
  const n = normalizeJudge(undefined, trunc)
  assert.equal(n.ok, true)
  assert.deepEqual(n.scores.s, [4, 3.5, 5])
  assert.equal(n.scores.content, 4)
  assert.equal(n.salvaged, true)
  assert.equal(n.scores.format, false, '抢救出来的应标记 format=false')
})
await t('normalizeJudge：合法 JSON 优先，不走抢救', () => {
  const n = normalizeJudge(JSON.parse(J([4, 4, 4], 4)), J([4, 4, 4], 4))
  assert.equal(n.salvaged, false)
})
await t('salvageScores：三段分数没写完就截断 -> 不抢救（不猜）', () => {
  assert.equal(normalizeJudge(undefined, '{"scores":{"s":[4,3.5,').ok, false)
})
await t('renderReport：表头显示配置的采样次数', async () => {
  const r = await scoreCandidates({ ...base, samples: 2, llmCall: fakeLlm })
  const txt = renderReport({ ...r, tau: 0.6 })
  assert.ok(txt.includes('每条采样 2 次取中位数'), txt.split('\n')[0])
})
await t('renderReport：采样不足时点名', async () => {
  // A 只在第 2 次调用（s1 的重试）成功，其余全胡写 -> 成功 1 次 < K=2
  let n = 0
  const partial = async (_p, _m, prompt) => {
    if (prompt.includes('A-温暖空')) { n++; return n === 2 ? J([2, 4, 4], 4) : '胡写' }
    return fakeLlm(_p, _m, prompt)
  }
  const r = await scoreCandidates({ ...base, samples: 2, llmCall: partial })
  const txt = renderReport({ ...r, tau: 0.6 })
  assert.ok(txt.includes('采样不足'), txt)
})

await t('截断也能直接用：scores 放最前，不再触发重试', async () => {
  let calls = 0
  const truncated = async (_p, _m, prompt) => {
    calls++
    const pick = prompt.includes('C-依据对且贴') ? [5, 5, 5, 5] : prompt.includes('B-') ? [4, 3, 3, 3] : [2, 4, 4, 4]
    // 模拟被 maxTokens 砍断：JSON 没有闭合，trace 只写了一半
    return `{"scores":{"s":[${pick[0]},${pick[1]},${pick[2]}],"content":${pick[3]},"audio":1},"trace":{"perception":"它假设用户只是普`
  }
  const notes = []
  const r = await scoreCandidates({
    user: 'u', candidates: ['A-温暖空', 'B-依据对但空', 'C-依据对且贴'], llmCall: truncated,
    onNote: (m) => notes.push(m),
  })
  assert.equal(r.table.length, 3, '三条都应可用')
  assert.equal(r.winner, 'C3')
  assert.equal(notes.length, 0, '不该出现重试提示，实际：' + notes.join(';'))
})

console.log(`\n${fail === 0 ? 'PASS' : 'FAIL'}  ${pass} passed, ${fail} failed`)
process.exit(fail === 0 ? 0 : 1)
