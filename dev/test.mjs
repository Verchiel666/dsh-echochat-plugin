// L1：纯逻辑层。用假 judge 注入，不需要真模型。
import assert from 'node:assert/strict'
import { pickProvider, extractJson, normalizeJudge, buildJudgePrompt, collectText, scoreCandidates, renderReport } from '../index.mjs'

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

console.log(`\n${fail === 0 ? 'PASS' : 'FAIL'}  ${pass} passed, ${fail} failed`)
process.exit(fail === 0 ? 0 : 1)
