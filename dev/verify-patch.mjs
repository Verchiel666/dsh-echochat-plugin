// L3：用引擎同款 YAML 方言校验两种部署形态
//   ① 包内 cordis.patch.yml（App 安装路线，name = 包名）
//   ② deploy/cordis.patch.snippet.yml（手工追加路线，name = 绝对路径）
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

function engineModules() {
  if (process.env.DSH_ENGINE_NODE_MODULES) return process.env.DSH_ENGINE_NODE_MODULES
  const dshHome = process.env.DSH_HOME
  if (!dshHome) return null
  const root = path.join(dshHome, '..', 'engine')
  let dirs = []
  try { dirs = fs.readdirSync(root) } catch { return null }
  for (const d of dirs) {
    const cand = path.join(root, d, 'node_modules')
    if (fs.existsSync(path.join(cand, 'js-yaml'))) return cand
  }
  return null
}
const ENGINE = engineModules()
if (!ENGINE) { console.error('找不到引擎 node_modules'); process.exit(2) }
const yaml = (await import(pathToFileURL(path.join(ENGINE, 'js-yaml', 'index.js')).href)).default
const JsExpr = new yaml.Type('tag:yaml.org,2002:js', { kind: 'scalar', resolve: (d) => typeof d === 'string', construct: (d) => ({ __jsExpr: d }) })
const load = (raw) => yaml.load(raw, { schema: yaml.JSON_SCHEMA.extend(JsExpr) })

let pass = 0, fail = 0
const t = async (name, fn) => {
  try { await fn(); pass++; console.log('PASS  ' + name) }
  catch (e) { fail++; console.log('FAIL  ' + name + '\n      ' + e.message) }
}
const root = path.join(import.meta.dirname, '..')
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'))
const bundleRaw = fs.readFileSync(path.join(root, 'cordis.patch.yml'), 'utf8')
const bundle = load(bundleRaw)
const entry = bundle?.[0]?.insert?.[0]

console.log('— ① 包内 cordis.patch.yml（App 安装路线）—')
await t('能解析为含 insert 的 entry 数组', () => {
  if (!Array.isArray(bundle) || bundle.length !== 1) throw new Error('顶层应是单条 entry')
  if (!entry) throw new Error('缺 insert[0]')
})
await t('id 正确且 name 是包名字面量（不是 !!js）', () => {
  if (entry.id !== 'echochat') throw new Error('id 不对: ' + entry.id)
  if (typeof entry.name !== 'string') throw new Error('name 不是字符串（!!js 陷阱）')
  if (entry.name !== pkg.name) throw new Error(`name(${entry.name}) 必须等于包名(${pkg.name})`)
})
await t('config 全是字面值且含 provider/model/tau', () => {
  for (const [k, v] of Object.entries(entry.config ?? {})) {
    if (v && typeof v === 'object' && '__jsExpr' in v) throw new Error(`config.${k} 被解析成 !!js`)
  }
  for (const k of ['provider', 'model', 'tau']) if (!(k in (entry.config ?? {}))) throw new Error('缺 ' + k)
})
await t('package.json 已声明 dsh.bundle.patch 指向该文件', () => {
  if (pkg.dsh?.bundle?.patch !== './cordis.patch.yml') throw new Error('dsh.bundle.patch 不对: ' + pkg.dsh?.bundle?.patch)
})
await t('files 打包包含 cordis.patch.yml 与 index.mjs', () => {
  for (const f of ['index.mjs', 'cordis.patch.yml']) if (!pkg.files.includes(f)) throw new Error('files 缺 ' + f)
})
await t('capabilities 声明了 tools:register', () => {
  if (!pkg.dsh?.capabilities?.includes('tools:register')) throw new Error('缺 tools:register')
})
await t('包名应小于 npm 长度上限 214', () => {
  if (pkg.name.length > 214) throw new Error('包名过长')
})

await t('包内 patch 的数值配置不能与模块默认值漂移（曾差点用 900 覆盖掉 2000）', async () => {
  const mod = await import(pathToFileURL(path.join(root, 'index.mjs')).href)
  const drift = []
  for (const k of ['tau', 'maxTokens', 'samples', 'tieMargin']) {
    if (entry.config && k in entry.config && mod.config && k in mod.config && entry.config[k] !== mod.config[k]) {
      drift.push(`${k}: patch=${entry.config[k]} vs 模块=${mod.config[k]}`)
    }
  }
  if (drift.length) throw new Error('配置漂移 ' + drift.join('；'))
})

console.log('— ② deploy/cordis.patch.snippet.yml（手工路线）—')
const snip = load(fs.readFileSync(path.join(root, 'deploy', 'cordis.patch.snippet.yml'), 'utf8'))
const snipEntry = snip?.[0]?.insert?.[0]
await t('片段可解析且 name 是存在的绝对路径', () => {
  if (!snipEntry) throw new Error('片段结构不对')
  if (!path.isAbsolute(snipEntry.name)) throw new Error('不是绝对路径')
  if (!fs.existsSync(snipEntry.name)) throw new Error('文件不存在: ' + snipEntry.name)
})

console.log('— ③ 真实 profile patch（只读检查，别动它）—')
const realRaw = fs.readFileSync(path.join(process.env.DSH_HOME, 'profiles', 'android', 'cordis.patch.yml'), 'utf8')
await t('真实 patch 可解析且里面没有 id: echochat', () => {
  const doc = load(realRaw)
  if (!Array.isArray(doc)) throw new Error('不是数组')
  if (realRaw.includes('id: echochat')) throw new Error('已存在，重复插入会冲突')
})
await t('真实 patch 里没有 entry 把 !!js 写进 name', () => {
  for (const e of load(realRaw)) for (const item of e.insert ?? []) {
    if (item.name && typeof item.name === 'object' && '__jsExpr' in item.name) throw new Error('entry ' + item.id + ' 的 name 用了 !!js')
  }
})

console.log(`\n${fail === 0 ? 'PASS' : 'FAIL'}  ${pass} passed, ${fail} failed`)
process.exit(fail === 0 ? 0 : 1)
