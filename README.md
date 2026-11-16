# 湿地多台站观测记录处理服务

把不同台站听到的同一次声响归并为**同一事件**，给出发生位置、发生时间与仍不确定的候选范围，
避免"一只动物被四台站听到"被统计成四次活动。

服务同时提供**识别器升级评估**子系统（`/eval/*`，见第 10 节）：把台站端识别器新旧两版的
片段输出与完整人工标注层冻结成一份不可变快照，区分"真正的新检出"与"旧声响被重新切片"，
并按台站 × 声类 × 噪声层给出新增、漏检、误检、拆分、合并的区间估计。

- 技术栈：Python 3.12 / FastAPI / PostgreSQL 18 / SQLAlchemy 2 / NumPy
- 交付形态：**后端服务**（无前端页面），因此提供 OpenAPI 3.1 文档与 Swagger UI
- 全部数据为**虚构平面坐标与模拟台站**，不依赖真实硬件，结果**不用于预警**

> 本服务不做音频识别。台站端识别器已经给出「本机时刻 / 声响类别 / 相似性特征向量 / 误差范围」，
> 服务负责去重、校准时钟、归并同一声响、反演候选区域、定稿与修订。

---

## 1. 关键设计（与需求的对应关系）

| 需求 | 实现位置 | 说明 |
| --- | --- | --- |
| 每条观测保存站点、启动批次、源序号、本机采集时间、上传时间、类别、特征向量、误差范围 | `app/models.py::Observation`、`POST /stations/{id}/observations` | `local_capture_us` 是**台站时钟读数**，`uploaded_at_us`/`ingest_at_us` 是参考时间；两者从不混用 |
| 同批次同序号内容相同只收一次；内容不同单列冲突，不覆盖 | `app/ingest.py` | 身份 = `(station_id, boot_id, source_seq)`；相同内容 → duplicate；不同内容 → `observation_conflicts`，原文保留；整批内容指纹相同的重放才是 no-op，`batch_key` 复用但内容不同时按记录逐条判定而非整批丢弃 |
| 用已知位置的校准声源建立本机时间 → 参考时间的分段线性映射 | `app/calibration.py` | 先减去 `距离 / 配置传播速度` 得到参考到达时刻，再拟合 `local = a·ref + b`；可按残差增益自动分段（漂移） |
| 重启前后不得共用参数 | `calibration_models` 唯一键 `(station_id, boot_id, version_hash)` | 每个 boot 独立建模；斜率不合理时判定为"把重启跳时当成漂移"，拒绝该模型 |
| 锚点太少 / 残差超限时保留未校准观测，不拿上传时间替代 | `fit_calibration` / `apply_calibration` | 低于最少锚点数、跨度不足、RMS/最大残差超限、超出锚点覆盖并被计入外推，都返回 `uncalibrated` 并写明理由 |
| 以校正后的到达区间筛选候选组合，同站同次事件至多一条 | `app/association.py` | 候选组合是**团（clique）**，逐对校验 |
| 按声类、特征距离、传播时差检查相容性 | `_compatible` | `|Δt| ≤ d(站,站)/v + u_i + u_j`（三角不等式必要条件）+ 特征欧氏距离上限 |
| 不允许仅靠两两邻近传递合并相距很远的声源 | 团枚举 + 整组可定位性 | 每个候选还必须整体通过网格搜索（存在共同位置与时刻），链式传递无法通过 |
| 有限区域内由粗到细网格搜索位置与发声时刻 | `app/localize.py` | 粗网格 → 逐级细化（默认 100 m → 12.5 m），以各站 `到达 − 传播时间` 的离散程度为拟合误差 |
| 满足全部误差约束的候选区域；多个不连通区域全部返回 | `Localization.regions` | 8 邻域连通分量；每个区域给出边界、面积、代表点、单元格 |
| 几何布局不足不能报一个确定坐标 | `position_certain` / `reason` / `warnings` | 共线站、区域过大、单区域但站数不足 → 不声称确定坐标 |
| 多个事件争用同一观测时枚举互斥组合 | `app/solver.py` | 精确集合划分（整数字典序目标）：先最大化可靠解释记录数，再最大化解释总数，最后最小化总残差 |
| 出现同分解需等待人工选择 | `status = awaiting_selection` + `GET /runs/{id}/options` + `POST /runs/{id}/select` | 完全并列时**不自动选**，每套分解各自独立、不共享观测 |
| 按台站已完整上报的采集进度判断可否定稿 | `pipeline._finalization_state` | 仅当"可能听到该事件的每个台站"都以**已校准的**进度声明覆盖窗口才 final；断联站不因时间流逝被视为已报齐 |
| 允许调查员明确排除某站并保存理由 | `POST /events/{id}/exclusions`、`POST /stations/{id}/exclusions` | 记录排除理由与时间范围，只有排除掉最后一个未决台站时才推动定稿 |
| 定稿后迟报/校准修订生成新事件修订及新旧对应 | 每次运行自动 `_link_revisions`；`mode="revision"` 递增 `revision` | 旧事件置 `superseded` 并记 `superseded_by_event_id`；新事件记 `supersedes_events`，并写 `event_revision_links` |
| 处理中断从同一输入截止点继续，输出不混用不同校准参数 | `checkpoint_json` + `cutoff_us` + `calibration_json` | 暂停/恢复复用同一 cutoff 与校准快照；`collect` 阶段把输入快照整体冻结 |
| 能从事件回查原观测、所用锚点、排除理由及候选区域 | `GET /events/{id}`、`/events/{id}/anchors`、`/events/{id}/observations` | 事件详情含观测、残差、校准模型、锚点 ID、候选区域单元格、排除理由、修订链接 |
| 独立按给定声源反算到达时间检验计算 | `verifier/verify.py` | 不 import 任何服务代码，按文档化前向模型重算每条本地读数并与发布结果对比 |

### 时间模型（为什么不用到达时间归并）

```
台站本机读数 local = a_boot · 参考到达时刻 + b_boot        （每个 boot 一组参数）
参考到达时刻       = 发声时刻 + 距离(声源, 台站) / 传播速度
```

校准声源的位置和发声时刻已知，所以先扣掉传播时间，再用最小二乘拟合 `a_boot, b_boot`。
重启产生新的 `boot_id`，`b` 可以任意跳变；同 boot 内允许缓慢漂移（分段线性）。
上传时间只做记录，从不参与归并——这正是断电重启后补传记录会错位的原因。

---

## 2. 快速启动

### 2.1 Docker Compose（推荐）

```bash
docker compose up --build -d
curl -s http://127.0.0.1:8000/health
# {"status":"ok","database":"ok","version":"1.0.0"}
```

- API: <http://127.0.0.1:8000>
- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>（仓库内亦存有 `openapi.json`）
- 数据库 PGDATA 在命名卷 `dbdata` 中，`docker compose down -v` 可彻底清空
- 本次交付环境未安装 Docker CLI，因此 `docker compose` 未在本机实际执行；镜像内容是
  `pip install -r requirements.txt` + 复制源码 + `uvicorn app.main:app`，与本地验证过的启动方式一致

### 2.2 本地开发（本机 PostgreSQL）

容器内已预装 PostgreSQL 18 与 Python 3.12。默认连接串为
`postgresql+psycopg://ccdev:ccdev@127.0.0.1:5432/wetland_dev`（用环境变量 `DATABASE_URL` 覆盖）。

```bash
/opt/python-3.12/bin/python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt          # 已预置镜像源
PGPASSWORD=ccdev /opt/postgresql-18/bin/psql -h 127.0.0.1 -U ccdev -d postgres \
  -c 'CREATE DATABASE wetland_dev;'
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

表结构在应用启动时自动创建（`app/db.py::init_db`）。所有受保护接口需要请求头
`X-API-Key: dev-api-key`（`WETLAND_API_KEY` 可改，`WETLAND_AUTH_DISABLED=true` 可关闭）。
`/health`、`/docs`、`/openapi.json` 不需要鉴权。

### 2.3 一键演示 + 自检

```bash
BASE=http://127.0.0.1:8000 ./scripts/demo.sh        # 观测归并五个场景
BASE=http://127.0.0.1:8000 ./scripts/demo_eval.sh   # 识别器升级评估四个场景
```

观测归并脚本对五个内置场景逐个执行：重置 → 模拟台站分批上传 → 运行处理 → 独立校验，
最后打印每条检查的通过情况。评估脚本对四个评估场景逐个执行：冻结快照 → 计算/发布/
采用报告 → 独立校验（含来源回查与未标注时长单列）。也可以手动分步执行（见第 5、10.4 节）。

---

## 3. 测试账号 / 密钥

本服务是机器对机器的数据接口，没有用户账号体系。唯一的凭据是 API Key：

| 项 | 开发默认值 | 覆盖方式 |
| --- | --- | --- |
| API Key | `dev-api-key` | 环境变量 `WETLAND_API_KEY` / 请求头 `X-API-Key` |
| 数据库 | `ccdev/ccdev@127.0.0.1:5432/wetland_dev` | `DATABASE_URL` |
| Compose 数据库 | `wetland/wetland@db:5432/wetland` | `docker-compose.yml` |
| 关闭鉴权 | 默认开启 | `WETLAND_AUTH_DISABLED=true`（仅限隔离环境） |
| 危险操作 | `/admin/reset?confirm=true` | `WETLAND_ALLOW_RESET=false` 可禁用 |

测试库与开发库分离：`tests/conftest.py` 默认使用 `wetland_test`，测试前自动建表、
结束后删除。创建测试库：

```bash
PGPASSWORD=ccdev /opt/postgresql-18/bin/psql -h 127.0.0.1 -U ccdev -d postgres \
  -c 'CREATE DATABASE wetland_test;'
```

---

## 4. 验证方式（实际执行过的）

```bash
# 1) 单元 + 集成 + 接口契约 + 独立校验器，共 89 个用例
.venv/bin/python -m pytest
# 89 passed

# 2) 五个观测归并场景的端到端自检（真实 HTTP 服务）
BASE=http://127.0.0.1:8000 ./scripts/demo.sh

# 3) 四个识别器升级评估场景的端到端自检（真实 HTTP 服务）
BASE=http://127.0.0.1:8000 ./scripts/demo_eval.sh

# 说明：pytest 含 6 个由独立校验器驱动的端到端用例、5 个接口契约用例、
# 30 个评估子系统用例；demo.sh / demo_eval.sh 使用真实 uvicorn 进程 +
# 真实 PostgreSQL，逐个场景输出校验结论。

# 4) 单独校验某个已加载场景
.venv/bin/python -m verifier.verify --scenario reboot \
  --base-url http://127.0.0.1:8000 --api-key dev-api-key
.venv/bin/python -m verifier.verify_eval --scenario tied \
  --base-url http://127.0.0.1:8000 --api-key dev-api-key

# 5) 重新生成样例真值文件
.venv/bin/python scripts/generate_samples.py
.venv/bin/python scripts/generate_eval_samples.py

# 6) 重新导出 OpenAPI 文档
.venv/bin/python scripts/export_openapi.py
```

最近一次完整执行的结果（`scripts/demo.sh` 写出 `samples/<name>/verification.json`，该文件是运行产物、已加入 `.gitignore`）：

| 场景 | 校验项 | 结果 | 关键观测 |
| --- | --- | --- | --- |
| `overlapping` 双声源重叠 | 8 | 全部通过 | 6 事件 × 4 台站；位置误差 7.3 m；发声时刻点估计误差 5.2 ms，真值均落在发布区间内 |
| `reboot` 重启跳时 + 迟报 | 9 | 全部通过 | S2 的 `b1`/`b2` 各有独立有效模型（斜率 1.0000596 / 0.9999546）；S3 迟到 20 分钟后补传仍参与全部事件 |
| `collinear` 共线站点 | 6 | 全部通过 | 每个事件返回 2 个镜像候选区域，`position.certain=false`，不报单一坐标 |
| `duplicates` 重复补传 | 10 | 全部通过 | 整批重放 `accepted=0`；冲突记录 1 条且原文未被覆盖；事件数与记录数不变 |
| `revision` 迟报 + 校准修订 | 11 | 全部通过 | 第二次运行校准哈希不同；旧事件 `superseded` 且由新事件 `supersedes_events` 指名；实际活动数 5 条而非叠加 |

独立校验器（`verifier/verify.py`）不引用 `app/` 下任何模块，按
`arrival = emission + 距离/速度`、`local = slope_boot·arrival + intercept_boot` 从真值反算，
再逐条与实际上传的本地时间戳对比（最差 12.1 ms，来自校准声源的模拟噪声）。

---

## 5. 手动走查（curl）

```bash
KEY='X-API-Key: dev-api-key'
BASE=http://127.0.0.1:8000

curl -s -X POST "$BASE/admin/reset?confirm=true"
curl -s -X POST "$BASE/stations" -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"id":"S1","name":"North","x":300,"y":100,"hearing_radius_m":1700}'
curl -s -X POST "$BASE/calibration-sources" -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"id":"C1","x":100,"y":-150,"emissions":["2026-11-13T23:20:00.000000Z"]}'
curl -s -X POST "$BASE/sound-classes" -H "$KEY" -H 'Content-Type: application/json' \
  -d '{"id":"chirp","feature_dim":4,"feature_distance_limit":0.05}'

# 台站上传（local_capture_at 不带时区 —— 它是台站时钟读数）
curl -s -X POST "$BASE/stations/S1/observations" -H "$KEY" -H 'Content-Type: application/json' -d '{
  "station_id":"S1","boot_id":"b1","batch_key":"S1-b1-1",
  "records":[
    {"source_seq":1,"kind":"calibration","class_label":"calibration",
     "local_capture_at":"2026-11-14T09:40:12.345678","features":[0.0,0.1,0.0,0.0],
     "uncertainty_ms":12,
     "calibration":{"source_id":"C1","emission_reference_at":"2026-11-13T23:20:00.000000Z"}},
    {"source_seq":2,"kind":"sound","class_label":"chirp",
     "local_capture_at":"2026-11-14T09:41:00.000000","features":[1.0,0.30,0.20,0.90],
     "uncertainty_ms":25}
  ],
  "progress":{"complete_through_local_at":"2026-11-14T09:50:00.000000","complete":true}
}'

curl -s -X POST "$BASE/runs" -H "$KEY" -H 'Content-Type: application/json' -d '{"wait":true}'
curl -s "$BASE/observations" -H "$KEY" | python -m json.tool
```

上面这条最小样例只有 1 个校准锚点，因此**预期**结果是该观测被保留为
`calibration_status = "uncalibrated"`、`calibration_reason` 说明锚点不足，且不参与归并
——这本身就是一条要验证的约束。完整的可定位流程用模拟器一步到位：

```bash
.venv/bin/python -m simulator.run_simulator --scenario reboot --reset
curl -s -X POST "$BASE/runs" -H "$KEY" -H 'Content-Type: application/json' -d '{"wait":true}'
curl -s "$BASE/events" -H "$KEY" | python -m json.tool         # 事件、位置、候选区域
curl -s "$BASE/events/1" -H "$KEY" | python -m json.tool       # 观测、锚点、排除理由、修订
curl -s "$BASE/events/1/anchors" -H "$KEY" | python -m json.tool
```

其它常用入口：

| 接口 | 作用 |
| --- | --- |
| `GET /observations?calibration_status=uncalibrated` | 查看被保留但未校准的观测及其理由 |
| `GET /conflicts` | 查看同序号异内容的补传 |
| `GET /calibrations?station_id=S2` | 查看每个 boot 的分段线性模型、残差、锚点 |
| `GET /runs/{id}/options` | 求解并列时各套分解的对比 |
| `POST /runs/{id}/select {"option_id":"OPT-02"}` | 人工选定分解 |
| `POST /runs/{id}/resume` | 从原 cutoff 续算 |
| `GET /stats` | 总体计数 |

---

## 6. 样例与模拟器

```
samples/
  index.json            五个场景的概要
  <scenario>/truth.json 台站位置、每 boot 真值时钟模型、校准声源时刻、上传波次、真实事件
  eval/
    index.json          四个评估场景的概要
    <scenario>/truth.json 评估快照载荷（范围/标签/噪声层/两版片段/解析版本）与期望分类
```

`truth.json` 中的 `waves[].records` 就是模拟器实际 POST 的载荷；`waves[].truth` 记录每条
记录的生成来源（校准源 / 声源+叫次、本地读数、到达时刻），供校验器独立反算。

```bash
# 仅生成真值文件，不访问服务
.venv/bin/python -m simulator.run_simulator --scenario collinear --ground-truth-only

# 只上传前 4 波（模拟处理中断）
.venv/bin/python -m simulator.run_simulator --scenario overlapping --reset --pause-after-wave 4

# 只上传 early 或 late 阶段
.venv/bin/python -m simulator.run_simulator --scenario revision --phases early
.venv/bin/python -m simulator.run_simulator --scenario revision --phases late

# 重放首批并制造一条内容冲突
.venv/bin/python -m simulator.run_simulator --scenario duplicates --reset --retransmit
```

五个内置场景：

1. **overlapping** — 两个相距约 200 m 的同类声源在数秒内各自鸣叫，S2/S3 同时听到两者；
   要求"四站一声 = 1 事件"且"两兽重叠 = 2 事件"。
2. **reboot** — S2 在 t=900 s 断电重启，时钟跳变 1500 s 且漂移反号；S3 断联 20 分钟后补传；
   要求按 boot 分别校准、迟报记录仍参与，未报齐时事件保持 provisional。
3. **collinear** — 四站严格共线，跨线镜像在传播时延上不可区分；要求返回两个不连通区域，
   且不声称单点坐标，但仍正确计数。
4. **duplicates** — 整批重放（应无效）+ 同序号异内容（应入冲突表且不覆盖原文）；
   要求事件与计数不变。
5. **revision** — 先无 S3 数据跑一次，再补传 S3 与迟到校准后跑修订；要求旧事件被取代、
   新事件指名被取代者，而不是一次活动悄悄裂成两条。

---

## 7. 算法要点与已知边界

**候选组合**：按声类分组 → 时间与特征索引构图 → Bron–Kerbosch 枚举团（限制最大站数）→
每个团做整组网格定位，不可行则尝试去掉 1–2 个成员（覆盖双声源争用同一台站记录的情形）。

**定位**：在配置观测区域内做两级以上网格。对一个候选格心，计算每站
`校正到达 − 距离/速度`，以不确定度加权求均值，要求所有站都落在
`u_i + 全局松弛 + 半对角线时差` 之内（保守扩张，保证不漏掉边界区域）。可行格逐级四等分细化，
最后取 8 邻域连通分量。输出的精度就是实际到达的最细格距；若因格数上限提前停止，
事件上的 `warnings` 会说明"精度低于请求值"。

**求解**：候选 = 观测集合（含权重量）。目标为整数字典序：可靠解释数（其窗口已可定稿的观测）
> 解释总数 > 总残差平方和。中间这一层是必要的实现补充：需求只规定"可靠解释数"与"总残差"两项，
但若台站尚未报齐（可靠数为 0），仅按残差最小会退化为"少报事件"，把已收到的记录无故丢弃；
因此显式加入"解释总数"作为第二优先级，并且仅在可靠数相同时起作用。先按"共享观测"把候选图拆成互不影响的连通分量（分量最优解相加即全局最优），
每个分量做分支定界；精确同分时置 `awaiting_selection`。求解器与暴力枚举在 400 组随机小实例上逐一对拍一致。

**定稿**：`final` 需要"所有可能听到该事件的台站"满足：已校准进度覆盖窗口末尾 + 余量、
或被调查员显式排除（带理由）、或几何上不在听力半径内。断联站与未校准站都会让事件保持
`provisional`，运行状态记为 `awaiting_review`。

**修订**：任何一次重新运行只要重新解释了仍在生效的旧事件记录（或有同声类重叠窗口且共享台站），
就把旧事件置 `superseded`、新事件记录 `supersedes_events`；旧行保留（`include_superseded=true`
可查），因此不会"拆成两条又保留旧计数"。`mode="revision"` 额外把 `revision` 递增。

**边界说明（有意为之的保守行为）**

- 只承诺**既定网格精度内的候选区域**，不做精确反演，也不做统计置信区间。
- 两个站点的记录永远不构成"确定坐标"；共线布局永远返回镜像区域。
- 传播速度取配置常数，不做温度/风场修正；这是题目给定的简化模型。
- 校准不合格的观测被**保留但不参与**归并，接口可按 `calibration_status` 回查原因。
- 迟报记录不会追溯修改已定稿事件本身，而是通过下一次运行的修订关系表达。
- `/runs` 支持 `wait=false` 后台执行以便观察中断（`debug_stop_after_phase`），
  生产使用建议 `wait=true` 或外部任务队列。

---

## 8. 目录结构

```
app/
  main.py          FastAPI 应用、鉴权中间件、OpenAPI 定制
  api.py           观测归并路由
  schemas.py       Pydantic 请求/响应
  models.py        SQLAlchemy 表结构
  db.py            引擎与会话
  config.py        全部可调阈值（会写入每次运行的快照）
  timeutil.py      整数微秒时间工具
  ingest.py        上传、去重、冲突、进度
  calibration.py   锚点解析与分段线性时钟校准
  localize.py      粗到细网格搜索与候选区域
  physics.py       传播时延与相容性原语
  association.py   候选团枚举
  solver.py        互斥集合划分求解与并列检测
  pipeline.py      阶段化处理、检查点、定稿、修订
  eval_models.py   评估子系统表结构（快照、报告、匹配、分组）
  eval_schemas.py  评估子系统请求/响应
  eval_snapshot.py 快照冻结、内容哈希、覆盖率与未标注时长
  eval_match.py    连通片段簇 + 有序边界序列动态对齐（保留同代价歧义）
  eval_truth.py    人工标注层真值估计与 Wilson 区间
  eval_report.py   报告装配、版本、发布与采用
  eval_api.py      评估子系统路由（/eval/*）
simulator/
  scenarios.py     五个虚构场景与每个 boot 的时钟模型
  run_simulator.py 真值生成、可暂停的分波上传、重放/冲突注入
  eval_scenarios.py 四个评估验收场景（快照载荷 + 期望分类）
  run_eval_simulator.py 评估场景驱动器
verifier/
  verify.py        观测归并独立前向模型校验器（不引用 app/）
  verify_eval.py   评估子系统独立校验器（不引用 app/）
scripts/
  demo.sh          观测归并一键演示 + 自检
  demo_eval.sh     识别器升级评估一键演示 + 自检
  generate_samples.py
  generate_eval_samples.py
  export_openapi.py
tests/             89 个用例（导入、校准、定位、关联、求解、流水线、校验器、接口契约、评估）
samples/           五个观测场景的真值文件
samples/eval/      四个评估场景的快照与期望分类
openapi.json       导出的 OpenAPI 3.1 文档
Dockerfile / docker-compose.yml
```

## 9. 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://ccdev:ccdev@127.0.0.1:5432/wetland_dev` | 数据库连接 |
| `WETLAND_API_KEY` | `dev-api-key` | 接口密钥 |
| `WETLAND_AUTH_DISABLED` | `false` | 关闭密钥校验（仅隔离环境） |
| `WETLAND_ALLOW_RESET` | `true` | 是否允许 `/admin/reset` |
| `PROPAGATION_SPEED_MPS` | `343.0` | 传播速度 |
| `AREA_X_MIN/X_MAX/Y_MIN/Y_MAX` | `0/2100/-1100/1100` | 观测区域 |
| `CAL_MIN_ANCHORS` / `CAL_MIN_SEGMENT_ANCHORS` | `4` / `3` | 最少锚点数 |
| `CAL_MAX_RESIDUAL_MS` / `CAL_MAX_RMS_RESIDUAL_MS` | `250` / `120` | 残差上限 |
| `CAL_MIN_ANCHOR_SPAN_S` | `60` | 锚点最短跨度 |
| `CAL_COVERAGE_MARGIN_S` | `3600` | 允许的外推余量 |
| `FEATURE_DISTANCE_LIMIT` | `0.25` | 未配置声类时的特征距离上限 |
| `ASSOCIATION_TIME_SLACK_MS` | `150` | 关联松弛 |
| `GRID_COARSE_M` / `GRID_REFINE_LEVELS` | `100` / `3` | 网格精度 |
| `MIN_EVENT_STATIONS` | `2` | 事件最少台站数 |
| `MAX_SOLUTION_OPTIONS` | `5` | 并列/备选分解上限 |
| `HEARING_RADIUS_M` / `PROGRESS_MARGIN_S` | `1600` / `120` | 定稿判定 |

---

## 10. 识别器升级评估子系统（`/eval/*`）

台站端识别器从 1.x 升到 2.x 后，新输出的片段里既有**真正的新检出**，也有**旧声响被重新
切片**。本子系统把评估所需的全部输入冻结成一份**不可变快照**，用两种可互换的方法做
新旧片段关系分类，再以完整人工标注层估计真值，输出按台站 × 声类 × 噪声层分组的
区间估计报告；所有结论可回查来源，已发布采用的结论永不被覆盖。

### 10.1 与需求的对应关系

| 需求 | 实现位置 | 说明 |
| --- | --- | --- |
| 录音范围、人工标签、噪声条件、两版片段、解析版本构成不可变评价快照 | `POST /eval/snapshots`、`app/eval_snapshot.py` | 整个载荷做 SHA-256 内容哈希；同键同内容重放是幂等 no-op（200），同键不同内容拒绝（409），从不覆盖 |
| 补标或标签修订另起报告版本，不覆盖已发布采用结论 | `revision_of` + `eval_reports.report_version` | 标签属于快照，修订即新快照（同 `series_id`）；新报告在该系列内递增版本号；已发布/已采用报告只增不改，采用历史完整保留（`GET /eval/series/{id}/adopted`） |
| 连通片段簇法：时间覆盖 + 声学相似性形成连通簇，簇内判定一对多/多对一/独立 | `app/eval_match.py::cluster_groups` | 边 = 时间相交或间隙 ≤ `cluster_gap_us` 且特征距离 ≤ 上限；连通分量即簇；簇内 (旧 m, 新 n) → 1:1 / 拆分 / 合并 / 复杂 / 仅新 / 仅旧 |
| 簇法弱点：相邻同类声可能被并入同簇 | `samples/eval/adjacent` | 3 个间隔 0.2 s 的相邻叫声被簇法并为 3:3 `complex` 簇（验收样例固定此行为） |
| 动态对齐法：两版有序边界序列对齐，允许一次旧事件对应连续新事件段 | `app/eval_match.py::align_sequences` | 整数微秒代价 DP：跳过旧/新各 `skip_cost_us`，块移动（1:1、1:k 拆分、k:1 合并、k:l 复杂）代价 = 边界时间扭曲 + 重切罚项；块内片段必须时间相邻（防止跨越长静音的虚假合并），从而控制邻近事件误并 |
| 多种同代价对齐必须保留为歧义 | `alignment_groups` + `eval_matches.alternatives` | 枚举全部最优对齐（仅跳过顺序不同的路径视为同一分解）；同一组片段存在多种最优分解时整组标记 `ambiguous`，全部分解存入 `alternatives`；`samples/eval/tied` 有 2×2=4 个最优分解 |
| 真实性只能由完整人工标注层估计，未标注时长单列 | `app/eval_truth.py` | 指标只在标注覆盖区间内计算；未标注时长与落入其中的片段按噪声层单列（`unannotated_seconds`、`unannotated_segments_*`），不并入任何比率 |
| 按台站、声类、噪声层给出新增/漏检/误检/拆分/合并的区间估计 | `eval_report_groups` | 每组给出计数、比率与 Wilson 得分区间（默认 95%，`ci_level` 可调）；新增 = 新版检出而旧版漏检的标签，另记倒退（旧版检出而新版漏检） |
| 分类与来源回查 | `GET /eval/reports/{id}/matches[/{group_key}]` | 每个关系组记录新旧片段键、涉及的人工标签键、真值状态与全部同代价备选分解；报告分组计数可由匹配行与标签完整复原 |

### 10.2 数据模型与生命周期

```
POST /eval/snapshots            冻结快照（幂等；同键异内容 409）
  ├─ recording_scopes           录音范围（每台站）
  ├─ noise_layers               噪声层（同台站内不重叠，保证归属唯一）
  ├─ annotation_ranges          完整人工标注范围（真值只能在这里估计）
  ├─ labels                     人工标签（必须落在标注范围内）
  ├─ segments                   两版识别器片段（version_role = old|new）
  └─ parser_version             解析版本（片段文件已被解析层归一到参考时间）

POST /eval/snapshots/{key}/reports {"method": "cluster"|"alignment", "params": {...}}
  → 计算报告：同 (快照, 方法, 参数) 重复请求幂等返回（200）；
    输入真正变化时才在该系列内递增 report_version
POST /eval/reports/{id}/publish 发布（结论冻结）
POST /eval/reports/{id}/adopt   采用（需先发布；不清除历史采用记录）
```

报告参数（全部随报告冻结，可复算）：`cluster_gap_us`、`feature_distance_limit`、
`skip_cost_us`、`reseg_penalty_us`、`align_max_gap_us`、`max_warp_us`、`max_block`、
`max_alignments`、`ci_level`。

### 10.3 验收样例（`samples/eval/`）

| 场景 | 验收点 | 结果 |
| --- | --- | --- |
| `reseg3` 长声被三段切开 | 两种方法都判 1 旧 : 3 新为 `split`；真值层记新版拆分率 1.0（含 Wilson 区间）；匹配行回链标签 L1 | 28 项检查全过 |
| `adjacent` 相邻重复叫声 | 簇法把 3 个相邻叫声并为 3:3 `complex`（弱点固化为预期）；对齐法保持 3 个 `one_to_one` | 30 项检查全过 |
| `rain` 雨天噪声 | 雨层误检 2 条单列；静风层新增 1 条；未标注 300 s 与其中 1 条片段单列；补标后 `eval-rain-r2` 另起 v2，v1 已发布结论逐字节不变 | 34 项检查全过 |
| `tied` 并列对齐 | 两处同代价（匹配 0.8 s ≡ 双跳过 0.8 s）→ 4 个最优分解全部保留，2 组标记 `ambiguous` 且备选分解齐全 | 32 项检查全过 |

### 10.4 手动走查（curl）

```bash
KEY='X-API-Key: dev-api-key'
BASE=http://127.0.0.1:8000

# 加载验收场景（幂等，可重复执行）
.venv/bin/python -m simulator.run_eval_simulator --scenario tied

# 报告与分组区间估计
curl -s "$BASE/eval/reports?snapshot_key=eval-tied" -H "$KEY" | python -m json.tool
curl -s "$BASE/eval/reports/1" -H "$KEY" | python -m json.tool

# 来源回查：歧义组的全部同代价备选分解
curl -s "$BASE/eval/reports/1/matches?ambiguous=true" -H "$KEY" | python -m json.tool

# 快照覆盖与未标注时长
curl -s "$BASE/eval/snapshots/eval-rain" -H "$KEY" | python -m json.tool
```

### 10.5 已知边界（有意为之）

- 簇法不感知次序，相邻同类声会被传递性并入同簇——这是该方法的固有弱点，
  验收样例 `adjacent` 把它固定为预期行为；需要区分邻近事件时改用对齐法。
- 对齐法的结果依赖时间扭曲与跳过代价的选择；代价随报告冻结，换一组代价即
  新的报告版本，旧结论不受影响。
- 歧义组不参与确定关系计数（`relation_counts`），单独计入 `ambiguous_groups`；
  对齐数超过 `max_alignments` 时报告标记 `alignment_count_truncated`。
- 真值比率只在完整标注层内估计；未标注时段与其中的片段永远单列，不参与任何比率。

