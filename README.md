# 套住那只羊 · 求解器

微信小游戏「套住那只羊」的本地截图识别、棋盘求解和安全点击工具。

当前版本采用单作业运行模型：采集、分析、求解、点击、重扫和校验都由同一个协调器顺序驱动。前端不再自行维护求解轮询、快速点击队列、批量执行定时器和自动恢复状态机。

## 启动

```powershell
python scripts/app.py
```

界面分为两种明确模式，并会记住用户选择；首次在窄屏打开时默认进入移动参考模式。

### 桌面操作模式

桌面操作模式有四个主动作：

1. `采集并分析`：截图、识别棋盘、生成安全报告。
2. `求解`：只计算计划，不触碰游戏。
3. `快速解法`：复用当前已识别棋盘，只清理当前及后续最多三层可直接离场的普通羊；仅在尚无棋盘时采集，整批后统一核验。
4. `连续安全执行`：新关卡先粗解并快速清掉可直接离场的普通羊，再精解剩余棋盘；特殊局面继续逐步验证。

`F6` 分析、`F7` 快速解法、`F8` 连续执行、`F12` / `Esc` 暂停。暂停不会撤销已经按下的当前点击，而是在当前点击结束后停止后续步骤。

“运行设置”可配置初始求解时限、单次续时和累计上限。启用“超时自动续时”后，搜索用完当前预算会自动追加一段时间，直到找到完整解或达到累计上限；右侧“算法过程”会实时显示搜索阶段、候选剩余数、扩展节点和续时记录。全部运行设置修改后即时保存，刷新、强刷或重启应用会自动恢复。

### 移动参考模式

移动参考模式按 iPhone 15 Pro 的 393 × 852 逻辑视口设计，只保留只读参考链路：

1. 从系统相册选择 PNG、JPEG 或 WebP 游戏截图。
2. 以截图本身作为画布，叠加网格、棋子与方向标注。
3. 从底部渐进式卡片查看识别报告，或生成解法。
4. 点击任一步骤回到画布查看可逆沙盘，不会触发游戏点击。

该模式不显示窗口选择、自动采集、快速点击、连续执行和运行参数。上传帧在后端标记为 `reference` 输入；即使绕过界面调用执行接口，也会被拒绝。

## 网页求解

<https://xhxiaiein.github.io/sliding-sheep/app/solve.html>

求解引擎有一份 TypeScript 实现（`web/solver/`，与 Python 版行为等价），编译为
`app/js/solver/` 后在浏览器 Web Worker 中运行：导入 `board.json` 或使用示例棋盘，
即可在纯网页里求解并逐步回放沙盘。识别与安全点击仍只在桌面端。

```powershell
npm ci
npm run check:solver   # tsc 类型检查
npm run test:solver    # node 直接运行 TS 等价测试
npm run build:solver   # 编译到 app/js/solver/
```

## 命令行

使用已有截图：

```powershell
python scripts/detect_occupancy.py
python scripts/solve_board.py
```

先捕获游戏窗口：

```powershell
python scripts/run.py --capture
```

## 架构

```text
游戏窗口 / 相册截图
  ├─ core/capture.py
  └─ reference upload
      └─ core/analysis.py               纯分析结果 AnalysisBundle
          ├─ board/grid.py              唯一透视/网格实现
          ├─ vision/                    视觉候选与特殊棋子识别
          ├─ recognition/               融合、全局占格、学习和时序
          └─ core/safety.py             场景与 execution_blockers

app.py（入口，Api 由 gui/ 的 mixin 组装）
  ├─ core/runtime.py OperationCoordinator  唯一后台作业、取消令牌和状态快照
  ├─ solver/planner.py                  GUI/CLI 共用的唯一求解策略
  │   ├─ solver/model.py                Board 规则模型和小盘最优 A*
  │   └─ solver/search.py               大盘 macro beam / weighted A*
  ├─ solver/learning.py                 静默增量策略画像与后台持久化
  ├─ 开局粗解：普通羊 EXIT 闭包 → 自适应快点 → 一次最终核验
  ├─ 快速解法：复用当前棋盘 → 最多三层普通羊 EXIT → 一次最终核验
  └─ 安全单步：预检 → 点击 → 重扫 → 校验 → 重求
```

详细边界见 [docs/architecture-2026-07-15.md](docs/architecture-2026-07-15.md)（文中的 `detect_occupancy.py`、`recognition.py`、`app.py` 单文件已分别拆分为 `vision/`、`recognition/`、`gui/` 包，职责边界不变）。

### 代码组织

```text
scripts/
  app.py                 pywebview 入口；Api = gui/ 全部 mixin 的组合
  run.py                 采集 + 识别 CLI 快捷入口
  solve_board.py         求解 CLI 入口
  detect_occupancy.py    识别 CLI 门面（实现在 vision/）
  paths.py               共享项目路径
  gui/                   Api 按职责拆分：common / geometry / window / settings /
                         analysis / editor / board_state / solving / workflow /
                         execution / wolf / calibration
  vision/                视觉识别：masks / segmentation / conflicts /
                         pipeline / render / export
    detectors/           每种棋子一个文件：arrow / pink_sheep / pig / goat /
                         rocket / bomb / cattle / elephant / black_sheep /
                         wolf / fence
  recognition/           模型层：features / fusion / manual_learning /
                         direction_learning / temporal
  solver/                求解域：model / search / planner / learning
  board/                 棋盘域：grid（透视网格）/ io（棋盘序列化）
  levels/                关卡域：cache（关卡缓存）/ reader（历史关卡读取）
  core/                  核心层：runtime / safety / analysis / capture
  tools/                 独立维护工具：cache_admin / recognition_regression
tests/                   pytest 测试与 conftest（负责 scripts/ 导入路径）
data/                    运行产物与持久配置（路径常量集中在 scripts/paths.py）
web/
  solver/                求解引擎 TypeScript 源码：types / board / heuristics /
                         closure / strategies / planner / worker
app/
  index.html             双模式语义结构
  solve.html             网页求解演示（Worker 求解 + 沙盘回放）
  js/                    按加载顺序拆分：core → mode → jobs → panels → board →
                         quick_review → editor → calibration → review_canvas → main
    solver/              web/solver 的编译产物（tsc，提交入库）
  css/                   desktop / shared / mobile 三层样式
```

### 关键模块

| 文件 | 职责 |
|---|---|
| `scripts/core/runtime.py` | 单后台作业、阶段、进度、暂停令牌、只读快照 |
| `scripts/core/analysis.py` | 把一次识别收敛成完整 `AnalysisBundle`，避免分析中途污染 GUI 状态 |
| `scripts/solver/planner.py` | GUI/CLI 共用求解策略；小盘最优 A*，大盘直出闭包 + macro/weighted 搜索 |
| `scripts/solver/learning.py` | 按相似棋盘累计各策略成功率、进度和耗时；内存读取、后台写盘，不阻塞求解 |
| `scripts/app.py` + `scripts/gui/` | pywebview 适配、实时窗口、人工复核、快速解法和安全执行 |
| `scripts/vision/` | 校正棋盘上的全部视觉检测器与调试渲染 |
| `scripts/recognition/` | 候选融合、全局占格、人工/方向学习与时序稳定 |
| `app/index.html` | 桌面操作与移动参考的语义结构 |
| `app/js/` | 一个作业轮询器、双模式投影和移动上传预处理，不拥有业务状态机 |
| `app/css/` | 桌面三栏与 iPhone 15 Pro Bento 截图工作台 |

## 安全语义

- 前端不传裸像素点击；执行使用 `board_revision + 当前计划第一步`。
- 自动模式在每关开局可批量执行已经证明单调安全的普通羊直出闭包；同通道动作会自动拉开间隔，整批结束后统一重扫核验并继续精解。
- 狼、炸弹羊或任一硬安全阻断出现时，开局快点不会启用；教程手势只作为画面提示，不阻止、不改序也不切换执行方式。
- 任一 `execution_blocker` 都会阻止点击。
- `manual_learning_confirmation_required`（单样本学习候选）仍是硬阻断，不能靠连续观察自动放行。
- 求解策略学习与识别安全门禁完全分离：它只静默调整后续搜索顺序和时间配比，不弹提示、不要求确认、不会授权或阻止点击。
- `hard_refresh` 清理当前截图、棋盘和诊断产物，但保留 `data/grid_params.json`、识别学习和历史缓存。
- 校准数据由 `data/grid_params.json` 持久保存；透视计算只存在于 `board/grid.py`。

## 人工复核

主画面点击棋子后打开“人工复核”：

- 修改类型或朝向会通过 `update_piece` 原位替换所选棋子，不会新增重复棋子。
- 选中反馈同时显示在真实截图覆盖层和复核列表。
- 人工棋盘需要显式确认；学习样本只有在完整证据包写入成功后才发布。
- 识别器会自动读取已保存样本：显式新增、删除、改类和朝向修正进入长期索引，完整人工棋盘中未修改的两格棋子也作为确认样本参与后续识别。
- 一条样本最多匹配当前棋盘中的一个最佳位置；单样本推断仍标记为待复核并触发 `manual_learning_confirmation_required`，负样本只有原图精确命中或得到两张独立截图支持后才会删除候选。

## 产物

当前运行产物：

```text
images/_game.png
board_grid.json
board.json
board_layout.json
scene_report.json
sheep_candidates.json
images/_occ_axis_rect.png
images/_grid_labels.png
images/_layout.png
images/_solution.png
```

持久数据：

```text
grid_params.json
retry_controls.json
runtime_settings.json
cache/source_levels/
cache/levels/
cache/manual_samples/
cache/recognition_learning/
cache/solver_strategy_learning.json
```

强刷只清理第一组运行产物。

## 验证

```powershell
python -m compileall -q scripts
Get-ChildItem app/js -Filter *.js | ForEach-Object { node --check $_.FullName }
python -m pytest -q
python scripts/detect_occupancy.py
python scripts/solve_board.py
```

识别可信度应以 `images/_occ_axis_rect.png`、`images/_grid_labels.png`、`data/sheep_candidates.json` 和 `data/scene_report.json` 为准。求解成功不代表识别一定正确；执行许可始终由安全报告决定。

## 依赖

- Python 3.11+
- `opencv-python`
- `numpy`
- `pywebview`
- `scipy`（全局候选优化）
- Windows 桌面环境

安装：

```powershell
pip install -r requirements.txt
```
