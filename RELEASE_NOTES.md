# PocketLab `v0.1.0-competition-preview`

这是供赛事评审和技术查看使用的候选快照。核心目标不是展示一段预制聊天，而是让评审者能够克隆仓库后，亲自走完两条有状态、有证据、有终止条件的 Agent 产品链路。

## 本候选版包含

- “新建诊断”中的洗衣机振动回放：2 次点击依次绑定基线与单变量对照，更新竞争假设并形成有边界的最终报告。
- “探索实验”中的光学回放：4 次点击完成两个距离条件的重复测量，显示照度条件图、充分度和物理解释。
- 用户可选的 High / Fast 模式、可见输出流式状态、服务端结果校验，以及只能由用户主动触发的 fallback。
- 结构失败时的真实基模修复路径：先保留严格协议，再用带验证反馈的兼容协议重试；不会暗中换成确定性答案。
- 注册、回放、刷新恢复、历史恢复、重复写入和控制台错误的 Playwright 黑盒验收。
- Git 安全扫描、锁定依赖和公开候选版 GitHub Actions 门禁。

## 可复现验收

```powershell
uv sync --frozen
uv run python scripts/check_git_safety.py --tracked
uv run ruff check .
uv run python scripts/run_public_smoke.py
uv run playwright install chromium
uv run python scripts/run_browser_e2e.py
```

`run_public_smoke.py` 和 `run_browser_e2e.py` 都是零 Key 验收，不调用外部模型。真实模型路径必须由使用者提供自己的 OpenAI-compatible 配置。

## 证据与范围边界

回放数据明确标记为非物理 fixture，只证明软件闭环与 UI 可以运行，不证明当前家庭设备或手机上的现实结论。公开仓库不包含 API Key、模型地址、账号数据库、Cookie、局域网 phyphox 地址、真实手机原始数据、私有 Evals、提交材料或原开发仓库历史。

真实 phyphox 采集只能在手机与电脑同处可信局域网、设备当前可达且用户实际执行实验时验收；公开截图和自动化测试不会冒充真机证据。
