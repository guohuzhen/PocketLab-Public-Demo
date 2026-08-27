# PocketLab Public Demo 快速体验

## 1. 安装与启动

在 PowerShell 中执行：

```powershell
git clone https://github.com/你的用户名/PocketLab-Public-Demo.git
cd PocketLab-Public-Demo
uv sync --frozen
Copy-Item .env.example .env.local
```

在 `.env.local` 中填写你自己的 OpenAI-compatible API Key、HTTPS Base URL 和准确模型名称，然后启动：

```powershell
uv run python main.py
```

浏览器访问 <http://127.0.0.1:8000/>。

## 2. 创建本地账号

首次打开时选择“创建账号”。用户名、密码哈希、模型配置摘要、案例和实验历史保存在当前电脑的本地数据库中，不依赖邮箱登录。

不要使用重要网站的相同密码。这个公开预览版没有找回密码和管理员后台。

## 3. 配置模型

进入“设备与设置 → 模型与 API 接口”：

1. 新增配置。
2. 填写配置名称、HTTPS Base URL、准确模型名和 API Key。
3. 点击“测试能力”。
4. 确认文本能力通过；结构化输出和工具调用至少有一条兼容路径通过。

完整 Key 不应重新显示在网页、数据库或 Agent 运行审计中。

## 4. 五分钟 Exploration 示例

进入“探索实验 → 自由探索 Beta”，直接填写结构化协议：

```text
实验名称：洗衣机运行与地面振动模拟
你想比较什么？：同一测点上，洗衣机运行时的振动是否稳定高于停机时？
本次只改变的因素：洗衣机运行状态
参考条件：洗衣机停机
比较条件：洗衣机稳定运行
可选安全对照：留空
```

选择：

- 执行方式：模拟演练
- 多传感器采集方式：依次采集
- 主要传感器：加速度计
- 其他传感器：不使用

创建冻结协议后，每轮执行：

1. 阅读当前唯一任务。
2. 勾选控制条件确认。
3. 选择“清晰条件差异”。
4. 点击生成本轮模拟证据并继续。

正常路径在第四轮后结束，参考/比较条件各有两次证据。报告应说明比较条件振动指标更高，并保留模拟来源、`physical=false` 和不计入 Gate C 的边界。

## 5. 可选真机路径

如需连接 phyphox：

1. 手机和电脑连接同一个可信 Wi-Fi 或个人热点。
2. 在 phyphox 打开当前任务要求的官方实验。
3. 短时启用远程访问。
4. 在 PocketLab 设备设置中填写手机显示的私有局域网地址。
5. 测试连接，确认实验与传感器匹配后再采集。
6. 完成后立即关闭 phyphox 远程访问。

不要在公共 Wi-Fi 中使用，不要把手机地址、API Key 或家庭位置截图提交到公开 Issue。

## 6. 本地自检

```powershell
uv run python scripts/run_public_smoke.py
uv run python scripts/check_git_safety.py --tracked
uv run ruff check .
```

`run_public_smoke.py` 不调用外部模型，只检查应用导入、健康接口、本地认证和模拟协议创建。
