# PocketLab Public Demo 快速体验

## 1. 安装与启动

在 PowerShell 中执行：

```powershell
git clone https://github.com/guohuzhen/PocketLab-Public-Demo.git
cd PocketLab-Public-Demo
uv sync --frozen
Copy-Item .env.example .env.local
```

可以直接启动并体验零等待回放；如需运行真实基模，再在 `.env.local` 中填写你自己的 OpenAI-compatible API Key、HTTPS Base URL、准确模型名称，并选择 `LLM_REASONING_STRATEGY=high` 或 `fast`：

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
3. 选择 `High · 质量优先` 或 `Fast · 速度优先`。
4. 点击“测试能力”。
5. 确认文本能力通过；结构化输出和工具调用至少有一条兼容路径通过。

完整 Key 不应重新显示在网页、数据库或 Agent 运行审计中。真实请求会显示连接、流式生成和服务端校验阶段；High 超过两分钟后可由用户切换到 Fast，Fast 不会反向切到 High，任何安全兜底都必须由用户主动选择。

## 4. 五分钟零等待体验

这两条路径不需要 API Key、网络模型或手机：

### 4.1 洗衣机诊断

1. 打开“新建诊断”，点击“洗衣机诊断 · 2 步”。
2. 阅读两个竞争解释和当前唯一任务。
3. 点击“回放本步并立即推进”完成偏载基线。
4. 再点击一次完成均匀分布对照。
5. 查看 RMS、主频、假设更新、终止依据与低风险建议。

### 4.2 光学探索

1. 打开“探索实验”，点击“光学探索 · 4 步”。
2. 按当前任务连续推进四轮。
3. 查看近距离与距离加倍条件、两组重复、照度对比图、充分度和描述性报告。

回放调用正式状态机和确定性分析器，但记录明确标为 `test_fixture` 或 `protocol_emulator`、`physical=false`、`Gate C +0`。它们证明软件闭环可运行，不证明当前洗衣机、台灯或手机的现实结论。

## 5. 可选真机路径

### 5.1 安装 phyphox

请使用稳定版并从官方入口安装：

- **iPhone / iPad**：打开 [phyphox 官方下载页](https://phyphox.org/download/)，点击 App Store 入口安装。
- **可使用 Google Play 的 Android**：从同一官方下载页进入 Google Play 安装。
- **无 Google Play 的 Android**：这在部分国产手机环境中较常见。打开 [F-Droid 的 phyphox 条目](https://f-droid.org/en/packages/de.rwth_aachen.phyphox/)，优先按页面建议安装 F-Droid 官方客户端；刷新软件源后搜索 `phyphox`，核对包名为 `de.rwth_aachen.phyphox`、作者为 RWTH Aachen University，再安装。

F-Droid 页面也提供直接 APK，但官方提示这种方式没有正常的更新通知，安全性也低于使用 F-Droid 客户端，因此只建议在客户端路径不可用时备用。Android 如提示“安装未知应用”，只对从 [F-Droid 官方网站](https://f-droid.org/) 获取的安装器临时授权，安装完成后按自己的更新需求复核或撤销该权限；不要使用第三方 APK 镜像站。

### 5.2 连接 PocketLab

1. 手机与运行 PocketLab 的电脑连接同一个可信 Wi-Fi 或个人热点。
2. 在 phyphox 中打开当前任务要求的实验。加速度任务推荐“加速度（不含重力）”或“加速度”；其他任务按 PocketLab 当前任务卡显示的传感器要求选择。
3. 进入该实验的菜单，启用“远程访问 / Remote access”，并记录手机显示的完整地址。Android 常见形式为 `http://192.168.x.x:8080`；iOS 通常使用 80 端口，可能不显示端口号。
4. 可先在电脑浏览器中打开该地址：如果能看到 phyphox 远程页面，说明局域网链路正常。
5. 在 PocketLab 打开“设备与设置”，填写设备名称和上述完整地址，保存并执行连接检测。
6. 确认 PocketLab 显示的实验名称、输入能力与当前任务传感器匹配，再开始采集。采集期间保持 phyphox 实验打开，并避免手机锁屏或切换实验。
7. 实验完成后立即在 phyphox 中关闭远程访问。

### 5.3 连接失败与安全边界

- 电脑浏览器也打不开地址：检查两台设备是否在同一网络；校园或公共 Wi-Fi 可能启用客户端隔离，建议换成可信个人热点。
- 浏览器能打开但 PocketLab 提示实验不匹配：在手机切换到任务要求的实验，然后回到 PocketLab 重新检测。
- phyphox 远程接口没有密码和加密，只应在可信局域网中短时启用；不要把手机地址、API Key、含家庭位置的信息或相关截图提交到公开 Issue。
- PocketLab 只接受私有局域网 IP 以及 phyphox 使用的 80/8080 端口，并且不会把手机地址发送给模型。

## 6. 本地自检

```powershell
uv run python scripts/run_public_smoke.py
uv run python scripts/check_git_safety.py --tracked
uv run ruff check .
```

`run_public_smoke.py` 不调用外部模型，只检查应用导入、健康接口、本地认证、两条零等待完整回放和模拟协议创建。
