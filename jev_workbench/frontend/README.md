# JEV Lab 前端工作台

这是一个无构建依赖的静态工作台。页面默认展示实验目录和待运行状态，点击“开始实验”后会向同源后端请求运行，并轮询状态、事件和结果。

## 打开

在项目根目录启动后端：

```bash
python -m jev_workbench.backend.server
```

然后打开 <http://127.0.0.1:8765/>。后端会同时托管本目录的 `index.html`、`styles.css` 和 `app.js`。

也可以直接打开 `index.html` 查看静态界面；此时选择 `Simulation` 并点击开始，前端会在无法连接后端时切换到内置本地模拟，方便检查页面交互。

## API 约定

前端从同源 `/api` 请求以下接口：

- `GET /api/experiments`
- `POST /api/runs`，发送实验 ID、模式、数据集和并发数
- `GET /api/runs/:run_id/state`
- `GET /api/runs/:run_id/events?after=:cursor`
- `GET /api/runs/:run_id/results`
- `POST /api/runs/:run_id/stop`

可以在页面加载前设置 `window.JEV_API_BASE` 指向其他本地服务。前端不读取、不保存、不发送 API key；JEV key 只应由后端环境变量读取。

## 验证

```bash
node --check app.js
node smoke_check.mjs
```

smoke check 会确认关键 DOM、响应式样式和 API 适配器存在。无 Chart.js 时页面使用内置 canvas 柱图降级，浏览器无需额外安装依赖。
