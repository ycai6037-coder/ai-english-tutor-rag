# AI English Coach 智能英语学习伴侣

一个专业的 AI 英语学习平台，具备完整的阅读能力培养、写作能力培养、学习计划与进度管理系统。

![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel)
![License](https://img.shields.io/badge/License-MIT-green)
![DeepSeek](https://img.shields.io/badge/API-DeepSeek-blue)

---

## 功能特性

### 核心能力模块

#### 一、个性化适配系统
- 用户画像收集（水平、目标、年龄、风格、性格）
- 语言风格动态调整（青少年/职场人士/完美主义者/易放弃型）
- AI人格设定（温暖伙伴/严谨教授/幽默朋友/严格教练）

#### 二、阅读能力培养
- 语法解析（时态、语态、从句）
- 词汇深度解析（词义辨析、语境标注、搭配推荐）
- 字词纠错（语法🔴/用词🟡/拼写🟢）
- 阅读理解辅助（长难句解析、理解检测问题）

#### 三、写作能力培养
- 语法纠错（逐句检查）
- 地道表达优化（3种以上替代表达）
- 用词推荐（A1-C2难度等级标注）
- 作文多维度评分（语法/词汇/地道性/连贯性）
- 学习鼓励与成长追踪

#### 四、学习计划与进度管理
- 个性化学习计划制定
- 动态难度调整
- 进度追踪与可视化
- 生词本自动积累与导出

---

## 界面预览

```
┌─────────────────────────────────────────────────────────────┐
│  🚀 AI English Coach                                        │
├──────────┬──────────────────────────────────────────────────┤
│          │  写作批改                    [写作模式] [阅读模式] │
│  ✍️ 写作批改├──────────────────────────────────────────────────┤
│          │                                                  │
│  📖 阅读精析│   输入你的英语作文...                          │
│          │                                                  │
│  📊 学习报告│                                                  │
│          │              [✨ 开始智能批改]                    │
│  ⚙️ 个人画像├──────────────────────────────────────────────────┤
│          │  AI 导师                    [温暖伙伴] [严谨教授] │
│          │  ─────────────────────────────────────────────── │
│  ─────── │  嗨！我来帮你批改作文...                        │
│  用户画像 │                                                  │
│  中级    │  📖 生词本                    本周进度: 65%      │
│  目标:雅思│  ─────────────────────────────────────────────── │
└──────────┴──────────────────────────────────────────────────┘
```

---

## 快速部署

### 第一步：准备 GitHub 仓库

1. 在 GitHub 创建新仓库 `ai-english-coach`
2. 上传项目所有文件

### 第二步：注册 Vercel 并导入

1. 访问 [vercel.com](https://vercel.com)
2. 使用 GitHub 账号登录
3. 点击 **Add New...** → **Project**
4. 选择你的仓库并点击 **Import**

### 第三步：配置环境变量

在 Vercel 项目设置中添加环境变量：

| 变量名 | 值 | 说明 |
|--------|-----|------|
| `API_KEY` | `sk-xxxxxx` | DeepSeek API Key |

**获取 DeepSeek API Key：**
1. 访问 [platform.deepseek.com](https://platform.deepseek.com)
2. 注册并登录
3. 点击 **API Keys** → **Create API Key**
4. 复制生成的 Key

### 第四步：完成部署

点击 **Deploy**，等待部署完成后访问你的域名。

---

## 本地开发

```bash
# 安装 Vercel CLI
npm install -g vercel

# 克隆项目
git clone https://github.com/你的用户名/ai-english-coach.git
cd ai-english-coach

# 创建环境变量文件
cp .env.example .env
# 编辑 .env 填入 API_KEY

# 启动开发服务器
vercel dev

# 访问 http://localhost:3000
```

---

## 项目结构

```
ai-english-coach/
├── package.json          # 项目配置
├── vercel.json           # Vercel 路由配置
├── .env.example          # 环境变量示例
├── .gitignore            # Git 忽略文件
├── README.md             # 项目文档
├── index.html            # 前端页面
└── api/
    └── chat.js           # 后端 API
```

---

## 环境变量说明

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `API_KEY` | ✅ | - | AI API 密钥 |
| `API_BASE_URL` | ❌ | `https://api.deepseek.com/v1` | API 地址 |
| `MODEL_NAME` | ❌ | `deepseek-chat` | 模型名称 |

支持多种 API：
- **DeepSeek** (推荐): `API_BASE_URL=https://api.deepseek.com/v1`
- **Moonshot**: `API_BASE_URL=https://api.moonshot.cn/v1`
- **OpenAI**: `API_BASE_URL=https://api.openai.com/v1`

---

## 使用指南

### 写作批改
1. 点击左侧「写作批改」
2. 在编辑区输入英语作文
3. 点击「开始智能批改」
4. 查看 AI 导师的详细反馈和评分

### 阅读精析
1. 点击左侧「阅读精析」
2. 粘贴想要学习的英语文章
3. 点击「开始阅读精析」
4. 获取语法解析、词汇讲解、理解问题

### 个人画像
1. 点击左侧「个人画像」
2. 填写你的水平、目标、学习风格
3. 保存后 AI 会根据你的情况调整教学内容

### 学习报告
- 查看已批改作文数、累计生词、平均评分
- 追踪学习天数和进度
- 回顾学习历程

---

## 常见问题

**Q: 部署后页面报错？**
A: 检查 Vercel Functions 日志，确认 API_KEY 配置正确。

**Q: 如何更换 AI 模型？**
A: 在 Vercel 环境变量中设置 `API_BASE_URL` 和 `MODEL_NAME`。

**Q: 免费额度够用吗？**
A: Vercel 免费版每月 100GB 流量，DeepSeek 新用户送 500 万 tokens，个人学习完全够用。

---

## 安全说明

- API Key 存储在服务器端，不会暴露给前端
- 所有 API 请求通过后端转发
- 对话历史存储在浏览器 localStorage 中

---

## 技术栈

- **前端**: HTML5 + CSS3 + Vanilla JavaScript
- **Markdown 解析**: Marked.js
- **后端**: Vercel Serverless Functions
- **AI API**: DeepSeek (兼容 OpenAI 格式)

---

## License

MIT License - 自由使用和修改
