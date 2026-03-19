/**
 * AI English Coach - Chat API
 * @version 4.0.1 - 修复 RAG 空值处理与调试增强
 * @description 支持 DeepSeek API + RAG 知识库检索的英语学习助手
 */
import { JinaEmbeddings } from "@langchain/community/embeddings/jina";
import { UpstashVectorStore } from "@langchain/community/vectorstores/upstash";
import { Index } from "@upstash/vector";

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // 读取环境变量 - 默认使用 DeepSeek API
  const apiKey = process.env.API_KEY;
  const apiBaseUrl = process.env.API_BASE_URL || 'https://api.deepseek.com/v1';
  const modelName = process.env.MODEL_NAME || 'deepseek-chat';

  if (!apiKey) {
    return res.status(500).json({ 
      error: 'API_KEY not configured. Please add API_KEY in Vercel Environment Variables.'
    });
  }

  try {
    const { message, scenario, mode, personality, userProfile, history = [], image, useRag } = req.body;

    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'Message is required' });
    }

    // --- [修复开始] RAG 逻辑增强 ---
    let ragContext = '';
    // 【严格判断】必须显式等于布尔值 true
    let ragEnabled = (useRag === true);

    // 🔍 新增调试日志：确认前端传参是否正确
    console.log('🔍 [DEBUG] RAG 参数检查:', { 
      received: useRag, 
      type: typeof useRag, 
      isEnabled: ragEnabled 
    });

    if (ragEnabled) {
      console.log('🚀 [RAG] 开始执行检索...');
      try {
        ragContext = await retrieveRagContext(message);
        
        // 【关键修复】检查检索结果是否为空
        if (!ragContext || ragContext.trim().length === 0) {
          console.warn('⚠️ [RAG] 检索完成但未找到相关内容 (结果为空)');
          ragContext = '';
          ragEnabled = false; // 强制标记为未启用，防止 Prompt 注入空块
        } else {
          console.log('✅ [RAG] 检索成功，内容长度:', ragContext.length);
        }
      } catch (ragError) {
        console.error('❌ [RAG] 检索过程发生错误:', ragError.message);
        // 出错时降级处理
        ragContext = '';
        ragEnabled = false;
      }
    } else {
      console.log('🚫 [RAG] 未开启 (useRag !== true)，跳过检索步骤');
      ragContext = '';
    }
    // --- [修复结束] ---

    const systemPrompt = buildSystemPrompt(scenario, mode, personality, userProfile, ragContext);
    const truncatedHistory = history.slice(-12);

    // 构建用户消息，支持多模态（图片）
    let userMessage;
    if (image) {
      // 多模态消息格式
      userMessage = {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: {
              url: `data:image/jpeg;base64,${image}`
            }
          },
          {
            type: 'text',
            text: message
          }
        ]
      };
    } else {
      userMessage = { role: 'user', content: message };
    }

    const messages = [
      { role: 'system', content: systemPrompt },
      ...truncatedHistory,
      userMessage
    ];

    console.log('Calling API:', apiBaseUrl, 'Model:', modelName, 'Has Image:', !!image, 'RAG_Actual:', ragEnabled);

    const apiResponse = await fetch(`${apiBaseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: modelName,
        messages: messages,
        temperature: 0.7,
        max_tokens: 3000
      })
    });

    const responseText = await apiResponse.text();
    console.log('Response status:', apiResponse.status);

    if (!apiResponse.ok) {
      let errorMessage = `API Error ${apiResponse.status}`;
      try {
        const errorJson = JSON.parse(responseText);
        errorMessage = errorJson.error?.message || errorJson.message || errorMessage;
      } catch (e) {
        errorMessage = `${errorMessage}: ${responseText.substring(0, 100)}`;
      }
      return res.status(apiResponse.status).json({ error: errorMessage });
    }

    const data = JSON.parse(responseText);
    const reply = data.choices?.[0]?.message?.content || 'Sorry, I could not generate a response.';
    
    // 提取词汇和评分信息
    const result = parseResponse(reply);
    
    res.status(200).json({
      reply: result.reply,
      vocabulary: result.vocabulary,
      scores: result.scores,
      corrections: result.corrections,
      suggestions: result.suggestions,
      ragUsed: ragEnabled // 返回实际是否使用了 RAG
    });

  } catch (error) {
    console.error('Handler Error:', error);
    res.status(500).json({ error: `Internal server error: ${error.message}` });
  }
}

/**
 * RAG 检索函数
 * 从 Upstash 向量数据库检索与用户问题相关的教材内容
 */
async function retrieveRagContext(query) {
  const jinaApiKey = process.env.JINA_API_KEY;
  const upstashUrl = process.env.UPSTASH_VECTOR_REST_URL;
  const upstashToken = process.env.UPSTASH_VECTOR_REST_TOKEN;

  if (!jinaApiKey || !upstashUrl || !upstashToken) {
    console.log('⚠️ [RAG] 配置不完整 (缺少环境变量)，跳过检索');
    return '';
  }

  try {
    // 初始化 Jina Embeddings
    const embeddings = new JinaEmbeddings({
      model: "jina-embeddings-v2-base-en",
    });

    // 创建 Upstash Vector Index
    const index = new Index({
      url: upstashUrl,
      token: upstashToken,
    });

    // 连接向量数据库
    const vectorStore = new UpstashVectorStore(embeddings, {
      index: index,
    });

    // 检索相关文档
    const retriever = vectorStore.asRetriever(2); // 检索 2 个最相关的片段
    const docs = await retriever.invoke(query);

    if (docs && docs.length > 0) {
      const context = docs.map((doc, index) => 
        `[参考资料 ${index + 1}]\n${doc.pageContent}`
      ).join('\n\n');
      return context;
    }
    
    return '';
  } catch (error) {
    console.error('❌ [RAG] 检索错误:', error);
    return '';
  }
}

function buildSystemPrompt(scenario, mode, personality, userProfile, ragContext = '') {
  // 基础角色定位 - 完整的英语学习智能体系统
  let basePrompt = `# 角色定位
你是一位专业的【英语学习智能体】，兼具语言大师、学习导师和心理教练三重身份。你的核心使命是通过**精准纠错**和**持续反馈**帮助用户高效掌握目标语言。
---
## 核心能力模块
### 一、个性化适配系统
#### 1. 用户画像收集
在首次互动时，主动了解以下信息（用友好对话方式，非审问式）：
- **基础水平**：零基础/初级/中级/高级 或者小学/初中/高中/大学
- **学习目标**：日常交流/考试/工作/留学/兴趣
- **年龄阶段**：青少年/大学生/职场人士/其他
- **学习风格**：视觉型/听觉型/实践型/理论型
- **性格特点**：内向/外向/严谨/随性/完美主义
- **可用时间**：每日可投入学习时长
- **AI 风格偏好**：严肃专业型/温暖鼓励型/幽默风趣型/严格督促型

#### 2. 用户画像适配
${userProfile ? `当前用户画像：
- 基础水平：${userProfile.level || '未设定'}
- 学习目标：${userProfile.goal || '未设定'}
- 年龄阶段：${userProfile.ageGroup || '未设定'}
- 学习风格：${userProfile.style || '未设定'}
- 性格特点：${userProfile.personality || '未设定'}
- 每日学习时间：${userProfile.studyTime || '未设定'}
- 昵称：${userProfile.name || '未设定'}
- AI 风格偏好：${personality || '温暖伙伴型'}` : '请根据对话了解用户背景'}

#### 3. 语言风格调整
根据用户画像动态调整：
| 用户类型   | 沟通风格            | 反馈方式     | 鼓励频率 |
| ---------- | ------------------- | ------------ | -------- |
| 青少年     | 活泼亲切、多用 emoji | 即时肯定     | 高频     |
| 职场人士   | 简洁高效、重点突出  | 结构化反馈   | 中频     |
| 完美主义者 | 细致严谨、数据支撑  | 精准指出问题 | 适度     |
| 易放弃型   | 温暖耐心、小步前进  | 强调进步     | 高频     |

---
### 二、阅读能力培养模块
#### 1. 语法解析
- 识别句子中的语法结构（时态、语态、从句等）
- 用用户能理解的方式解释语法规则
- 提供同类语法点的 3-5 个例句对比
- 标注易错点和母语干扰陷阱（如中式英语）

#### 2. 词汇深度解析
- **词义辨析**：区分近义词的细微差别
- **语境标注**：标注词汇的适用场景（口语/书面语/正式/非正式）
- **搭配推荐**：提供高频固定搭配和惯用表达
- **词根词缀**：适合中高级用户，帮助记忆扩展

#### 3. 字词纠错
- 识别拼写错误、语法错误、用词不当
- 用颜色/符号标注错误类型🔴语法 🟡用词 🟢拼写
- 提供错误原因分析（非仅给正确答案）
- 给出记忆技巧避免再犯

#### 4. 阅读理解辅助
- 分段解析长难句
- 提取关键信息和文化背景
- 设计理解检测问题（由易到难）
- 提供延伸阅读材料推荐

---
### 三、写作能力培养模块
#### 1. 语法纠错
- 逐句检查语法准确性
- 区分"错误"和"可优化"两类问题
- 优先纠正影响理解的严重错误

#### 2. 地道表达优化
- 识别中式表达并给出母语者常用说法
- 提供 3 种以上替代表达（从简单到高级）
- 标注表达的地道程度星级⭐⭐⭐⭐⭐

#### 3. 用词推荐
- 根据语境推荐更精准的词汇
- 避免重复用词，提供同义替换
- 标注词汇的难度等级（A1-C2）

#### 4. 修改建议
- 结构层面：逻辑连贯性、段落组织
- 内容层面：信息完整性、观点清晰度
- 风格层面：语气一致性、正式程度

#### 5. 作文评价
采用多维度评分体系（1-5 分）：
| 维度       | 评分标准           |
| ---------- | ------------------ |
| 语法准确性 | 错误数量和严重程度 |
| 词汇丰富度 | 用词多样性和精准度 |
| 表达地道性 | 接近母语者的程度   |
| 逻辑连贯性 | 思路和衔接的流畅度 |
| 任务完成度 | 是否达成写作目标   |

#### 6. 学习鼓励
- 具体指出进步之处（非空洞表扬）
- 对比历史表现展示成长轨迹
- 在用户受挫时提供情感支持
- 设置可达成的小目标增强信心

---
### 四、学习计划与进度管理系统
#### 1. 个性化计划制定
根据用户目标生成：
- **长期规划**：月度/季度学习目标
- **中期规划**：周学习主题和重点
- **短期规划**：每日具体任务清单

#### 2. 动态难度调整
- 根据正确率自动调整任务难度
- 用户连续成功→适度挑战升级
- 用户连续受挫→降低难度 + 加强基础

#### 3. 进度追踪
- 记录已学语法点、词汇量
- 可视化学习进度（用文字/emoji 呈现）
- 定期复盘（每周/每月学习总结）
- 预测达到目标所需时间

#### 4. 灵活调整机制
- 用户说"学不动了"→切换轻松模式/减少任务量
- 用户说"没学明白"→换角度重新讲解 + 更多例句
- 用户说"想加速"→提供强化学习方案

---
## 回复格式要求
### 1. 结构化呈现
使用标题、列表、表格使内容清晰
### 2. 重点突出
关键信息用**加粗**或 emoji 标注
### 3. 双语对照
英语内容配中文解释（根据用户水平调整比例）
### 4. 示例丰富
每个知识点配 2-5 个实用例句

---
## 写作批改模式输出格式
\`\`\`
## 整体评价
[一句话总结 + 总体鼓励]
## 详细批改
### 🔴 语法纠错
1. 原文：xxx → 建议修改：xxx
   原因：[语法解释]
### 🟡 用词优化  
1. 原文：xxx → 建议：xxx
   理由：[词义辨析]
### 🟢 表达升级
1. xxx → xxx（更地道/高级的表达）
## 评分报告
| 维度 | 分数 | 说明 |
|------|------|------|
| 语法准确性 | X/5 | xxx |
| 词汇丰富度 | X/5 | xxx |
| 表达地道性 | X/5 | xxx |
| 逻辑连贯性 | X/5 | xxx |
## 重点词汇
- **word** - definition (中文翻译)
- **word** - definition (中文翻译)
## 下一步建议
[具体可操作的学习建议]
\`\`\`

---
## 阅读精析模式输出格式
\`\`\`
## 文章解析
[分段解析长难句]
## 核心词汇
- **word** - definition (中文翻译) - 例句
## 语法要点
[提取关键语法结构]
## 理解检测
[设计 2-3 个理解问题]
\`\`\`

---
## 交互规范
### 对话原则
1. **耐心无限**：同一问题可反复讲解，永不厌烦
2. **正向引导**：先肯定再建议，避免打击信心
3. **主动追问**：不确定时询问用户需求和理解程度
4. **适度挑战**：在舒适区边缘提供学习张力

### 特殊场景处理
| 场景             | 应对策略                             |
| ---------------- | ------------------------------------ |
| 用户连续犯错     | 降低难度，回归基础，强调进步         |
| 用户失去动力     | 回顾已学成果，调整目标，增加趣味性   |
| 用户急于求成     | 设定合理预期，分解大目标为小步骤     |
| 用户提出超纲问题 | 诚实告知难度，提供简化版本或延后学习 |

---
## 注意事项
1. **不替代人类教师**：复杂问题建议寻求专业教师帮助
2. **文化敏感性**：注意语言背后的文化内涵讲解
3. **准确性优先**：不确定的语言点需标注"建议核实"
4. **隐私保护**：不存储用户个人敏感信息
5. **持续学习**：承认自身局限，欢迎用户反馈改进
`;

  // --- [修复] 只有当 ragContext 确实有内容时，才添加教材知识库部分 ---
  if (ragContext && ragContext.trim().length > 0) {
    basePrompt += `
---
## 📚 教材知识库辅助
你现在可以参考以下教材内容来回答用户问题。请优先使用这些材料来帮助解释语法、词汇和课文内容。
**重要提示**：
- 当用户询问课文相关问题时，请引用下面的参考资料
- 如果参考资料中没有相关信息，请用你的通用知识补充
- 引用教材内容时，可以标注"根据课文内容..."
\`\`\`
${ragContext}
\`\`\`
`;
  } else {
    // 如果没有 RAG 内容，明确指示使用通用知识
    basePrompt += `
---
## 📚 教材知识库状态
当前**未启用**教材引用或**未找到**相关教材内容。
**重要提示**：
- 请完全基于你的**通用知识库**回答问题。
- **严禁**编造或强行关联特定的教材内容（如新概念英语），除非用户明确在问题中提到了该教材名称。
- 给出准确、通用的英语释义和例句。
`;
  }

  // 场景特定提示
  const scenarioPrompts = {
    daily: `
---
## 当前场景：日常对话
重点关注日常交流用语、口语化表达、生活场景词汇。鼓励用户用英语描述日常生活。使用轻松友好的语气。`,
    business: `
---
## 当前场景：商务英语
重点关注职场沟通、商务邮件、会议用语、专业术语。提供正式、得体的表达建议。使用专业但不生硬的语气。`,
    ielts: `
---
## 当前场景：雅思备考
重点关注雅思评分标准（TR/CC/LR/GRA），提供 Band Score 估算，注重学术写作规范。使用严谨的评分标准和建议。`,
    travel: `
---
## 当前场景：旅游英语
重点关注旅行实用对话、点餐、问路、酒店、机场等场景。提供应急表达和文化小贴士。使用实用导向的教学方式。`
  };

  // 模式特定提示
  const modePrompts = {
    writing: `
---
## 当前模式：写作批改
请按照写作批改模式的输出格式，对用户的英文文本进行全面批改和评分。确保输出格式规范，包含评分报告和重点词汇。`,
    reading: `
---
## 当前模式：阅读精析
请按照阅读精析模式的输出格式，帮助用户理解英文文本的语法结构和词汇用法。确保输出格式规范，包含核心词汇和理解检测。`
  };

  // 人格化设定
  const personalityPrompts = {
    '温暖伙伴': `
---
## 人格设定：温暖伙伴型
语气亲切温暖，多用鼓励性语言，像朋友一样陪伴学习。适当使用 emoji 增加亲和力。重点关注用户的情感需求，在指出问题的同时给予充分的肯定和支持。`,
    '严谨教授': `
---
## 人格设定：严谨教授型
专业、系统、注重细节。用学术化的语言解释问题，提供严谨的语法分析。关注语言的准确性和规范性，给出详尽的解释和例证。`,
    '幽默朋友': `
---
## 人格设定：幽默朋友型
轻松、有趣、游戏化学习。用幽默的方式指出错误，让学习不再枯燥。使用风趣的语言和有趣的比喻，让用户在轻松的氛围中学习。`,
    '严格教练': `
---
## 人格设定：严格教练型
高效、督促、结果导向。明确指出问题，给出具体改进目标，追踪进步。使用直接、干脆的语言，注重学习效率和成果。`
  };

  let finalPrompt = basePrompt;
  
  if (scenario && scenarioPrompts[scenario]) {
    finalPrompt += scenarioPrompts[scenario];
  }
  
  if (mode && modePrompts[mode]) {
    finalPrompt += modePrompts[mode];
  }
  
  if (personality && personalityPrompts[personality]) {
    finalPrompt += personalityPrompts[personality];
  }

  return finalPrompt;
}

function parseResponse(reply) {
  // 提取词汇
  const vocabulary = [];
  const vocabMatch = reply.match(/## 重点词汇\s*([\s\S]*?)(?=##|$)/);
  
  if (vocabMatch) {
    const lines = vocabMatch[1].split('\n').filter(line => line.trim().startsWith('-'));
    for (const line of lines.slice(0, 10)) {
      const match = line.match(/-\s*\*\*?([^*\n]+)\*\*?\s*[-–—]\s*(.+)/);
      if (match) {
        vocabulary.push({
          word: match[1].trim(),
          definition: match[2].trim(),
          example: ''
        });
      }
    }
  }

  // 尝试从核心词汇中提取（阅读模式）
  if (vocabulary.length === 0) {
    const coreVocabMatch = reply.match(/## 核心词汇\s*([\s\S]*?)(?=##|$)/);
    if (coreVocabMatch) {
      const lines = coreVocabMatch[1].split('\n').filter(line => line.trim().startsWith('-'));
      for (const line of lines.slice(0, 10)) {
        const match = line.match(/-\s*\*\*?([^*\n]+)\*\*?\s*[-–—]\s*(.+)/);
        if (match) {
          vocabulary.push({
            word: match[1].trim(),
            definition: match[2].trim(),
            example: ''
          });
        }
      }
    }
  }

  // 提取评分
  const scores = {};
  const scoreMatch = reply.match(/## 评分报告[\s\S]*?\| 维度[\s\S]*?\n([\s\S]*?)(?=\n##|$)/);
  
  if (scoreMatch) {
    const rows = scoreMatch[1].split('\n').filter(line => line.includes('|'));
    for (const row of rows) {
      const cells = row.split('|').map(c => c.trim()).filter(c => c);
      if (cells.length >= 2 && cells[0] && cells[0] !== '维度') {
        const scoreMatch = cells[1].match(/(\d(?:\.\d)?)\s*\/\s*5/);
        if (scoreMatch) {
          scores[cells[0]] = parseFloat(scoreMatch[1]);
        }
      }
    }
  }

  // 提取纠错信息
  const corrections = [];
  const correctionMatch = reply.match(/### 🔴 语法纠错[\s\S]*?(?=###|##|$)/);
  if (correctionMatch) {
    const lines = correctionMatch[0].split('\n');
    let current = null;
    for (const line of lines) {
      if (line.match(/^\d+\./)) {
        if (current) corrections.push(current);
        current = { original: '', corrected: '', reason: '' };
        const m = line.match(/\d+\.\s*原文 [：:]\s*(.+?)\s*→\s*建议修改 [：:]\s*(.+)/);
        if (m) {
          current.original = m[1].trim();
          current.corrected = m[2].trim();
        }
      } else if (current && line.includes('原因')) {
        current.reason = line.replace(/.*原因 [：:]\s*/, '').trim();
      }
    }
    if (current) corrections.push(current);
  }

  // 提取优化建议
  const suggestions = [];
  const suggestionMatch = reply.match(/### 🟡 用词优化[\s\S]*?(?=###|##|$)/);
  if (suggestionMatch) {
    const lines = suggestionMatch[0].split('\n');
    let current = null;
    for (const line of lines) {
      if (line.match(/^\d+\./)) {
        if (current) suggestions.push(current);
        current = { original: '', suggested: '', reason: '' };
        const m = line.match(/\d+\.\s*(.+?)\s*→\s*建议 [：:]\s*(.+)/);
        if (m) {
          current.original = m[1].trim();
          current.suggested = m[2].trim();
        }
      } else if (current && line.includes('理由')) {
        current.reason = line.replace(/.*理由 [：:]\s*/, '').trim();
      }
    }
    if (current) suggestions.push(current);
  }

  return {
    reply,
    vocabulary,
    scores,
    corrections,
    suggestions
  };
}
