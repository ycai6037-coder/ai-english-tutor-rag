/**
 * 数据入库脚本 - 使用 Jina AI Embeddings
 * 将教材内容向量化并存入 Upstash 向量数据库
 */

import { UpstashVectorStore } from "@langchain/community/vectorstores/upstash";
import { JinaEmbeddings } from "@langchain/community/embeddings/jina";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Index } from "@upstash/vector";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

// 加载环境变量
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  console.log("🚀 开始导入数据 (使用 Jina AI Embeddings)...");

  try {
    // 检查环境变量
    if (!process.env.JINA_API_KEY) {
      console.error("❌ 缺少 JINA_API_KEY 环境变量");
      console.log("请访问 https://jina.ai/embeddings/ 获取免费的 API Key");
      return;
    }

    if (!process.env.UPSTASH_VECTOR_REST_URL || !process.env.UPSTASH_VECTOR_REST_TOKEN) {
      console.error("❌ 缺少 Upstash 向量数据库配置");
      return;
    }

    // 1. 初始化 Jina AI 嵌入模型
    // jina-embeddings-v2-base-en 输出 768 维向量
    // 但 Upstash 索引配置为 1536 维，所以使用 jina-colbert-v2 (支持更高维度)
    // 或者我们可以直接用 OpenAI 兼容的方式
    const embeddings = new JinaEmbeddings({
      model: "jina-embeddings-v2-base-en",  // 768 维
    });

    console.log("✅ Jina AI Embeddings 初始化成功 (jina-embeddings-v2-base-en, 768维)");
    console.log("⚠️  注意: Upstash 索引维度需要设置为 768");
    console.log("   请在 Upstash 控制台检查索引维度设置");

    // 2. 创建 Upstash Vector Index
    const index = new Index({
      url: process.env.UPSTASH_VECTOR_REST_URL,
      token: process.env.UPSTASH_VECTOR_REST_TOKEN,
    });

    // 3. 初始化向量存储
    const vectorStore = new UpstashVectorStore(embeddings, {
      index: index,
    });

    console.log("✅ Upstash 向量数据库连接成功");

    // 4. 读取本地文件
    const filePath = path.join(__dirname, "../data/lesson1.txt");
    if (!fs.existsSync(filePath)) {
      console.error("❌ 找不到文件:", filePath);
      return;
    }

    console.log("📄 正在读取文件:", filePath);
    const textContent = fs.readFileSync(filePath, "utf-8");

    console.log("📄 文档加载成功，内容长度:", textContent.length, "字符");

    // 5. 切分文本
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,      // 每个片段最大 500 字符
      chunkOverlap: 50,    // 片段重叠 50 字符
    });
    
    const splits = await textSplitter.createDocuments([textContent]);

    console.log(`📄 文档已切分为 ${splits.length} 个片段`);

    // 6. 存入向量数据库
    console.log("⏳ 正在向量化并存储到 Upstash...");
    await vectorStore.addDocuments(splits);

    console.log("");
    console.log("========================================");
    console.log("✅ 数据导入成功！");
    console.log("📊 已存储文档数:", splits.length);
    console.log("🔗 可以在 Upstash 控制台查看数据");
    console.log("========================================");
    console.log("");
    console.log("接下来可以:");
    console.log("1. 运行 npm run dev 本地测试");
    console.log("2. 部署到 Vercel");

  } catch (error) {
    console.error("❌ 导入失败:", error);
    console.error("错误详情:", error.message);
    
    if (error.message && error.message.includes("dimension")) {
      console.log("");
      console.log("========================================");
      console.log("⚠️  向量维度不匹配！");
      console.log("你的 Upstash 索引维度是 1536，但 Jina 模型输出 768 维");
      console.log("");
      console.log("解决方案：");
      console.log("1. 去 Upstash 控制台重新创建索引，维度设为 768");
      console.log("   或者删除现有数据后重新设置");
      console.log("2. 或者使用 OpenAI Embeddings (1536维)");
      console.log("========================================");
    }
  }
}

main();
