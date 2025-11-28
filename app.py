import os

# --- 0. 設定環境變數 (隱藏 TensorFlow/OneDNN 警告) ---
# ⚠️ 注意：這必須寫在所有 import 之前才有效
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import aisuite as ai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login

# --- 1. 頁面設定 ---
st.set_page_config(page_title="國際新聞 AI 編譯", page_icon="📰")
st.title("📰 國際新聞 AI 編譯助手")
st.caption("使用 RAG 技術與 Embedding Gemma 模型")

# --- 2. 處理 Secrets (金鑰) ---
try:
    # 嘗試從 Streamlit secrets 讀取 (本地端讀取 .streamlit/secrets.toml)
    hf_token = st.secrets["HF_TOKEN"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
    
    # 設定環境變數
    os.environ['GROQ_API_KEY'] = groq_api_key
    
    # ⬇️ [關鍵修正] 必須明確設定這個環境變數，下面的 Embedding Class 才能讀取到 Token
    os.environ['HuggingFaceHub_API_TOKEN'] = hf_token 
    
    login(token=hf_token)
except Exception as e:
    st.error("❌ 金鑰未設定！請確認 .streamlit/secrets.toml (本地) 或 Streamlit Cloud Secrets 設定正確。")
    st.stop()

# --- 3. 定義自訂 Embeddings 類別 ---
class EmbeddingGemmaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="google/embeddinggemma-300m",
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={
                # ⬇️ [關鍵修正] 允許執行 Google 的自定義模型程式碼
                "trust_remote_code": True, 
                # ⬇️ [關鍵修正] 明確傳入 Token，解決雲端環境權限問題
                "token": os.environ.get("HuggingFaceHub_API_TOKEN") 
            },
            **kwargs
        )

    def embed_documents(self, texts):
        texts = [f"title: none | text: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"task: search result | query: {text}")

# --- 4. 載入資源 (UI 與邏輯分離版) ---
# ⬇️ [關鍵修正] 使用 show_spinner 讓 Streamlit 自動處理載入動畫，避免 CacheReplayClosureError
@st.cache_resource(show_spinner="🔄 系統初始化中，正在載入模型與資料庫 (初次執行需數分鐘)...")
def load_resources():
    # 這裡面「只留 print」，把所有 st.info, st.empty 全部拿掉
    # 這樣就不會報 CacheReplayClosureError 了

    print("\n" + "="*50)
    print("🚀 系統啟動中...")

    # --- A. 載入 Embedding ---
    print("⏳ Step 1: 正在初始化 Embedding 模型...")
    embedding_model = EmbeddingGemmaEmbeddings()
    print("✅ Step 1: Embedding 模型載入完成！")

    # --- B. 載入 FAISS 資料庫 ---
    print("⏳ Step 2: 正在讀取 FAISS 向量資料庫...")
    if not os.path.exists("faiss_db"):
        # 這裡改用 raise Exception，讓外層去抓錯誤，不要在快取內用 st.error
        raise FileNotFoundError("找不到 faiss_db 資料夾！請確認已將資料夾放入專案目錄。")

    vectorstore = FAISS.load_local(
        "faiss_db",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("✅ Step 2: FAISS 資料庫讀取完成！")

    # --- C. 初始化 LLM Client ---
    print("⏳ Step 3: 初始化 LLM Client...")
    client = ai.Client()
    print("✅ Step 3: LLM Client 準備就緒！")
    print("="*50 + "\n")

    return retriever, client

# 執行載入 (如果卡住，請看終端機)
try:
    retriever, client = load_resources()
except Exception as e:
    st.error(f"❌ 系統載入失敗: {e}")
    st.stop()

# --- 5. 定義 Prompt 與生成邏輯 ---
system_prompt = "你是資深的國際新聞編譯，專門負責將外電資訊整理成台灣讀者容易理解的報導。請保持客觀、專業、精簡的語氣，並使用台灣慣用的翻譯名詞（例如：雪梨而非悉尼、紐西蘭而非新西蘭），並用台灣慣用的中文回應。"

prompt_template = """
請參考下列新聞資料片段：
{retrieved_chunks}

讀者提問：{question}

請根據上述資料撰寫回應：
1. 重點摘要：直接針對問題回答核心事實（人事時地物）。
2. 若上述資料無法完整回答問題，請誠實告知資訊不足，並建議讀者查閱「BBC」、「CNN」或「中央社」等權威媒體以獲取最新消息。

回應內容：
"""

def chat_with_rag(user_input):
    # 1. 檢索
    docs = retriever.invoke(user_input)
    retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])

    # 2. 組合 Prompt
    final_prompt = prompt_template.format(retrieved_chunks=retrieved_chunks, question=user_input)

    # 3. 呼叫 LLM
    # 若 Groq 有更新模型名稱，請在此調整
    model_name = "groq:llama-3.3-70b-versatile" 
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ 呼叫 LLM 時發生錯誤：{str(e)}"

# --- 6. 聊天介面 (UI) ---
if "messages" not in st.session_state:
    welcome_msg = """
    你好！我是國際新聞 AI 翻譯助手。以下是目前的國際新聞重點摘要：

    1. **俄羅斯空襲西烏克蘭**：造成至少 28 人死傷，救援行動持續進行中。
    2. **美國停止生產一分硬幣**：因成本增加，五分硬幣可能成為下一個目標。
    3. **防疫十項建議**：包括設立應急機構、定期演練、簡化結構及改善數據系統。
    4. **戰火下的烏克蘭酒莊**：製造商努力在戰爭中保存傳統。
    5. **《夏洛特的網》政治化爭議**：作者孫女對此表示批評。
    6. **IISS 指控俄羅斯混合戰**：研究指出俄羅斯正進行破壞、間諜等行動以動搖歐洲穩定。

    你可以針對以上新聞做更深入的提問，若資料庫中有相關報導，我將為你詳細解答！
    """
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

# 顯示歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 輸入框
if prompt := st.chat_input("請輸入你想查詢的國際新聞..."):
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 顯示 AI 回應
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在檢索資料並撰寫報導..."):
            response = chat_with_rag(prompt)
            st.markdown(response)
            
    # 儲存回應
    st.session_state.messages.append({"role": "assistant", "content": response})