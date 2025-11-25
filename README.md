🌍 國際新聞 AI 編譯助手 (International News RAG Agent)

📑 專案摘要 (Abstract)

本專案旨在開發一套基於**檢索增強生成（Retrieval-Augmented Generation, RAG）**技術的智慧型國際新聞編譯系統，解決傳統大型語言模型（LLM）在處理時事新聞時常見的「幻覺（Hallucination）」與「資訊過時」問題。本系統整合了先進的自然語言處理技術，能夠精準地根據使用者提問，從本地知識庫中檢索相關新聞片段，並扮演「資深國際新聞編譯」的角色，將外電資訊轉化為台灣讀者習慣的繁體中文報導。

🛠️ 技術架構與實作

本系統採用了全開源與高效能的技術堆疊：

語言模型 (LLM)：利用 Groq API 驅動的 Llama-3-70b 模型，實現極低延遲的高速推論，確保即時的使用者體驗。

向量嵌入 (Embeddings)：採用 Google 的 embeddinggemma-300m 模型。為了提升檢索準確度，我們在程式碼中實作了自定義的 Instruction Tuning 類別，為文件與查詢分別加上 task: search result 等特定前綴，優化了語意向量的空間分佈。

向量資料庫 (Vector Store)：使用 FAISS (Facebook AI Similarity Search) 進行高效的相似度搜尋，將新聞文本切塊（Chunking）後建立索引，實現毫秒級的相關文檔檢索。

開發框架：基於 LangChain v0.3+ 架構進行開發，解決了新舊版本相容性問題（如 tf-keras 與 httpx 衝突），並利用 RecursiveCharacterTextSplitter 進行文本處理。

🚀 應用部署

前端介面採用 Streamlit 框架構建，設計了現代化的對話視窗，並具備以下特點：

雲端部署：專案已成功從本地環境（Local）部署至 Streamlit Community Cloud。

效能優化：利用 @st.cache_resource 實作模型快取機制，避免重複載入大型 Embedding 模型，大幅縮短啟動時間。

在地化優化：透過 Prompt Engineering（提示工程），嚴格限制 AI 僅能根據檢索到的事實回答，並強制使用台灣慣用的翻譯名詞（如「紐西蘭」而非「新西蘭」），確保產出內容的專業性與可讀性。

本專案展示了如何整合多種最先進的 AI 工具（State-of-the-Art tools），從資料前處理、模型串接到雲端部署，打造一個完整且實用的垂直領域 AI 應用。

🔑 關鍵字 (Keywords)

RAG, LangChain, Llama-3, Groq API, FAISS, Streamlit, Embedding Gemma, Python