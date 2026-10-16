import io
import os
import re
import unicodedata
import fitz  # PyMuPDF
import ftfy  # 自動修復 Unicode 亂碼
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PDF AI 關鍵字與指數分析工具", page_icon="🤖", layout="wide"
)

st.title("🤖 PDF AI 關鍵字統計與 AI_Index 計算工具")
st.write(
    "採用 **長詞優先 (Longest-term-first)**、**非重疊劃詞 (Spans Check)**、**同義詞正規化** 與 **亂碼品質過濾** 計算 AI_Index。"
)

# 預設 AI 關鍵字
DEFAULT_KEYWORDS = [
    "Artificial Intelligence",
    "machine learning",
    "deep learning",
    "generative AI",
    "大型語言模型",
    "智慧製造",
    "ChatGPT",
    "人工智慧",
    "AI",
]

# ----------------- 側邊欄設定 -----------------
st.sidebar.header("⚙️ 搜尋與指標設定")
source_type = st.sidebar.radio("選擇 PDF 來源：", ["批次上傳 PDF 檔案", "本地資料夾路徑"])

keywords_input = st.sidebar.text_area(
    "關鍵字清單（每行或用逗號分隔）：",
    value="\n".join(DEFAULT_KEYWORDS),
    height=200,
)

context_len = st.sidebar.slider("上下文預覽字數：", 20, 100, 40)
filter_garbled = st.sidebar.checkbox(
    "🛡️ 自動過濾亂碼段落 (過濾 CMap 缺漏防護)", value=True
)


# ----------------- 文字清理、品質檢測與正規化 -----------------
def normalize_text(text):
    """文字預處理與修復亂碼"""
    if not text:
        return ""
    text = ftfy.fix_text(text)  # 修正 Unicode 亂碼
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)

    # 同義詞並列預先取代（例如 人工智慧(AI) -> 人工智慧）
    replacements = [
        (r"人工智慧\s*[\(（]?\s*AI\s*[\)）]?", "人工智慧"),
        (
            r"Artificial Intelligence\s*[\(（]?\s*AI\s*[\)）]?",
            "Artificial Intelligence",
        ),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def is_garbled_context(snippet, threshold=0.12):
    """檢測上下文摘要是否屬於字型 CMap 缺失引起的亂碼段落"""
    if not snippet:
        return False

    # 計算非繁簡中文、基本英數與常見全形標點的異類符號數量
    garbled_chars = [
        ch
        for ch in snippet
        if not (
            "\u4e00" <= ch <= "\u9fff"  # 中文字符
            or "\u0020" <= ch <= "\u007e"  # 基本 ASCII (英數/常見標點)
            or "\u3000" <= ch <= "\u303f"  # 全形標點
            or "\uff00" <= ch <= "\uffef"  # 全形英數與符號
        )
    ]

    ratio = len(garbled_chars) / len(snippet) if len(snippet) > 0 else 0
    return ratio > threshold


def calculate_text_units(text):
    """計算總字數：中文字符數 + 英文單字數"""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"\b[a-zA-Z0-9]+\b", text))
    return chinese_chars + english_words


# ----------------- 逐頁核心演算法 (含亂碼過濾機制) -----------------
def process_page_keywords(
    text, page_num, keywords, file_name, context_len=40
):
    """針對單一頁面進行長詞優先、非重疊劃詞、亂碼過濾，並記錄頁碼"""
    text = normalize_text(text)
    text_lower = text.lower()
    spans = []
    page_counts = {k: 0 for k in keywords}
    page_details = []

    sorted_keywords = sorted(keywords, key=len, reverse=True)

    for keyword in sorted_keywords:
        kw_str = keyword.strip()
        if not kw_str:
            continue

        # 英文/數字關鍵字加上獨立單字邊界防護，防止 Kaisha 誤抓 AI
        if kw_str.isascii():
            pattern = re.compile(
                rf"(?<![a-z0-9]){re.escape(kw_str.lower())}(?![a-z0-9])",
                re.IGNORECASE,
            )
        else:
            pattern = re.compile(re.escape(kw_str), re.IGNORECASE)

        for match in pattern.finditer(text_lower):
            m_start, m_end = match.span()

            # 檢查是否與已命中的長詞重疊
            overlap = any(
                m_start < s_end and m_end > s_start for s_start, s_end in spans
            )

            if not overlap:
                # 擷取上下文摘要
                ctx_start = max(0, m_start - context_len)
                ctx_end = min(len(text), m_end + context_len)
                snippet = text[ctx_start:ctx_end]

                # 亂碼過濾防護：若開啟過濾且摘要被認定為 CMap 缺失亂碼，則忽略該筆命中
                if filter_garbled and is_garbled_context(snippet):
                    continue

                spans.append((m_start, m_end))
                page_counts[kw_str] += 1

                page_details.append(
                    {
                        "檔案名稱": file_name,
                        "頁碼": page_num,
                        "關鍵字": kw_str,
                        "上下文摘要": f"...{snippet}...",
                        "start_idx": m_start,
                    }
                )

    page_details.sort(key=lambda x: x["start_idx"])
    for d in page_details:
        del d["start_idx"]

    return page_counts, page_details, text


# ----------------- PDF 解析與指標計算 -----------------
def analyze_pdf(pdf_stream_or_path, file_name, keywords, context_len=40):
    total_counts = {k: 0 for k in keywords}
    all_details = []
    total_text_units = 0

    try:
        if isinstance(pdf_stream_or_path, str):
            doc = fitz.open(pdf_stream_or_path)
        else:
            doc = fitz.open(stream=pdf_stream_or_path, filetype="pdf")

        for page_index in range(len(doc)):
            page = doc[page_index]
            txt = page.get_text("text")
            if not txt:
                continue

            page_num = page_index + 1
            p_counts, p_details, cleaned_txt = process_page_keywords(
                txt, page_num, keywords, file_name, context_len
            )

            for k, v in p_counts.items():
                total_counts[k] += v
            all_details.extend(p_details)

            total_text_units += calculate_text_units(cleaned_txt)

        doc.close()
    except Exception as e:
        st.error(f"讀取檔案 {file_name} 失敗: {e}")
        return None, []

    total_ai = sum(total_counts.values())
    density = total_ai / total_text_units if total_text_units > 0 else 0
    ai_index = density * 10000

    summary_result = {
        "檔案名稱": file_name,
        "總字數 (Text_Units)": total_text_units,
        "AI 關鍵字總次數 (AI_Total)": total_ai,
        "Density (密度)": density,
        "AI_Index": ai_index,
    }
    summary_result.update(total_counts)

    return summary_result, all_details


# ----------------- Streamlit UI 主邏輯 -----------------
files_to_process = []

if source_type == "批次上傳 PDF 檔案":
    uploaded_files = st.file_uploader(
        "選擇並上傳 PDF 檔案：", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        for uf in uploaded_files:
            files_to_process.append((uf.name, uf.read()))
else:
    folder_path = st.text_input("輸入本地資料夾絕對路徑 (例如 E:\\PDFs)：")
    if folder_path and os.path.exists(folder_path):
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(".pdf"):
                    files_to_process.append((f, os.path.join(root, f)))
    elif folder_path:
        st.warning("⚠️ 找不到該資料夾路徑，請確認輸入是否正確。")

keywords_list = [
    k.strip() for k in re.split(r"[\n,]", keywords_input) if k.strip()
]

if st.button("🚀 開始計算 AI 關鍵字指數 (AI_Index)", type="primary"):
    if not files_to_process:
        st.warning("請先上傳 PDF 檔案或輸入有效的資料夾路徑！")
    elif not keywords_list:
        st.warning("請輸入至少一個關鍵字！")
    else:
        all_summaries = []
        all_details = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (fname, fitem) in enumerate(files_to_process):
            status_text.text(
                f"正在分析 ({idx + 1}/{len(files_to_process)}): {fname}"
            )

            summary_res, details_res = analyze_pdf(
                fitem, fname, keywords_list, context_len
            )
            if summary_res:
                all_summaries.append(summary_res)
                all_details.extend(details_res)

            progress_bar.progress((idx + 1) / len(files_to_process))

        status_text.text("✅ 分析完成！")

        df_summary = pd.DataFrame(all_summaries)
        df_details = pd.DataFrame(all_details)

        st.subheader("📊 AI 指數分析結果摘要")

        df_summary_display = df_summary.copy()
        df_summary_display["Density (密度)"] = df_summary_display[
            "Density (密度)"
        ].map("{:.6f}".format)
        df_summary_display["AI_Index"] = df_summary_display["AI_Index"].map(
            "{:.2f}".format
        )
        st.dataframe(df_summary_display, use_container_width=True)

        tab1, tab2 = st.tabs(["📝 關鍵字比對明細", "📈 各詞出現次數彙整"])

        with tab1:
            if not df_details.empty:
                st.dataframe(df_details, use_container_width=True)
            else:
                st.info("未找到任何關鍵字相符紀錄。")

        with tab2:
            if not df_details.empty:
                kw_freq = (
                    df_details.groupby(["檔案名稱", "關鍵字"])
                    .size()
                    .reset_index(name="次數")
                )
                st.dataframe(kw_freq, use_container_width=True)
            else:
                kw_freq = pd.DataFrame(columns=["檔案名稱", "關鍵字", "次數"])
                st.info("無關鍵字頻率資料。")

        # ----------------- 下載功能區域 -----------------
        st.divider()
        col1, col2 = st.columns(2)

        csv_data = df_summary.to_csv(index=False, encoding="utf-8-sig")
        col1.download_button(
            label="📥 下載指標摘要 CSV",
            data=csv_data,
            file_name="AI_Index_Summary.csv",
            mime="text/csv",
        )

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="AI_Index 指標摘要", index=False)
            if not df_details.empty:
                kw_freq.to_excel(writer, sheet_name="關鍵字頻率統計", index=False)
                df_details.to_excel(
                    writer, sheet_name="關鍵字上下文明細", index=False
                )

        col2.download_button(
            label="📊 下載完整 Excel 報表 (含 AI_Index 與明細)",
            data=excel_buffer.getvalue(),
            file_name="AI_Index_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
