import os
import io
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


APP_TITLE = "고객 피드백 분석"
DEFAULT_DATA_FILENAME = "@feedback-data.csv"


def find_default_data_file() -> Optional[str]:
    candidate_path = os.path.join(os.getcwd(), DEFAULT_DATA_FILENAME)
    return candidate_path if os.path.exists(candidate_path) else None


def load_dataframe(file_bytes: Optional[bytes], filename_hint: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if file_bytes is None:
        default_path = find_default_data_file()
        if default_path is None:
            return None, None
        try:
            df = pd.read_csv(default_path)
            return df, default_path
        except Exception:
            return None, None
    else:
        try:
            if filename_hint and filename_hint.lower().endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(file_bytes))
            else:
                df = pd.read_csv(io.BytesIO(file_bytes))
            return df, filename_hint or "uploaded"
        except Exception:
            return None, None


def ensure_text_column(df: pd.DataFrame) -> Optional[str]:
    candidate_columns = [
        "text", "review", "feedback", "comment", "message", "내용", "리뷰", "피드백"
    ]
    for col in candidate_columns:
        if col in df.columns:
            return col
    # fallback: try the first object column
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            return col
    return None


def simple_rule_sentiment(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "neutral"
    t = text.lower()
    positive_cues = ["good", "great", "excellent", "love", "satisfied", "만족", "좋다", "최고"]
    negative_cues = ["bad", "terrible", "hate", "poor", "unsatisfied", "불만", "별로", "최악"]
    pos = any(cue in t for cue in positive_cues)
    neg = any(cue in t for cue in negative_cues)
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"


def extract_keywords(texts, max_features: int = 50, ngram_range=(1, 2)) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")
    try:
        X = vectorizer.fit_transform([t if isinstance(t, str) else "" for t in texts])
    except ValueError:
        return pd.DataFrame(columns=["keyword", "score"])  # empty
    scores = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    df_scores = pd.DataFrame({"keyword": terms, "score": scores})
    df_scores.sort_values("score", ascending=False, inplace=True)
    # Scale scores for nicer bars
    if len(df_scores) > 0:
        scaler = MinMaxScaler(feature_range=(0.1, 1.0))
        df_scores["norm_score"] = scaler.fit_transform(df_scores[["score"]])
    else:
        df_scores["norm_score"] = []
    return df_scores


def render_app():
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(APP_TITLE)
    st.caption("피드백 데이터를 업로드하거나 프로젝트 폴더의 @feedback-data.csv 자동 로드")

    with st.sidebar:
        st.header("데이터 업로드")
        uploaded = st.file_uploader("CSV 또는 Excel 업로드", type=["csv", "xlsx"]) 
        sample_hint = st.checkbox("기본 파일 사용 (@feedback-data.csv)", value=True)
        max_kw = st.slider("키워드 수", min_value=10, max_value=100, value=30, step=5)
        ngram = st.select_slider("N-gram 범위", options=["1", "1-2"], value="1-2")
        ngram_range = (1, 1) if ngram == "1" else (1, 2)

    file_bytes = uploaded.read() if uploaded is not None else None
    df, source = load_dataframe(file_bytes if uploaded is not None else (None if sample_hint else None), uploaded.name if uploaded is not None else None)

    if df is None or len(df) == 0:
        st.info("데이터를 업로드하거나 @feedback-data.csv 파일을 프로젝트 폴더에 두세요.")
        return

    text_col = ensure_text_column(df)
    if text_col is None:
        st.error("텍스트 컬럼을 찾지 못했습니다. 'text', 'feedback', '내용' 등의 컬럼명을 사용해주세요.")
        st.dataframe(df.head())
        return

    st.subheader("원본 데이터 미리보기")
    st.write(f"소스: {source}")
    st.dataframe(df.head(20))

    st.subheader("감성 분석")
    with st.spinner("감성 분석 중..."):
        sentiments = df[text_col].astype(str).map(simple_rule_sentiment)
    df_result = df.copy()
    df_result["sentiment"] = sentiments

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**감성 분포**")
        st.bar_chart(df_result["sentiment"].value_counts())
    with col2:
        st.markdown("**샘플 레코드**")
        st.dataframe(df_result[[text_col, "sentiment"]].head(50))

    st.subheader("키워드 추출")
    keywords_df = extract_keywords(df[text_col].astype(str).tolist(), max_features=max_kw, ngram_range=ngram_range)
    if len(keywords_df) == 0:
        st.info("키워드를 찾지 못했습니다. 데이터 양 또는 언어를 확인하세요.")
    else:
        st.bar_chart(keywords_df.set_index("keyword")["norm_score"]) 


if __name__ == "__main__":
    render_app()




