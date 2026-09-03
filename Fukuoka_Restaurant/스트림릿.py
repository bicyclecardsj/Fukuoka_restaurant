import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse # 구글 맵 검색어 URL 인코딩을 위한 라이브러리

# 1. 페이지 설정
st.set_page_config(page_title="후쿠오카 음식점 선택 도우미", layout="wide")

st.title("후쿠오카 음식점 1:1 비교")
st.markdown("### AI리뷰 분석기반 음식점 비교 서비스")
st.markdown("---")

# 2. 데이터 로드 및 초기 필터링
@st.cache_data
def load_master_and_reviews():
    df_stores = pd.read_csv("./data/가게리스트_평점.csv", encoding="utf-8-sig")
    
    # 🚫 '동물카페' 카테고리는 맛집 결투장 콘셉트와 맞지 않으므로 데이터 로드 시점에 원천 배제
    df_stores = df_stores[df_stores['category'] != '동물카페']
    
    # 최종 리뷰 파일 로드 (플레이스명, 텍스트, 연도, AI 라벨, 확신도 확률)
    review_cols = ['place_name', 'text', 'year', 'ai_label', 'ai_prob']
    df_reviews = pd.read_csv("./data/후쿠오카_리뷰_최종.csv", usecols=review_cols, encoding="utf-8-sig")
    
    return df_stores, df_reviews

try:
    df_stores, df_reviews = load_master_and_reviews()
except FileNotFoundError as e:
    st.error(f"📂 데이터를 불러오는 중 오류가 발생했습니다. 파일명과 경로를 확인해 주세요! ({e})")
    st.stop()

# ==========================================
# 🔍 상단 메인 필터 및 검색창 배치
# ==========================================
st.markdown("### 🗂️ 카테고리 및 매장 검색/선택")

cat_counts = df_stores['category'].value_counts()
valid_categories = cat_counts[cat_counts >= 2].index.tolist()
category_list = sorted(valid_categories)

if not category_list:
    st.error("⚠️ 비교 가능한 카테고리(매장 2개 이상 등록됨)가 존재하지 않습니다.")
    st.stop()

selected_category = st.selectbox("🎯 먼저 음식 카테고리를 선택하세요", category_list, index=0)

filtered_stores = df_stores[df_stores['category'] == selected_category]
store_list = sorted(filtered_stores['place_name'].unique())

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    store_A = st.selectbox(
        "🏪 후보 A 매장 선택 (클릭 후 타이핑하면 검색 가능)", 
        store_list, 
        index=0,
        key="store_select_A"
    )

with col_sel2:
    default_b_index = min(1, len(store_list) - 1)
    store_B = st.selectbox(
        "🏪 후보 B 매장 선택 (클릭 후 타이핑하면 검색 가능)", 
        store_list, 
        index=default_b_index,
        key="store_select_B"
    )

if store_A == store_B:
    st.warning("⚠️ 똑같은 매장을 선택하셨습니다! 서로 다른 두 가게를 골라주세요.")
    st.stop()

st.markdown("---")

google_rating_A = filtered_stores[filtered_stores['place_name'] == store_A]['google_rating'].values[0]
google_rating_B = filtered_stores[filtered_stores['place_name'] == store_B]['google_rating'].values[0]

df_A = df_reviews[df_reviews['place_name'] == store_A]
df_B = df_reviews[df_reviews['place_name'] == store_B]


# 4. 백엔드 알고리즘: 통계 지표 및 표본 오차 계산
def calculate_metrics(store_df, official_rating):
    total = len(store_df)
    if total == 0:
        return 0, 0, official_rating, 0, pd.DataFrame()
    
    pos_df = store_df[store_df['ai_label'].astype(str).str.contains('긍정|1')]
    neg_df = store_df[store_df['ai_label'].astype(str).str.contains('부정|0')]
    
    pos_ratio = (len(pos_df) / total) * 100
    margin_of_error = (1 / np.sqrt(total)) * 100
    worst_review = neg_df.sort_values(by='ai_prob', ascending=False)
    
    return total, pos_ratio, official_rating, margin_of_error, worst_review

total_A, ratio_A, google_A, error_A, worst_df_A = calculate_metrics(df_A, google_rating_A)
total_B, ratio_B, google_B, error_B, worst_df_B = calculate_metrics(df_B, google_rating_B)

def get_density_badge(count):
    if count >= 500: return "🟢 AI 판정 신뢰도: 상 (리뷰가 많아 신뢰도 높음)"
    elif count >= 100: return "🟡 AI 판정 신뢰도: 중 (참고 가능한 수준)"
    else: return "🔴 AI 판정 신뢰도: 하 (표본 부족 / 변동성 주의)"


# ==========================================
# 📊 SECTION 1. 매장 체급 및 긍정 지분율 대조
# ==========================================
st.subheader("평점 및 긍정 리뷰 비율 비교")

col1, col_vs, col2 = st.columns([4, 1, 4])

with col1:
    st.subheader(f"후보 A: {store_A}")
    st.caption(get_density_badge(total_A))
    st.write(f"💬 수집된 총 리뷰 수: **{total_A}건**")
    st.metric(label="⭐ 구글 맵 평점", value=f"{google_A:.1f} / 5.0")
    
    search_keyword_A = urllib.parse.quote(f"후쿠오카 {store_A}")
    map_url_A = f"https://www.google.com/maps/search/?api=1&query={search_keyword_A}"
    st.link_button("📍 구글 지도에서 위치 보기", map_url_A)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**분석 리뷰중 긍정 리뷰 비율**")
    st.progress(int(ratio_A))
    st.markdown(f"## {ratio_A:.1f}% <span style='font-size:16px; color:gray;'>(±{error_A:.1f}%p 오차범위)</span>", unsafe_allow_html=True)

with col_vs:
    st.markdown("<h1 style='text-align: center; color: red; margin-top: 50px;'>VS</h1>", unsafe_allow_html=True)

with col2:
    st.subheader(f"후보 B: {store_B}")
    st.caption(get_density_badge(total_B))
    st.write(f"💬 수집된 총 리뷰 수: **{total_B}건**")
    st.metric(label="⭐ 구글 맵 평점", value=f"{google_B:.1f} / 5.0")
    
    search_keyword_B = urllib.parse.quote(f"후쿠오카 {store_B}")
    map_url_B = f"https://www.google.com/maps/search/?api=1&query={search_keyword_B}"
    st.link_button("📍 구글 지도에서 위치 보기", map_url_B)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**분석 리뷰중 긍정 리뷰 비율**")
    st.progress(int(ratio_B))
    st.markdown(f"## {ratio_B:.1f}% <span style='font-size:16px; color:gray;'>(±{error_B:.1f}%p 오차범위)</span>", unsafe_allow_html=True)

st.markdown("---")


# ==========================================
# 📈 SECTION 2. 시간축 트렌드 차트 (직관적인 범례 개선 버전)
# ==========================================
st.subheader("연도별 긍정/부정 추이")
st.markdown("하단 버튼을 통해 긍정/부정 트렌드를 교차 조회할 수 있습니다.")

# 💡 긍부정 토글 라디오 버튼 배치
view_mode = st.radio(
    "조회할 긍정/부정 추이를 선택하세요",
    ["긍정 리뷰 추이 보기", "부정 리뷰 추이 보기"],
    horizontal=True
)

# 데이터 바인딩 연산
df_combined = pd.concat([df_A, df_B])
df_combined['is_pos'] = df_combined['ai_label'].astype(str).str.contains('긍정|1').astype(int)

# 연도별/가게별 표본 수와 평균 긍정률 계산
trend_raw = df_combined.groupby(['year', 'place_name'])['is_pos'].agg(['count', 'mean'])

# [방어 로직] 5건 미만 표본 연도 제거
min_reviews_per_year = 5
trend_clean = trend_raw[trend_raw['count'] >= min_reviews_per_year]

if not trend_clean.empty:
    # 데이터 재구조화
    trend_table_df = (trend_clean['mean'] * 100).unstack()
    trend_table_df = trend_table_df.dropna(subset=[store_A, store_B]) # 두 가게 공통 분기점 필터링
    
    if not trend_table_df.empty:
        # 토글 버튼 선택에 따른 데이터 반전 처리 (부정 선택 시 100 - 긍정률)
        if "부정" in view_mode:
            chart_data = 100 - trend_table_df
            title_suffix = "부정 리뷰 비율 추이 (%)"
        else:
            chart_data = trend_table_df
            title_suffix = "긍정 리뷰 비율 추이 (%)"
            
        # 정렬 및 시계열 축 지정
        chart_data = chart_data.sort_index(ascending=True) # 시간 순서대로 정렬
        
        # 💡 [핵심 개선] 차트 범례와 선에 매장 이름과 후보 구분을 명확하게 주입
        rename_cols = {
            store_A: f"후보 A",
            store_B: f"후보 B"
        }
        chart_data = chart_data.rename(columns=rename_cols)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 차트 바로 위에 컬러 매칭 가이드 배치
        st.markdown(
            f"### 🔵 {rename_cols[store_A]} &nbsp;&nbsp;&nbsp;&nbsp; 🟠 {rename_cols[store_B]}"
        )
        
        # 💡 [색상 고정] 스트림릿 차트에 들어갈 두 선의 색상을 hex 코드로 강제 지정합니다.
        # 첫 번째 컬럼(A매장)은 파란색(#1F77B4), 두 번째 컬럼(B매장)은 주황색(#FF7F0E)
        chart_colors = ["#1F77B4", "#FF7F0E"]
        
        # 📈 선 색상이 강제 지정된 스트림릿 라인 차트 출력
        st.markdown(f"##### 📉 연도별 {title_suffix}")
        st.line_chart(
            chart_data, 
            color=chart_colors  # 💡 이 옵션을 통해 리스트 순서대로 색상이 강제 부여됩니다!
        )
        
    else:
        st.warning("📋 공통으로 비교 가능한 수준(연간 최소 5건 이상씩 대조군 존재)의 시계열 누적 표본이 없어 그래프를 생성할 수 없습니다.")
else:
    st.warning("📋 데이터 요건을 충족하는 연도별 표본이 부족하여 그래프를 표시할 수 없습니다.")
    
st.markdown("---")

st.subheader("매장별 부정리뷰")
st.markdown("AI 모델이 부정으로 판단한 리뷰들의 모음입니다. (부정확률 내림차순)")

col_warn1, _, col_warn2 = st.columns([4, 1, 4])

# --- [후보 A 매장 출력부] ---
with col_warn1:
    st.markdown(f"💬 **[{store_A}]에 대한 고객 의견**")
    if not worst_df_A.empty:
        # 1. 상위 3개 리뷰 먼저 노출 (노란색 박스 적용)
        top_3_A = worst_df_A.head(3)
        for i, (_, row) in enumerate(top_3_A.iterrows(), 1):
            st.markdown(f"**주요 피드백 {i}** <span style='color:gray; font-size:12px;'>(AI 분석 확신도: {row['ai_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
            st.warning(f"\"{row['text']}\"")
            st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
            
        # 2. 4위부터 나머지 리뷰 전체 펼치기 (More) 제공
        remaining_A = worst_df_A.iloc[3:]
        if not remaining_A.empty:
            with st.expander(f"➕ [{store_A}] 부정 리뷰 전체 보기 ({len(worst_df_A)}건 중 {len(remaining_A)}건 더보기)"):
                for i, (_, row) in enumerate(remaining_A.iterrows(), 4):
                    st.markdown(f"**피드백 {i}** <span style='color:gray; font-size:11px;'>(AI 분석 확신도: {row['ai_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
                    st.caption(f"\"{row['text']}\"")
                    st.markdown("<hr style='margin:10px 0; border-top:1px dashed #ddd;'>", unsafe_allow_html=True)
    else:
        st.success("✅ 이 매장은 수집된 리뷰 중 AI가 검출한 주요 아쉬운 점 문맥이 없습니다.")

# --- [후보 B 매장 출력부] ---
with col_warn2:
    st.markdown(f"💬 **[{store_B}]에 대한 고객 의견**")
    if not worst_df_B.empty:
        # 1. 상위 3개 리뷰 먼저 노출 (노란색 박스 적용)
        top_3_B = worst_df_B.head(3)
        for i, (_, row) in enumerate(top_3_B.iterrows(), 1):
            st.markdown(f"**주요 피드백 {i}** <span style='color:gray; font-size:12px;'>(AI 분석 확신도: {row['ai_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
            st.warning(f"\"{row['text']}\"")
            st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
            
        # 2. 4위부터 나머지 리뷰 전체 펼치기 (More) 제공
        remaining_B = worst_df_B.iloc[3:]
        if not remaining_B.empty:
            with st.expander(f"➕ [{store_B}] 부정 리뷰 전체 보기 ({len(worst_df_B)}건 중 {len(remaining_B)}건 더보기)"):
                for i, (_, row) in enumerate(remaining_B.iterrows(), 4):
                    st.markdown(f"**피드백 {i}** <span style='color:gray; font-size:11px;'>(AI 분석 확신도: {row['ai_prob']*100:.1f}%)</span>", unsafe_allow_html=True)
                    st.caption(f"\"{row['text']}\"")
                    st.markdown("<hr style='margin:10px 0; border-top:1px dashed #ddd;'>", unsafe_allow_html=True)
    else:
        st.success("✅ 이 매장은 수집된 리뷰 중 AI가 검출한 주요 아쉬운 점 문맥이 없습니다.")