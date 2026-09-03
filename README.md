# 🍜 후쿠오카 음식점 리뷰 분석 대시보드

Google Maps에서 후쿠오카 음식점 리뷰를 수집하고, 한국어 자연어 처리와 딥러닝 감성 분석을 거쳐 음식점별 평점·긍정 리뷰 비율·연도별 추이를 비교하는 Streamlit 웹 대시보드입니다.

---

## 🛠 기술 스택

### Environment

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

### Framework & Libraries

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white)](https://www.selenium.dev/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![KoNLPy](https://img.shields.io/badge/KoNLPy-Okt-4B8BBE?style=for-the-badge)](https://konlpy.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
---

## 👥 프로젝트 목적

후쿠오카 음식점에 대한 대규모 리뷰를 정량적으로 분석하여, 단순 평균 평점만으로는 확인하기 어려운 고객 반응과 시간에 따른 평가 변화를 비교하고 음식점 선택에 도움을 주는 것을 목표로 합니다.

---

## 📌 주요 기능

- **Google Maps 리뷰 수집**
  - 후쿠오카 관광·숙박·음식점 관련 장소 검색
  - 장소명, 카테고리, 작성자, 평점, 작성일, 리뷰 본문 수집
  - 장소 단위 중간 저장 및 중단 지점부터 이어서 수집

- **한국어 리뷰 전처리**
  - 결측값과 중복 리뷰 제거
  - 일본어 리뷰 및 분석에 불필요한 데이터 필터링
  - KoNLPy의 `Okt`를 이용해 명사·형용사·동사·부사 추출
  - 상대 날짜를 연도 및 연월 데이터로 변환

- **AI 감성 분석**
  - CNN과 양방향 GRU를 결합한 딥러닝 모델 사용
  - 5-Fold 교차검증으로 학습한 모델의 예측 결과를 앙상블
  - 리뷰별 긍정·부정 라벨과 AI 확신도 제공

- **음식점 1:1 비교 대시보드**
  - 카테고리별 음식점 두 곳 선택 및 비교
  - Google Maps 평점과 AI 긍정 리뷰 비율 표시
  - 연도별 긍정·부정 리뷰 비율 추이 시각화
  - AI가 분류한 주요 부정 리뷰를 확신도순으로 제공
  - 선택한 음식점의 Google Maps 검색 페이지 연결

---

## 📸 실행 화면

| 1. 음식점 및 카테고리 선택 | 2. 평점·긍정 비율 비교 | 3. 연도별 추이 및 부정 리뷰 |
| :---: | :---: | :---: |
| 이미지 추가 예정 | 이미지 추가 예정 | 이미지 추가 예정 |

---

## 🚀 시작 가이드

### 1. 프로젝트 준비

```bash
git clone <repository-url>
cd Fukuoka_Restaurant
```

가상환경을 생성하고 필요한 라이브러리를 설치합니다.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install streamlit pandas numpy selenium konlpy tensorflow scikit-learn
```

> KoNLPy의 `Okt` 사용을 위해 Java가 필요할 수 있습니다. 실행 전 Java 설치 및 환경 변수 설정을 확인해 주세요.

### 2. Streamlit 대시보드 실행

GitHub 저장소에는 AI 분석이 완료된 대시보드용 데이터가 포함됩니다. 현재 대시보드는 저장된 분석 결과를 조회하므로 학습 모델을 별도로 내려받지 않아도 실행할 수 있습니다.

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
streamlit run 스트림릿.py
```

실행 후 터미널에 표시되는 로컬 주소로 접속합니다. 기본 주소는 일반적으로 다음과 같습니다.

```text
http://localhost:8501
```

### 3. 리뷰 수집하기 (선택)

음식점 카테고리의 장소와 리뷰를 수집하는 예시입니다.

```bash
python GoogleReviewCrawler.py --topic 음식점 --max-places 40 --max-reviews 5000
```

특정 카테고리 또는 단일 장소만 수집할 수도 있습니다.

```bash
python GoogleReviewCrawler.py --topic 음식점 --category 라멘
python GoogleReviewCrawler.py --url "<Google Maps 장소 URL>" --max-reviews 1000
```

수집 결과는 기본적으로 `output/` 디렉터리에 저장됩니다.

### 4. 전체 리뷰 감성 분석하기 (선택)

GitHub 저장소에 포함된 모델, tokenizer 및 압축된 전체 정제 데이터를 사용해 리뷰를 다시 분석할 수 있습니다. Pandas는 `.csv.gz` 파일을 직접 읽을 수 있으므로 별도 압축 해제는 필요하지 않습니다.

`preprocess_fast.py`의 입력 파일 경로가 다음과 같이 압축 파일을 가리키는지 확인합니다.

```python
df = pd.read_csv("./data/후쿠오카_리뷰_최종정제본.csv.gz")
```

그다음 일괄 추론을 실행합니다.

```bash
python preprocess_fast.py
```

대시보드에서는 `data/후쿠오카_리뷰_최종.csv.gz`를 읽습니다. 새 결과를 생성했다면 gzip으로 압축한 뒤 해당 경로에 배치합니다.

---

## 🧠 모델 학습 및 재현

포트폴리오에서 모델링 과정을 확인할 수 있도록 학습·평가 코드는 GitHub 저장소에 공개합니다.

- `preprocessing.ipynb`: 결측치·중복치 제거, 언어 필터링 및 학습 라벨 생성
- `후쿠오카_음식점_리뷰분석.ipynb`: 토큰화, 모델 설계, 5-Fold 학습 및 평가
- `모델테스트.ipynb`: 저장된 앙상블 모델의 단일 리뷰 추론 테스트
- `preprocess_fast.py`: 전체 리뷰를 대상으로 하는 배치 추론
- `mylib/sentiment_analyzer.py`: 모델과 tokenizer 로딩 및 감성 분석

감성 분석 모델은 CNN과 Bidirectional GRU를 결합한 구조입니다. 5-Fold로 학습된 모델들의 예측 확률을 평균내며, 부정 클래스 확률에 임계값 `0.53`을 적용해 최종 라벨을 결정합니다.

학습 과정에 사용된 전체 정제 데이터, 모델 5개 및 tokenizer도 GitHub 저장소에 포함하므로 저장소를 clone하면 모델 구조와 추론 과정을 함께 확인할 수 있습니다.

---

## 📦 GitHub 파일 관리 정책

포트폴리오 검토자가 하나의 저장소에서 수집·전처리·학습·추론·대시보드 과정을 모두 확인할 수 있도록 필요한 파일을 GitHub에 함께 관리합니다. 용량이 큰 CSV 파일만 gzip으로 압축합니다.

### GitHub에 포함하는 파일

| 구분 | 파일 | 목적 |
|---|---|---|
| 애플리케이션 | `스트림릿.py` | Streamlit 배포 및 대시보드 실행 |
| 수집·추론 코드 | `GoogleReviewCrawler.py`, `preprocess_fast.py`, `mylib/` | 전체 파이프라인 공개 |
| 학습 과정 | `*.ipynb` | 전처리, 모델 설계, 학습 및 평가 과정 공개 |
| 매장 정보 | `data/가게리스트_평점.csv` | 음식점 카테고리와 Google 평점 표시 |
| 최종 분석 결과 | `data/후쿠오카_리뷰_최종.csv.gz` | 모델 없이 대시보드 실행 |
| 전체 정제 데이터 | `data/후쿠오카_리뷰_최종정제본.csv.gz` | 모델 재학습 및 전체 배치 추론 |
| 학습 완료 모델 | `model/best_model_fold1.h5` ~ `best_model_fold5.h5` | 5-Fold 앙상블 재현 |
| Tokenizer | `model/my_best_tokenizer.pickle` | 학습 당시 단어 인덱스 재사용 |
| 프로젝트 문서 | `README.md`, `requirements.txt`, `.gitignore` | 설치·실행·보관 방법 안내 |

### GitHub에서 제외하는 파일

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/

# 압축 전 대용량 CSV
data/후쿠오카_리뷰_최종.csv
data/후쿠오카_리뷰_최종정제본.csv
```

`__pycache__`와 `*.pyc`는 Python이 실행 중 자동 생성하는 캐시이므로 저장소에 포함하지 않습니다. 압축 전 CSV는 로컬에 보관하고 GitHub에는 `.csv.gz` 파일만 커밋합니다.

---

## 📁 프로젝트 구조

```text
Fukuoka_Restaurant/
├── data/
│   ├── 가게리스트_평점.csv               # 음식점 카테고리 및 Google 평점
│   ├── 후쿠오카_리뷰_최종정제본.csv.gz    # 압축된 전체 정제 리뷰 데이터
│   ├── 후쿠오카_리뷰_최종.csv.gz          # 압축된 AI 감성 분석 결과
│   └── progress_음식점.json              # 리뷰 수집 진행 기록
├── model/
│   ├── best_model_fold1.h5               # 5-Fold 학습 모델
│   ├── best_model_fold2.h5
│   ├── best_model_fold3.h5
│   ├── best_model_fold4.h5
│   ├── best_model_fold5.h5
│   └── my_best_tokenizer.pickle          # 학습 데이터 기반 tokenizer
├── mylib/
│   ├── sentiment_analyzer.py             # 단일 리뷰 감성 분석 모듈
│   └── my_utils.py                       # 전처리 및 통계 보조 함수
├── GoogleReviewCrawler.py                # Google Maps 리뷰 수집기
├── preprocess_fast.py                    # 전체 리뷰 5-Fold 앙상블 추론
├── 스트림릿.py                           # Streamlit 대시보드 메인 파일
├── preprocessing.ipynb                   # 리뷰 전처리 과정
├── 후쿠오카_음식점_리뷰분석.ipynb         # 모델 학습 및 평가 과정
├── 모델테스트.ipynb                      # 저장 모델 테스트
├── CSV생성_컬럼제거.ipynb                # 매장 목록 및 최종 CSV 생성
└── README.md                             # 프로젝트 설명 문서
```

---

## 📊 데이터 구성

### 대시보드용 음식점 정보

| 컬럼 | 설명 |
|---|---|
| `place_name` | 음식점 이름 |
| `category` | 음식점 카테고리 |
| `google_rating` | Google Maps 평점 |

### AI 분석 완료 리뷰

| 컬럼 | 설명 |
|---|---|
| `place_name` | 음식점 이름 |
| `text` | 리뷰 본문 |
| `year` | 리뷰 작성 연도 |
| `ai_label` | 감성 분석 결과 (`0`: 부정, `1`: 긍정) |
| `ai_prob` | 예측된 감성에 대한 AI 확신도 |
