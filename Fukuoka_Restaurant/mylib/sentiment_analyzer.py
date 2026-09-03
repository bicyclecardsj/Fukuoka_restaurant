import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import numpy as np


class SentimentAnalyzer:

    def __init__(self, model_prefix, tokenizer_file):
        # 1. 5개 폴드의 최강 모델 파일(.h5)들을 리스트에 한 번에 로드합니다.
        self.models = []
        for i in range(1, 6):
            model_path = f"{model_prefix}_fold{i}.h5"
            self.models.append(load_model(model_path))
        print(f" 총 {len(self.models)}개의 Fold 모델 로드 완료!")

        # 2. 1단계에서 저장한 토크나이저 파일을 불러옵니다.
        with open(tokenizer_file, "rb") as handle:
            self.encoder = pickle.load(handle)
        print(" 맞춤형 단어 사전(Tokenizer) 로드 완료!")

        self.okt = Okt()

    def analyze_sentiment(self, review):
        # 3. [동사/부사 추가 + 원형 복원] 핵심 전처리 적용
        pos_tagged = self.okt.pos(str(review), stem=True)
        tokens = [
            word
            for word, tag in pos_tagged
            if tag in ["Noun", "Adjective", "Verb", "Adverb"]
        ]

        # 4. 글자를 숫자로 변환 후 패딩 길이 고정 (max_len=90)
        encoded_tokens = self.encoder.texts_to_sequences([tokens])
        X = pad_sequences(encoded_tokens, maxlen=90, padding="pre")

        # 5. 5개 모델의 예측 확률을 모두 더해서 평균 내기 (앙상블 지표 적용)
        combined_preds = np.zeros((1, 2))
        for model in self.models:
            combined_preds += model.predict(X, verbose=0)
        final_prob = combined_preds / 5.0  # 5개 모델의 평균 확률

        # 6. 자동 탐색으로 얻은 최적의 임계값 0.53 적용
        # final_prob[0][0]은 0번(부정) 데이터일 확률입니다.
        my_threshold = 0.53

        if final_prob[0][0] >= my_threshold:
            user_output = "부정"
            score = final_prob[0][0]  # 부정 확신도
        else:
            user_output = "긍정"
            score = final_prob[0][1]  # 긍정 확신도

        return user_output, score