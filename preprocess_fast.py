import os
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 텐서플로우 로그 제어 및 GPU 최적화 세팅
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from mylib.sentiment_analyzer import SentimentAnalyzer

if __name__ == "__main__":
    # 2. 원본 데이터 및 분석기 로드
    df = pd.read_csv('./data/후쿠오카_리뷰_최종정제본.csv')
    sa = SentimentAnalyzer(
        model_prefix="./model/best_model",
        tokenizer_file="./model/my_best_tokenizer.pickle",
    )
    
    okt = Okt()
    
    print("\n[1단계] 온전한 모델 형태 유지를 위한 형태소 파싱 시작...")
    
    tokenized_reviews = []    
    
    for idx, text in enumerate(df['text']):
        pos_tagged = okt.pos(str(text), stem=True)
        
        # 모델 추론용 규격 (기존 5-Fold 모델이 학습한 명사, 형용사, 동사, 부사 형태 유지)
        model_tokens = [w for w, tag in pos_tagged if tag in ["Noun", "Adjective", "Verb", "Adverb"]]
        tokenized_reviews.append(model_tokens)
        
        if (idx + 1) % 20000 == 0:
            print(f"   > 텍스트 파싱 진행률: {idx + 1}/{len(df)} 건 완료")

    # ----------------------------------------------------
    # 단계 2: 내 모델 사전 기반 고속 배치(Batch) 추론
    # ----------------------------------------------------
    print("\n🚀 [2단계] GPU/CPU 배치 모드로 5-Fold 앙상블 추론 시작...")
    
    sequences = sa.encoder.texts_to_sequences(tokenized_reviews)
    padded_data = pad_sequences(sequences, maxlen=90, padding="pre")
    
    ensemble_preds = np.zeros((len(df), 2))
    
    for i, model in enumerate(sa.models):
        print(f"  🧠 [Fold {i+1}/5] 모델 배치 추론 중...")
        fold_preds = model.predict(padded_data, batch_size=1024, verbose=0)
        ensemble_preds += fold_preds
        
    # 5개 모델의 예측 확률 평균 구하기
    final_probs = ensemble_preds / 5.0
    
    ai_labels = []
    ai_probs = []
    my_threshold = 0.53
    
    for prob in final_probs:
        if prob[0] >= my_threshold:
            ai_labels.append(0)  # 부정
            ai_probs.append(prob[0])
        else:
            ai_labels.append(1)  # 긍정
            ai_probs.append(prob[1])
            
    df['ai_label'] = ai_labels
    df['ai_prob'] = ai_probs
    
    # ----------------------------------------------------
    # 단계 3: 파일 최종 저장 및 정리
    # ----------------------------------------------------
    # 스트림릿 대시보드 스펙에 맞게 딱 필요한 컬럼만 남기고 정리
    final_cols = ['place_name', 'text', 'year', 'ai_label', 'ai_prob']
    df_final = df[final_cols]
    
    df_final.to_csv('후쿠오카_리뷰_최종.csv', index=False, encoding='utf-8-sig')
    
    print("\n[예측 완료] 긍부정 예측 완료 데이터 저장")