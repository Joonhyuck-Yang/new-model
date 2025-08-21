# 학습된 모델 저장 디렉토리

이 디렉토리에는 학습이 완료된 모델들이 저장됩니다.

## 구조
- `latest_model/`: 가장 최근에 학습된 모델
- `feedback_memos.jsonl`: 사용자 피드백 메모들

## 모델 파일들
- `config.json`: 모델 설정
- `pytorch_model.bin`: 모델 가중치
- `tokenizer.json`: 토크나이저 설정
- `classifier.pkl`: 분류기 가중치
- `label_mapping.json`: 레이블 매핑 정보
- `label_embeddings.pkl`: 레이블 임베딩

## 백업 및 공유
이 디렉토리의 내용을 다른 컴퓨터로 복사하면 학습된 모델을 그대로 사용할 수 있습니다.
