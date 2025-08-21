# XML-RoBERTa 시스템 설정 및 실행 가이드

## 1. 시스템 요구사항

### 하드웨어
- **RAM**: 최소 8GB (16GB 권장)
- **저장공간**: 최소 10GB (모델 파일 포함)
- **GPU**: 선택사항 (CUDA 지원 GPU가 있으면 학습 속도 향상)

### 소프트웨어
- **Python**: 3.8 이상
- **Node.js**: 16 이상
- **npm**: 8 이상

## 2. 초기 설정

### 2.1 Python 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2.2 백엔드 의존성 설치
```bash
cd service
pip install -r requirements.txt
```

### 2.3 프론트엔드 의존성 설치
```bash
cd frontend
npm install
```

## 3. 실행 방법

### 3.1 백엔드 서버 실행
```bash
cd service
python main.py
```
또는 Windows에서 `run_backend.bat` 실행

**주의**: 첫 실행 시 `xlm-roberta-base` 모델 다운로드로 인해 시간이 오래 걸릴 수 있습니다.

### 3.2 프론트엔드 실행
```bash
cd frontend
npm run dev
```
또는 Windows에서 `run_frontend.bat` 실행

## 4. 사용 순서

### 4.1 초기 학습
1. `data/dataforstudy/` 폴더에 엑셀 파일 배치
2. 엑셀 파일에 `input_text1`, `input_text2`, `input_text3`, `label` 컬럼 포함
3. 프론트엔드에서 "학습 시작" 버튼 클릭
4. 학습 진행 상황 모니터링

### 4.2 텍스트 분류
1. 학습 완료 후 텍스트 입력
2. "분류하기" 버튼 클릭
3. 상위 3개 라벨과 유사도 확인

### 4.3 피드백 제공
1. 예측 결과의 정확성 평가
2. 올바른 라벨 입력
3. 필요시 메모 추가
4. "피드백 제출" 버튼 클릭

## 5. 문제 해결

### 5.1 모델 다운로드 실패
- 인터넷 연결 확인
- 방화벽 설정 확인
- Hugging Face 접근 가능 여부 확인

### 5.2 메모리 부족 오류
- 배치 크기 줄이기 (`batch_size`를 4 또는 2로 설정)
- 다른 프로그램 종료
- 가상 메모리 증가

### 5.3 CUDA 오류
- GPU 드라이버 업데이트
- PyTorch CUDA 버전 확인
- CPU 모드로 전환 (자동 처리됨)

## 6. 성능 최적화

### 6.1 학습 속도 향상
- GPU 사용 (CUDA 지원)
- 배치 크기 증가 (메모리 허용 시)
- 에포크 수 조정

### 6.2 메모리 사용량 최적화
- 배치 크기 줄이기
- 최대 시퀀스 길이 제한 (현재 512)
- 그래디언트 체크포인팅 사용

## 7. 백업 및 복원

### 7.1 모델 백업
```bash
# data/studied/latest_model/ 폴더 전체 복사
cp -r data/studied/latest_model/ backup_model/
```

### 7.2 다른 컴퓨터에서 복원
1. 백업된 모델 폴더를 `data/studied/latest_model/`에 복사
2. 백엔드 서버 재시작
3. 모델 정보 확인

## 8. 모니터링

### 8.1 학습 진행 상황
- 프론트엔드의 진행률 바 확인
- 백엔드 콘솔 로그 모니터링
- `GET /training-status` API 호출

### 8.2 모델 성능
- 피드백 정확도 추적
- 새로운 라벨 추가 빈도
- 유사도 점수 분포

## 9. 확장 가능성

### 9.1 새로운 데이터 추가
- 엑셀 파일에 새 데이터 추가
- 재학습 수행
- 기존 모델에 누적 학습

### 9.2 모델 아키텍처 변경
- `service/app/ananke/model.py` 수정
- 다른 Transformer 모델 사용 가능
- 커스텀 분류기 구현

## 10. 지원 및 문의

문제가 발생하거나 추가 기능이 필요한 경우:
1. 로그 파일 확인
2. API 응답 상태 확인
3. 모델 파일 무결성 검증
