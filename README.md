# XML-RoBERTa 학습 및 추론 시스템

XML-RoBERTa 모델을 사용한 텍스트 분류 및 학습 시스템입니다.

## 프로젝트 구조

```
res_ananke/
├── data/
│   ├── dataforstudy/          # 학습 데이터 (엑셀 파일)
│   └── studied/               # 학습된 모델 저장소
├── frontend/                  # Next.js 프론트엔드
├── service/                   # FastAPI 백엔드
│   ├── app/
│   │   ├── ananke/           # 모델 관련 코드
│   │   │   ├── model.py      # XML-RoBERTa 분류기
│   │   │   └── excel_to_jsonl.py  # 엑셀→JSONL 변환
│   │   ├── direct_study/     # 직접 학습 모듈
│   │   ├── local_study/      # 로컬 학습 모듈
│   │   └── studied/          # 학습된 모델 관리
│   ├── main.py               # FastAPI 메인 서버
│   └── requirements.txt      # Python 의존성
└── README.md
```

## 주요 기능

### 1. 모델 학습
- 엑셀 파일의 `input_text1`, `input_text2`, `input_text3` 컬럼을 JSONL로 변환
- XML-RoBERTa 모델을 사용한 텍스트 분류 학습
- 백그라운드에서 비동기 학습 진행
- 학습 진행 상황 실시간 모니터링

### 2. 텍스트 분류
- 입력된 텍스트를 기반으로 유사도 계산
- 상위 3개 라벨을 순위별로 표시
- 각 라벨별 유사도 백분율 표시

### 3. 피드백 시스템
- 사용자가 예측 결과의 정확성 평가
- 올바른 예측: 해당 라벨의 유사도 학습 강화
- 틀린 예측: 새로운 라벨 추가 및 메모 저장
- 모든 피드백이 모델에 누적 반영

### 4. 모델 저장 및 공유
- 학습된 모델을 `@studied/` 폴더에 자동 저장
- 다른 컴퓨터에서도 학습된 모델 사용 가능
- 피드백 메모를 통한 지속적인 모델 개선

## 설치 및 실행

### 1. 백엔드 설정
```bash
cd service
pip install -r requirements.txt
python main.py
```

### 2. 프론트엔드 설정

#### npm 사용
```bash
cd frontend
npm install
npm run dev
```

#### pnpm 사용 (권장)
```bash
cd frontend
pnpm install
pnpm dev
```

또는 루트 디렉토리에서 `run_frontend_pnpm.bat` 실행

### 3. 모델 다운로드
시스템이 처음 실행될 때 `xlm-roberta-base` 모델이 자동으로 다운로드됩니다.

## API 엔드포인트

- `GET /`: 서버 상태 확인
- `POST /train`: 모델 학습 시작
- `GET /training-status`: 학습 진행 상황 확인
- `POST /predict`: 텍스트 분류 예측
- `POST /feedback`: 사용자 피드백 제출
- `GET /model-info`: 모델 정보 조회

## 사용법

1. **초기 학습**: 엑셀 파일을 `data/dataforstudy/`에 넣고 학습 시작
2. **텍스트 분류**: 입력 텍스트에 대한 분류 결과 확인
3. **피드백 제공**: 예측 결과의 정확성 평가 및 수정
4. **모델 개선**: 피드백을 통한 지속적인 모델 성능 향상

## 기술 스택

- **백엔드**: FastAPI, PyTorch, Transformers, scikit-learn
- **프론트엔드**: Next.js, React, TypeScript, Tailwind CSS
- **모델**: XML-RoBERTa (xlm-roberta-base)
- **데이터 처리**: pandas, openpyxl
- **패키지 매니저**: pnpm (권장), npm

## pnpm 사용법

### pnpm 설치
```bash
npm install -g pnpm
```

### pnpm 명령어
- `pnpm install`: 의존성 설치
- `pnpm dev`: 개발 서버 시작
- `pnpm build`: 프로덕션 빌드
- `pnpm start`: 프로덕션 서버 시작
- `pnpm clean`: 의존성 정리

## 주의사항

- 첫 실행 시 모델 다운로드로 인해 시간이 오래 걸릴 수 있습니다
- GPU가 있다면 자동으로 사용되며, CPU만으로도 동작합니다
- 학습 중에는 예측 기능을 사용할 수 없습니다
- 모델 파일은 용량이 클 수 있으므로 충분한 저장 공간을 확보하세요
