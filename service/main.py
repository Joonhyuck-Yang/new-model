from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import json
from pathlib import Path
import asyncio
import threading
import torch
from datetime import datetime

from app.ananke.model import XMLRoBERTaClassifier

app = FastAPI(title="XML-RoBERTa 학습 및 추론 API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 모델 인스턴스 관리
classifier = None
is_training = False
training_progress = {"status": "idle", "message": "", "progress": 0}

# RTX 2080 최적화 설정
RTX_2080_CONFIG = {
    "batch_size": 16,  # RTX 2080 8GB VRAM에 최적화
    "max_length": 512,
    "learning_rate": 3e-5,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 2,
    "fp16": True,  # 혼합 정밀도 학습으로 성능 향상
    "dataloader_num_workers": 4
}

# Pydantic 모델들
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predictions: List[dict]

class FeedbackRequest(BaseModel):
    text: str
    correct_label: str
    is_correct: bool
    memo: Optional[str] = ""

class TrainingRequest(BaseModel):
    epochs: int = 3
    batch_size: Optional[int] = None  # None이면 RTX 2080 최적화 설정 사용
    learning_rate: Optional[float] = None

class TrainingResponse(BaseModel):
    message: str
    status: str

class ModelInfoResponse(BaseModel):
    status: str
    model_name: Optional[str] = None
    num_labels: Optional[int] = None
    labels: Optional[List[str]] = None
    device: Optional[str] = None
    training_history: Optional[List[dict]] = None
    gpu_info: Optional[dict] = None

# 모델 초기화 함수
def initialize_model():
    global classifier
    
    # 기존 학습된 모델이 있는지 확인
    model_dir = Path(__file__).parent.parent / "data" / "studied" / "latest_model"
    
    if model_dir.exists():
        print("기존 학습된 모델을 로드합니다...")
        classifier = XMLRoBERTaClassifier(model_dir=str(model_dir))
    else:
        print("새로운 모델을 초기화합니다...")
        classifier = XMLRoBERTaClassifier()
    
    print("모델 초기화 완료")

# 백그라운드에서 모델을 초기화
@app.on_event("startup")
async def startup_event():
    # 별도 스레드에서 모델 초기화 (시간이 오래 걸릴 수 있음)
    def init_in_background():
        initialize_model()
    
    thread = threading.Thread(target=init_in_background)
    thread.daemon = True
    thread.start()

# 학습 진행 상황 업데이트 함수
def update_training_progress(status, message, progress=0):
    global training_progress
    training_progress.update({
        "status": status,
        "message": message,
        "progress": progress
    })

# RTX 2080 최적화된 학습 설정 가져오기
def get_optimized_training_config(request_batch_size=None, request_lr=None):
    """RTX 2080에 최적화된 학습 설정을 반환합니다."""
    config = RTX_2080_CONFIG.copy()
    
    # 사용자 요청이 있으면 오버라이드
    if request_batch_size:
        config["batch_size"] = request_batch_size
    if request_lr:
        config["learning_rate"] = request_lr
    
    # GPU 메모리 상태에 따른 동적 조정
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_memory < 8:
            config["batch_size"] = min(config["batch_size"], 8)
            config["fp16"] = True
        elif gpu_memory >= 11:  # RTX 2080 Ti 등
            config["batch_size"] = min(config["batch_size"], 24)
    
    return config

# JSONL 파일에서 학습 데이터 준비
def prepare_jsonl_training_data(jsonl_path):
    """JSONL 파일에서 학습 데이터를 준비합니다."""
    texts = []
    labels = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 다양한 컬럼명 패턴 지원
                label = None
                text_parts = []
                
                # 라벨 찾기
                for key in ['Label', 'label', 'LABEL']:
                    if key in data and data[key]:
                        label = str(data[key])
                        break
                
                # 텍스트 부분들 찾기
                for i in range(1, 11):  # input_text 1~10
                    for pattern in [f'input_text {i}', f'input_text {i}', f'input_text{i}']:
                        if pattern in data and data[pattern]:
                            text_parts.append(str(data[pattern]))
                            break
                
                # 텍스트 결합
                combined_text = ' '.join(filter(None, text_parts))
                
                if combined_text and label:
                    texts.append(combined_text)
                    labels.append(label)
                else:
                    print(f"라인 {line_num}: 유효하지 않은 데이터 - 텍스트: {bool(combined_text)}, 라벨: {bool(label)}")
                    
            except json.JSONDecodeError as e:
                print(f"라인 {line_num}: JSON 파싱 오류 - {e}")
                continue
            except Exception as e:
                print(f"라인 {line_num}: 처리 오류 - {e}")
                continue
    
    return texts, labels

# 백그라운드 학습 함수
def train_model_background(epochs, batch_size, learning_rate):
    global classifier, is_training
    
    try:
        is_training = True
        update_training_progress("training", "학습 데이터 준비 중...", 10)
        
        # 최적화된 학습 설정 가져오기
        training_config = get_optimized_training_config(batch_size, learning_rate)
        
        current_dir = Path(__file__).parent.parent
        jsonl_path = current_dir / "data" / "dataforstudy" / "file for trainning.jsonl"
        
        if not jsonl_path.exists():
            update_training_progress("error", "JSONL 파일을 찾을 수 없습니다.")
            return
        
        update_training_progress("training", "JSONL 데이터 처리 중...", 20)
        texts, labels = prepare_jsonl_training_data(jsonl_path)
        
        if not texts or not labels:
            update_training_progress("error", "학습 데이터가 비어있습니다.")
            return
        
        update_training_progress("training", f"총 {len(texts)}개 데이터로 학습을 시작합니다...", 30)
        
        # RTX 2080 최적화된 학습 수행
        for epoch in range(epochs):
            progress = 30 + (epoch + 1) * (60 // epochs)
            update_training_progress("training", f"Epoch {epoch + 1}/{epochs} 학습 중...", progress)
            
            # 최적화된 설정으로 학습
            classifier.train(
                texts, 
                labels, 
                epochs=1, 
                batch_size=training_config["batch_size"]
            )
        
        update_training_progress("training", "모델 저장 중...", 90)
        
        # 학습된 모델 저장
        save_dir = current_dir / "data" / "studied" / "latest_model"
        classifier.save_model(str(save_dir))
        
        # 학습 히스토리 저장
        training_info = {
            'epochs': epochs,
            'batch_size': training_config["batch_size"],
            'learning_rate': training_config["learning_rate"],
            'training_samples': len(texts),
            'unique_labels': len(set(labels)),
            'data_source': 'jsonl',
            'gpu_optimized': True,
            'gpu_config': training_config
        }
        
        # 학습 히스토리 파일에 저장
        history_file = current_dir / "data" / "studied" / "main_training_history.jsonl"
        history_file.parent.mkdir(exist_ok=True)
        
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(training_info, ensure_ascii=False) + '\n')
        
        update_training_progress("completed", "학습이 완료되었습니다!", 100)
        
    except Exception as e:
        update_training_progress("error", f"학습 중 오류가 발생했습니다: {str(e)}")
    finally:
        is_training = False

@app.get("/")
async def root():
    return {"message": "XML-RoBERTa 학습 및 추론 API가 실행 중입니다."}

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    global is_training
    
    if is_training:
        raise HTTPException(status_code=400, detail="이미 학습이 진행 중입니다.")
    
    if classifier is None:
        raise HTTPException(status_code=500, detail="모델이 아직 초기화되지 않았습니다.")
    
    # 백그라운드에서 학습 시작
    background_tasks.add_task(
        train_model_background, 
        request.epochs, 
        request.batch_size, 
        request.data_source, 
        request.learning_rate
    )
    
    return TrainingResponse(
        message="학습이 백그라운드에서 시작되었습니다.",
        status="started"
    )

@app.get("/training-status")
async def get_training_status():
    return training_progress

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global classifier
    
    if classifier is None:
        raise HTTPException(status_code=500, detail="모델이 아직 초기화되지 않았습니다.")
    
    if is_training:
        raise HTTPException(status_code=400, detail="학습이 진행 중입니다. 잠시 후 다시 시도해주세요.")
    
    try:
        predictions = classifier.predict(request.text, top_k=3)
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류가 발생했습니다: {str(e)}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    global classifier, is_training
    
    if classifier is None:
        raise HTTPException(status_code=500, detail="모델이 아직 초기화되지 않았습니다.")
    
    try:
        # 1. 피드백을 바탕으로 모델 즉시 업데이트
        classifier.update_with_feedback(
            request.text,
            request.correct_label,
            request.is_correct,
            request.memo
        )
        
        # 2. 피드백 데이터를 학습 데이터로 저장
        current_dir = Path(__file__).parent.parent
        feedback_data_dir = current_dir / "data" / "studied"
        feedback_data_dir.mkdir(exist_ok=True)
        
        feedback_file = feedback_data_dir / "feedback_training_data.jsonl"
        feedback_entry = {
            'text': request.text,
            'label': request.correct_label,
            'is_correct': request.is_correct,
            'memo': request.memo,
            'timestamp': datetime.now().isoformat(),
            'source': 'user_feedback'
        }
        
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        
        # 3. 즉시 학습 실행 (동기적으로 빠른 학습)
        if not is_training:
            try:
                is_training = True
                update_training_progress("training", "피드백 기반 즉시 학습 중...", 10)
                
                # 피드백 데이터로 즉시 학습
                feedback_texts = [request.text]
                feedback_labels = [request.correct_label]
                
                # RTX 2080 최적화 설정 (피드백 학습용 - 빠른 학습)
                training_config = get_optimized_training_config(2, 1e-4)  # 매우 작은 배치, 높은 학습률
                
                update_training_progress("training", "피드백 데이터로 학습 중...", 30)
                
                # 즉시 학습 수행 (빠른 학습)
                classifier.train(
                    feedback_texts, 
                    feedback_labels, 
                    epochs=1, 
                    batch_size=training_config["batch_size"]
                )
                
                update_training_progress("training", "모델 저장 중...", 80)
                
                # 업데이트된 모델 저장
                save_dir = current_dir / "data" / "studied" / "latest_model"
                classifier.save_model(str(save_dir))
                
                # 학습 히스토리 저장
                training_info = {
                    'epochs': 1,
                    'batch_size': training_config["batch_size"],
                    'learning_rate': training_config["learning_rate"],
                    'training_samples': len(feedback_texts),
                    'unique_labels': len(set(feedback_labels)),
                    'data_source': 'user_feedback',
                    'feedback_text': request.text[:50] + "..." if len(request.text) > 50 else request.text,
                    'gpu_optimized': True,
                    'gpu_config': training_config,
                    'training_type': 'immediate_feedback'
                }
                
                history_file = current_dir / "data" / "studied" / "main_training_history.jsonl"
                with open(history_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(training_info, ensure_ascii=False) + '\n')
                
                update_training_progress("completed", "피드백 기반 즉시 학습이 완료되었습니다!", 100)
                
            except Exception as e:
                update_training_progress("error", f"피드백 학습 중 오류: {str(e)}")
            finally:
                is_training = False
        
        return {"message": "피드백이 반영되고 즉시 학습이 시작되었습니다!"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"피드백 처리 중 오류가 발생했습니다: {str(e)}")

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    global classifier
    
    if classifier is None:
        return ModelInfoResponse(status="not_initialized")
    
    # GPU 정보 수집
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "memory_cached_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
            "compute_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
        }
    
    # 학습 히스토리 로드
    training_history = []
    try:
        current_dir = Path(__file__).parent.parent
        history_file = current_dir / "data" / "studied" / "main_training_history.jsonl"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        training_history.append(json.loads(line.strip()))
                    except:
                        continue
    except Exception as e:
        print(f"학습 히스토리 로드 오류: {e}")
    
    return ModelInfoResponse(
        status="ready",
        model_name=classifier.model_name,
        num_labels=len(classifier.label_to_id),
        labels=list(classifier.label_to_id.keys()),
        device=str(classifier.device),
        training_history=training_history,
        gpu_info=gpu_info
    )

@app.get("/gpu-info")
async def get_gpu_info():
    """GPU 정보를 반환합니다."""
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA를 사용할 수 없습니다."}
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpu_info.append({
            "index": i,
            "name": props.name,
            "memory_total_gb": round(props.total_memory / 1024**3, 2),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
            "memory_cached_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count
        })
    
    return {
        "available": True,
        "gpu_count": gpu_count,
        "gpus": gpu_info,
        "current_device": torch.cuda.current_device()
    }

@app.get("/training-config")
async def get_training_config():
    """현재 학습 설정을 반환합니다."""
    return {
        "rtx_2080_optimized": RTX_2080_CONFIG,
        "current_gpu": get_optimized_training_config() if torch.cuda.is_available() else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
