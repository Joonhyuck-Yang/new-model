import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import pickle
import pandas as pd

class XMLRoBERTaClassifier:
    def __init__(self, model_name="xlm-roberta-base", model_dir=None):
        # GPU 설정 강화
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # GPU 메모리 최적화
            torch.cuda.empty_cache()
            # 혼합 정밀도 학습 활성화
            torch.backends.cudnn.benchmark = True
            print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
            print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            self.device = torch.device("cpu")
            print("CPU 사용")
        
        self.model_name = model_name
        
        # 레이블 매핑 초기화 (먼저!)
        self.label_to_id = {}
        self.id_to_label = {}
        self.label_embeddings = {}
        
        if model_dir and os.path.exists(model_dir):
            # 기존 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModel.from_pretrained(model_dir)
            # 분류기 크기는 load_classifier에서 설정됨
            self.load_classifier(model_dir)
        else:
            # 새 모델 초기화 (분류기 크기는 나중에 설정)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            # 분류기는 prepare_training_data에서 생성됨
        
        # 모델을 GPU로 이동 및 최적화
        self.model.to(self.device)
        
        # 분류기가 아직 생성되지 않았다면 임시로 생성 (prepare_training_data에서 재생성됨)
        if not hasattr(self, 'classifier'):
            self.classifier = None
        
        # GPU 메모리 사용량 출력
        if torch.cuda.is_available():
            print(f"모델 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        
    def load_classifier(self, model_dir):
        """저장된 분류기 가중치를 로드합니다."""
        classifier_path = os.path.join(model_dir, "classifier.pkl")
        if os.path.exists(classifier_path):
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
                self.classifier.to(self.device)
                print(f"분류기 로드 완료: {self.classifier.in_features} → {self.classifier.out_features}")
        else:
            print(f"분류기 파일이 없습니다: {classifier_path}")
        
        # 레이블 매핑 로드
        label_map_path = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                self.label_to_id = json.load(f)
                self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        # 레이블 임베딩 로드
        embeddings_path = os.path.join(model_dir, "label_embeddings.pkl")
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                self.label_embeddings = pickle.load(f)
                print(f"레이블 임베딩 로드 완료: {len(self.label_embeddings)}개")
        else:
            print(f"레이블 임베딩 파일이 없습니다: {embeddings_path}")
    
    def save_model(self, save_dir):
        """모델과 분류기를 저장합니다."""
        os.makedirs(save_dir, exist_ok=True)
        
        # 모델 저장
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # 분류기 저장
        with open(os.path.join(save_dir, "classifier.pkl"), 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # 레이블 매핑 저장
        with open(os.path.join(save_dir, "label_mapping.json"), 'w', encoding='utf-8') as f:
            json.dump(self.label_to_id, f, ensure_ascii=False, indent=2)
        
        # 레이블 임베딩 저장
        with open(os.path.join(save_dir, "label_embeddings.pkl"), 'wb') as f:
            pickle.dump(self.label_embeddings, f)
    
    def prepare_training_data(self, jsonl_path):
        """JSONL 파일에서 학습 데이터를 준비합니다."""
        texts = []
        labels = []
        
        print(f"JSONL 파일 경로: {jsonl_path}")
        print(f"파일 존재 여부: {os.path.exists(jsonl_path)}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    print(f"라인 {i+1}: {list(data.keys())}")
                    
                    # 각 input_text를 개별 샘플로 처리 (1~10)
                    label = data.get('Label', '')
                    if not label:
                        print(f"라인 {i+1}: 라벨이 없음, 스킵")
                        continue
                    
                    sample_count = 0
                    for j in range(1, 11):  # input_text 1부터 10까지
                        text_key = f'input_text {j}'
                        text = data.get(text_key, '').strip()
                        
                        if text:  # 텍스트가 있으면 개별 샘플로 추가
                            texts.append(text)
                            labels.append(label)
                            sample_count += 1
                            print(f"샘플 추가: {label} -> {text[:30]}...")
                    
                    print(f"라인 {i+1} ({label}): {sample_count}개 샘플 추가")
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류 라인 {i+1}: {e}")
        
        print(f"최종 결과: {len(texts)}개 텍스트, {len(labels)}개 라벨")
        
        # 레이블 매핑 생성
        unique_labels = list(set(labels))
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
        print(f"레이블 매핑 생성 완료: {len(self.label_to_id)}개 고유 라벨")
        
        # 올바른 크기의 분류기 생성 (레이블 수에 맞춤)
        num_labels = len(self.label_to_id)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.classifier.to(self.device)
        
        print(f"분류기 생성 완료: 입력 {self.model.config.hidden_size} → 출력 {num_labels}")
        
        # 레이블 임베딩 생성
        self._create_label_embeddings(texts, labels)
        
        return texts, labels
    
    def _create_label_embeddings(self, texts, labels):
        """레이블별로 대표 임베딩을 생성합니다."""
        label_texts = {}
        for text, label in zip(texts, labels):
            if label not in label_texts:
                label_texts[label] = []
            label_texts[label].append(text)
        
        self.label_embeddings = {}
        for label, texts_list in label_texts.items():
            # 해당 레이블의 모든 텍스트에 대한 평균 임베딩 계산
            embeddings = []
            for text in texts_list[:10]:  # 최대 10개 텍스트만 사용
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # [CLS] 토큰의 임베딩 사용
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(embedding)
            
            if embeddings:
                self.label_embeddings[label] = np.mean(embeddings, axis=0)
    
    def train(self, texts, labels, epochs=3, batch_size=8, learning_rate=2e-5):
        """모델을 학습합니다."""
        
        # RTX 4080 최적화를 위한 동적 배치 크기 조정
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 16:  # RTX 4080 (16GB) 이상
                if batch_size < 32:
                    batch_size = min(64, max(32, batch_size))  # 최소 32, 최대 64
                    learning_rate = min(5e-5, learning_rate * 2)  # 큰 배치에 맞춰 학습률 조정
                    print(f"🚀 RTX 4080 최적화: 배치 크기 {batch_size}, 학습률 {learning_rate}")
        
        print(f"\n🚀 모델 학습 시작!")
        print(f"{'='*60}")
        print(f"📊 학습 설정:")
        print(f"  📝 샘플 수: {len(texts)}개")
        print(f"  🏷️  고유 라벨: {len(set(labels))}개")
        print(f"  📦 배치 크기: {batch_size}")
        print(f"  📚 학습률: {learning_rate}")
        print(f"  🔄 에포크: {epochs}")
        print(f"  💻 장치: {self.device}")
        if torch.cuda.is_available():
            print(f"  🖥️  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print(f"  🔥 혼합 정밀도 학습: 활성화")
            print(f"  ⚡ GPU 최적화: RTX 4080 전용 설정")
        print(f"{'='*60}\n")
        
        if torch.cuda.is_available():
            print(f"GPU 메모리 사용량 (학습 전): {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        
        # 데이터 토크나이징
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        
        # 레이블을 ID로 변환
        label_ids = [self.label_to_id[label] for label in labels]
        
        # 데이터셋 생성
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(label_ids)
        )
        
        # 데이터로더 생성 (GPU 최적화)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0 if torch.cuda.is_available() else 2  # GPU 사용 시 단일 워커
        )
        
        # RTX 4080 최적화 옵티마이저 및 손실 함수
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.classifier.parameters()), 
            lr=learning_rate,
            weight_decay=0.01,  # 정규화
            eps=1e-6  # RTX 4080 혼합 정밀도 최적화
        )
        
        # 학습률 스케줄러 추가 (GPU 효율성 향상)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/10)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 라벨 스무딩으로 일반화 성능 향상
        
        # RTX 4080 혼합 정밀도 최적화를 위한 GradScaler
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # 학습 루프
        self.model.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs} 시작")
            print(f"{'='*50}")
            
            for batch_idx, batch in enumerate(dataloader):
                input_ids, attention_mask, label_ids = [b.to(self.device, non_blocking=True) for b in batch]
                
                optimizer.zero_grad()
                
                # RTX 4080 최적화 혼합 정밀도 학습
                if torch.cuda.is_available() and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
                        loss = criterion(logits, label_ids)
                    
                    # GradScaler로 역전파 최적화
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = self.classifier(outputs.last_hidden_state[:, 0, :])
                    loss = criterion(logits, label_ids)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                
                # 정확도 계산 (학습 중)
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == label_ids).sum().item()
                total_predictions += label_ids.size(0)
                
                # 배치별 진행상황 출력 (10배치마다)
                if batch_idx % 10 == 0:
                    batch_accuracy = (predicted == label_ids).sum().item() / label_ids.size(0)
                    print(f"  Batch {batch_idx+1}/{len(dataloader)}: Loss={loss.item():.4f}, Accuracy={batch_accuracy:.2%}")
                
                # GPU 메모리 정리 (주기적으로)
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            # 에포크별 전체 지표
            avg_loss = total_loss / len(dataloader)
            epoch_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            print(f"\n📊 Epoch {epoch+1}/{epochs} 결과:")
            print(f"  📉 평균 Loss: {avg_loss:.6f}")
            print(f"  🎯 정확도: {epoch_accuracy:.2%} ({correct_predictions}/{total_predictions})")
            print(f"  ⏱️  배치 수: {len(dataloader)}")
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                print(f"  🖥️  GPU 메모리: {gpu_memory:.2f}GB")
            
            # 학습률 스케줄러 업데이트
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            next_lr = scheduler.get_last_lr()[0]
            print(f"  📚 학습률: {current_lr:.2e} → {next_lr:.2e}")
            
            print(f"{'='*50}")
        
        # 학습 완료 후 GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n🎉 학습 완료!")
            print(f"{'='*60}")
            print(f"📊 최종 학습 결과:")
            print(f"  🔄 총 에포크: {epochs}")
            print(f"  📝 총 샘플: {len(texts)}개")
            print(f"  🏷️  고유 라벨: {len(set(labels))}개")
            print(f"  🖥️  GPU 메모리: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
            print(f"{'='*60}\n")
    
    def predict(self, text, top_k=3):
        """텍스트에 대한 예측을 수행합니다."""
        print(f"예측 시작: '{text}', 레이블 임베딩 수: {len(self.label_embeddings)}")
        self.model.eval()
        
        # 입력 텍스트 토크나이징
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            text_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        print(f"텍스트 임베딩 크기: {text_embedding.shape}")
        
        # 레이블 임베딩과의 유사도 계산
        similarities = {}
        for label, label_embedding in self.label_embeddings.items():
            try:
                # 차원 맞추기: label_embedding이 (1, 768)이면 flatten
                if label_embedding.ndim > 1:
                    label_embedding = label_embedding.flatten()
                similarity = self._cosine_similarity(text_embedding[0], label_embedding)
                similarities[label] = similarity
            except Exception as e:
                print(f"유사도 계산 오류 ({label}): {e}")
                print(f"label_embedding 크기: {label_embedding.shape if hasattr(label_embedding, 'shape') else type(label_embedding)}")
        
        print(f"유사도 계산 완료: {len(similarities)}개")
        
        # 유사도 기준으로 정렬
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 k개 결과 반환
        results = []
        for i, (label, similarity) in enumerate(sorted_similarities[:top_k]):
            results.append({
                'rank': i + 1,
                'label': label,
                'similarity': float(similarity * 100)  # 백분율로 변환
            })
        
        print(f"최종 결과 수: {len(results)}")
        return results
    
    def _cosine_similarity(self, vec1, vec2):
        """코사인 유사도를 계산합니다."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def update_with_feedback(self, text, correct_label, is_correct, memo=""):
        """사용자 피드백을 바탕으로 모델을 업데이트합니다."""
        if is_correct:
            # 올바른 예측이었을 경우, 해당 레이블의 임베딩을 강화
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                text_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 기존 임베딩과 새로운 임베딩을 가중 평균으로 업데이트
            if correct_label in self.label_embeddings:
                old_embedding = self.label_embeddings[correct_label]
                new_embedding = text_embedding[0]
                # 가중 평균 (기존: 0.7, 새로운: 0.3)
                self.label_embeddings[correct_label] = 0.7 * old_embedding + 0.3 * new_embedding
            else:
                self.label_embeddings[correct_label] = text_embedding[0]
        else:
            # 틀린 예측이었을 경우, 새로운 레이블 추가 또는 기존 레이블 업데이트
            if correct_label not in self.label_to_id:
                # 새로운 레이블 추가
                new_id = len(self.label_to_id)
                self.label_to_id[correct_label] = new_id
                self.id_to_label[new_id] = correct_label
            
            # 새로운 텍스트로 해당 레이블의 임베딩 생성
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                text_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            self.label_embeddings[correct_label] = text_embedding[0]
        
        # 메모가 있다면 저장 (향후 딥러닝을 위해)
        if memo:
            memo_file = Path(__file__).parent.parent.parent.parent / "data" / "studied" / "feedback_memos.jsonl"
            memo_file.parent.mkdir(exist_ok=True)
            
            memo_data = {
                'text': text,
                'correct_label': correct_label,
                'memo': memo,
                'timestamp': str(pd.Timestamp.now())
            }
            
            with open(memo_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(memo_data, ensure_ascii=False) + '\n')
