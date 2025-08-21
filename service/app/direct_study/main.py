"""
직접 학습 모듈

이 모듈은 사용자가 직접 입력한 데이터로 실시간 학습을 수행합니다.
피드백을 받아 즉시 모델을 업데이트하고, 학습된 내용을 누적하여 저장합니다.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import threading
import time

# 상위 디렉토리의 모델 클래스 import
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.ananke.model import XMLRoBERTaClassifier

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectStudyManager:
    """직접 학습 관리자 클래스"""
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        초기화
        
        Args:
            model_dir: 기존 학습된 모델이 있는 디렉토리 경로
        """
        self.current_dir = Path(__file__).parent.parent.parent.parent
        self.data_dir = self.current_dir / "data"
        self.studied_dir = self.data_dir / "studied"
        
        # 디렉토리 생성
        self.studied_dir.mkdir(exist_ok=True)
        
        # 모델 초기화
        if model_dir and os.path.exists(model_dir):
            logger.info(f"기존 모델을 로드합니다: {model_dir}")
            self.classifier = XMLRoBERTaClassifier(model_dir=model_dir)
        else:
            logger.info("새로운 모델을 초기화합니다")
            self.classifier = XMLRoBERTaClassifier()
        
        # 실시간 학습 데이터
        self.realtime_data = []
        self.feedback_queue = []
        
        # 학습 상태
        self.is_training = False
        self.training_thread = None
        
        # 학습 히스토리
        self.training_history = self._load_training_history()
        
        # 실시간 학습 시작
        self._start_realtime_training()
    
    def _load_training_history(self) -> List[Dict]:
        """학습 히스토리를 로드합니다."""
        history_file = self.studied_dir / "direct_training_history.jsonl"
        if history_file.exists():
            history = []
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        history.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            return history
        return []
    
    def _save_training_history(self, training_info: Dict):
        """학습 히스토리를 저장합니다."""
        history_file = self.studied_dir / "direct_training_history.jsonl"
        
        # 타임스탬프 추가
        training_info['timestamp'] = datetime.now().isoformat()
        training_info['training_id'] = len(self.training_history) + 1
        
        self.training_history.append(training_info)
        
        # JSONL 파일에 저장
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(training_info, ensure_ascii=False) + '\n')
        
        logger.info(f"직접 학습 히스토리가 저장되었습니다: {training_info['training_id']}번째 학습")
    
    def add_training_data(self, text: str, label: str, source: str = "manual"):
        """
        학습 데이터를 추가합니다.
        
        Args:
            text: 학습할 텍스트
            label: 해당 라벨
            source: 데이터 출처 (manual, feedback, import 등)
        """
        data_item = {
            'text': text,
            'label': label,
            'source': source,
            'timestamp': datetime.now().isoformat()
        }
        
        self.realtime_data.append(data_item)
        logger.info(f"학습 데이터가 추가되었습니다: {label} (출처: {source})")
        
        # 데이터가 충분히 쌓이면 자동 학습
        if len(self.realtime_data) >= 5:  # 5개 이상 쌓이면 학습
            self._trigger_auto_training()
    
    def add_feedback(self, text: str, correct_label: str, is_correct: bool, memo: str = ""):
        """
        사용자 피드백을 추가합니다.
        
        Args:
            text: 원본 텍스트
            correct_label: 올바른 라벨
            is_correct: 예측이 올바른지 여부
            memo: 추가 메모
        """
        feedback_item = {
            'text': text,
            'correct_label': correct_label,
            'is_correct': is_correct,
            'memo': memo,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_queue.append(feedback_item)
        logger.info(f"피드백이 추가되었습니다: {correct_label} (정확: {is_correct})")
        
        # 피드백을 즉시 모델에 반영
        self._apply_feedback_immediately(feedback_item)
        
        # 피드백 시 즉시 학습 실행
        self._immediate_feedback_training(feedback_item)
    
    def _apply_feedback_immediately(self, feedback: Dict):
        """피드백을 즉시 모델에 반영합니다."""
        try:
            self.classifier.update_with_feedback(
                feedback['text'],
                feedback['correct_label'],
                feedback['is_correct'],
                feedback['memo']
            )
            
            # 피드백 데이터를 학습 데이터로도 추가
            if not feedback['is_correct']:
                self.add_training_data(
                    feedback['text'], 
                    feedback['correct_label'], 
                    "feedback_correction"
                )
            
            logger.info("피드백이 즉시 모델에 반영되었습니다")
            
        except Exception as e:
            logger.error(f"피드백 적용 중 오류가 발생했습니다: {e}")
    
    def _immediate_feedback_training(self, feedback: Dict):
        """피드백을 받은 즉시 학습을 실행합니다."""
        if self.is_training:
            logger.info("이미 학습이 진행 중이므로 피드백 학습을 대기합니다")
            return
        
        try:
            logger.info("피드백 기반 즉시 학습을 시작합니다")
            self.is_training = True
            
            # 피드백 데이터로 즉시 학습
            texts = [feedback['text']]
            labels = [feedback['correct_label']]
            
            # RTX 2080 최적화 설정
            batch_size = 4  # 피드백 학습용 작은 배치
            epochs = 1      # 피드백 학습용 1 에포크
            
            logger.info(f"피드백 데이터로 학습 중: {feedback['correct_label']}")
            
            # 즉시 학습 수행
            self.classifier.train(texts, labels, epochs=epochs, batch_size=batch_size)
            
            # 학습 완료 시간
            end_time = datetime.now()
            
            # 학습 결과 정보
            training_info = {
                'epochs': epochs,
                'batch_size': batch_size,
                'training_samples': len(texts),
                'unique_labels': len(set(labels)),
                'training_duration_seconds': 0,  # 즉시 학습이므로 짧음
                'start_time': feedback['timestamp'],
                'end_time': end_time.isoformat(),
                'labels': list(set(labels)),
                'data_sources': ['feedback_immediate'],
                'feedback_text': feedback['text'][:100] + "..." if len(feedback['text']) > 100 else feedback['text'],
                'feedback_memo': feedback['memo']
            }
            
            # 학습 히스토리 저장
            self._save_training_history(training_info)
            
            # 모델 저장
            self._save_model()
            
            logger.info("피드백 기반 즉시 학습이 완료되었습니다")
            
        except Exception as e:
            logger.error(f"피드백 즉시 학습 중 오류가 발생했습니다: {e}")
        finally:
            self.is_training = False
    
    def _trigger_auto_training(self):
        """자동 학습을 트리거합니다."""
        if not self.is_training and len(self.realtime_data) >= 5:
            logger.info("자동 학습을 시작합니다")
            self.start_training()
    
    def start_training(self, epochs: int = 2, batch_size: int = 4):
        """
        학습을 시작합니다.
        
        Args:
            epochs: 학습 에포크 수
            batch_size: 배치 크기
        """
        if self.is_training:
            logger.warning("이미 학습이 진행 중입니다")
            return
        
        self.is_training = True
        
        # 별도 스레드에서 학습 실행
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(epochs, batch_size)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        
        logger.info("학습이 시작되었습니다")
    
    def _training_worker(self, epochs: int, batch_size: int):
        """학습 작업을 수행하는 워커 스레드"""
        try:
            # 학습 데이터 준비
            texts = [item['text'] for item in self.realtime_data]
            labels = [item['label'] for item in item in self.realtime_data]
            
            if not texts:
                logger.warning("학습할 데이터가 없습니다")
                return
            
            # 학습 시작 시간
            start_time = datetime.now()
            
            logger.info(f"학습을 시작합니다. 데이터: {len(texts)}개, 에포크: {epochs}")
            
            # 모델 학습
            self.classifier.train(texts, labels, epochs=epochs, batch_size=batch_size)
            
            # 학습 완료 시간
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            # 학습 결과 정보
            training_info = {
                'epochs': epochs,
                'batch_size': batch_size,
                'training_samples': len(texts),
                'unique_labels': len(set(labels)),
                'training_duration_seconds': training_duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'labels': list(set(labels)),
                'data_sources': list(set(item['source'] for item in self.realtime_data))
            }
            
            # 학습 히스토리 저장
            self._save_training_history(training_info)
            
            # 모델 저장
            self._save_model()
            
            # 학습 데이터 초기화 (학습 완료 후)
            self.realtime_data.clear()
            
            logger.info(f"학습이 완료되었습니다. 소요 시간: {training_duration:.2f}초")
            
        except Exception as e:
            logger.error(f"학습 중 오류가 발생했습니다: {e}")
        finally:
            self.is_training = False
    
    def _save_model(self):
        """학습된 모델을 저장합니다."""
        # 버전별 모델 저장
        version = len(self.training_history)
        model_dir = self.studied_dir / f"direct_model_v{version}"
        
        # latest_model 심볼릭 링크 업데이트
        latest_model_dir = self.studied_dir / "latest_model"
        
        # 기존 latest_model 삭제
        if latest_model_dir.exists():
            if latest_model_dir.is_symlink():
                latest_model_dir.unlink()
            else:
                import shutil
                shutil.rmtree(latest_model_dir)
        
        # 새 모델 저장
        self.classifier.save_model(str(model_dir))
        
        # latest_model 심볼릭 링크 생성 (Windows에서는 복사)
        try:
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copytree(model_dir, latest_model_dir)
            else:  # Unix/Linux
                latest_model_dir.symlink_to(model_dir, target_is_directory=True)
        except Exception as e:
            logger.warning(f"심볼릭 링크 생성 실패, 복사로 대체: {e}")
            import shutil
            if latest_model_dir.exists():
                shutil.rmtree(latest_model_dir)
            shutil.copytree(model_dir, latest_model_dir)
        
        logger.info(f"직접 학습 모델이 저장되었습니다: {model_dir}")
    
    def _start_realtime_training(self):
        """실시간 학습을 시작합니다."""
        def realtime_training_loop():
            while True:
                try:
                    # 피드백 큐 처리
                    while self.feedback_queue:
                        feedback = self.feedback_queue.pop(0)
                        self._apply_feedback_immediately(feedback)
                    
                    # 자동 학습 트리거
                    if not self.is_training and len(self.realtime_data) >= 5:
                        self._trigger_auto_training()
                    
                    time.sleep(1)  # 1초마다 체크
                    
                except Exception as e:
                    logger.error(f"실시간 학습 루프 오류: {e}")
                    time.sleep(5)  # 오류 시 5초 대기
        
        # 실시간 학습 스레드 시작
        realtime_thread = threading.Thread(target=realtime_training_loop, daemon=True)
        realtime_thread.start()
        logger.info("실시간 학습이 시작되었습니다")
    
    def predict_text(self, text: str, top_k: int = 3) -> List[Dict]:
        """
        텍스트에 대한 예측을 수행합니다.
        
        Args:
            text: 예측할 텍스트
            top_k: 반환할 상위 결과 수
            
        Returns:
            예측 결과 리스트
        """
        return self.classifier.predict(text, top_k=top_k)
    
    def get_model_info(self) -> Dict:
        """현재 모델 정보를 반환합니다."""
        return {
            'model_name': self.classifier.model_name,
            'device': str(self.classifier.device),
            'num_labels': len(self.classifier.label_to_id),
            'labels': list(self.classifier.label_to_id.keys()),
            'training_history_count': len(self.training_history),
            'realtime_data_count': len(self.realtime_data),
            'feedback_queue_count': len(self.feedback_queue),
            'is_training': self.is_training,
            'last_training': self.training_history[-1] if self.training_history else None
        }
    
    def get_training_history(self) -> List[Dict]:
        """학습 히스토리를 반환합니다."""
        return self.training_history
    
    def get_realtime_data(self) -> List[Dict]:
        """실시간 학습 데이터를 반환합니다."""
        return self.realtime_data
    
    def get_feedback_queue(self) -> List[Dict]:
        """피드백 큐를 반환합니다."""
        return self.feedback_queue
    
    def clear_realtime_data(self):
        """실시간 학습 데이터를 초기화합니다."""
        self.realtime_data.clear()
        logger.info("실시간 학습 데이터가 초기화되었습니다")
    
    def clear_feedback_queue(self):
        """피드백 큐를 초기화합니다."""
        self.feedback_queue.clear()
        logger.info("피드백 큐가 초기화되었습니다")
    
    def export_model(self, export_path: str):
        """
        현재 모델을 지정된 경로로 내보냅니다.
        
        Args:
            export_path: 내보낼 경로
        """
        export_dir = Path(export_path)
        self.classifier.save_model(str(export_dir))
        logger.info(f"모델이 내보내졌습니다: {export_path}")
    
    def import_model(self, import_path: str):
        """
        지정된 경로에서 모델을 가져옵니다.
        
        Args:
            import_path: 가져올 모델 경로
        """
        if not os.path.exists(import_path):
            raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {import_path}")
        
        # 기존 모델 백업
        if hasattr(self, 'classifier'):
            backup_dir = self.studied_dir / f"direct_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.classifier.save_model(str(backup_dir))
            logger.info(f"기존 모델이 백업되었습니다: {backup_dir}")
        
        # 새 모델 로드
        self.classifier = XMLRoBERTaClassifier(model_dir=import_path)
        logger.info(f"모델이 가져와졌습니다: {import_path}")


def main():
    """메인 실행 함수"""
    try:
        # 직접 학습 관리자 초기화
        study_manager = DirectStudyManager()
        
        print("=== 직접 학습 시스템 시작 ===")
        print("실시간 학습 및 피드백 처리가 활성화되었습니다.")
        print("Ctrl+C로 종료할 수 있습니다.")
        
        # 모델 정보 출력
        while True:
            try:
                time.sleep(10)  # 10초마다 상태 출력
                
                model_info = study_manager.get_model_info()
                print(f"\n=== 상태 정보 ===")
                print(f"모델명: {model_info['model_name']}")
                print(f"레이블 수: {model_info['num_labels']}")
                print(f"학습 히스토리: {model_info['training_history_count']}회")
                print(f"실시간 데이터: {model_info['realtime_data_count']}개")
                print(f"피드백 큐: {model_info['feedback_queue_count']}개")
                print(f"학습 중: {model_info['is_training']}")
                
            except KeyboardInterrupt:
                print("\n\n시스템을 종료합니다...")
                break
            except Exception as e:
                logger.error(f"상태 출력 중 오류: {e}")
                time.sleep(5)
        
    except Exception as e:
        logger.error(f"실행 중 오류가 발생했습니다: {e}")
        raise


if __name__ == "__main__":
    main()
