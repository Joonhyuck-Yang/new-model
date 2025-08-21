"""
로컬 학습 모듈

이 모듈은 XML-RoBERTa 모델의 로컬 학습을 담당합니다.
엑셀 파일을 JSONL로 변환하고, 모델을 학습시키며, 
학습된 내용을 누적하여 저장하는 기능을 제공합니다.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

# 상위 디렉토리의 모델 클래스 import
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.ananke.model import XMLRoBERTaClassifier
from app.ananke.excel_to_jsonl import convert_excel_to_jsonl

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalStudyManager:
    """로컬 학습 관리자 클래스"""
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        초기화
        
        Args:
            model_dir: 기존 학습된 모델이 있는 디렉토리 경로
        """
        self.current_dir = Path(__file__).parent.parent.parent.parent
        self.data_dir = self.current_dir / "data"
        self.studied_dir = self.data_dir / "studied"
        self.dataforstudy_dir = self.data_dir / "dataforstudy"
        
        # 디렉토리 생성
        self.studied_dir.mkdir(exist_ok=True)
        
        # 모델 초기화
        if model_dir and os.path.exists(model_dir):
            logger.info(f"기존 모델을 로드합니다: {model_dir}")
            self.classifier = XMLRoBERTaClassifier(model_dir=model_dir)
        else:
            logger.info("새로운 모델을 초기화합니다")
            self.classifier = XMLRoBERTaClassifier()
        
        # 학습 히스토리
        self.training_history = self._load_training_history()
        
    def _load_training_history(self) -> List[Dict]:
        """학습 히스토리를 로드합니다."""
        history_file = self.studied_dir / "training_history.jsonl"
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
        history_file = self.studied_dir / "training_history.jsonl"
        
        # 타임스탬프 추가
        training_info['timestamp'] = datetime.now().isoformat()
        training_info['model_version'] = len(self.training_history) + 1
        
        self.training_history.append(training_info)
        
        # JSONL 파일에 저장
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(training_info, ensure_ascii=False) + '\n')
        
        logger.info(f"학습 히스토리가 저장되었습니다: {training_info['model_version']}번째 학습")
    
    def prepare_training_data(self, excel_filename: str = "학습 데이터셋.xlsx") -> Tuple[List[str], List[str]]:
        """
        엑셀 파일을 JSONL로 변환하고 학습 데이터를 준비합니다.
        
        Args:
            excel_filename: 엑셀 파일명
            
        Returns:
            texts: 텍스트 리스트
            labels: 라벨 리스트
        """
        excel_path = self.dataforstudy_dir / excel_filename
        jsonl_path = self.dataforstudy_dir / "training_data.jsonl"
        
        if not excel_path.exists():
            raise FileNotFoundError(f"엑셀 파일을 찾을 수 없습니다: {excel_path}")
        
        logger.info("엑셀 파일을 JSONL로 변환 중...")
        success = convert_excel_to_jsonl(excel_path, jsonl_path)
        
        if not success:
            raise RuntimeError("엑셀 파일 변환에 실패했습니다")
        
        logger.info("학습 데이터 준비 중...")
        texts, labels = self.classifier.prepare_training_data(jsonl_path)
        
        logger.info(f"총 {len(texts)}개의 학습 데이터가 준비되었습니다")
        return texts, labels
    
    def train_model(self, 
                   texts: List[str], 
                   labels: List[str], 
                   epochs: int = 3, 
                   batch_size: int = 8,
                   learning_rate: float = 2e-5) -> Dict:
        """
        모델을 학습합니다.
        
        Args:
            texts: 학습 텍스트 리스트
            labels: 학습 라벨 리스트
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            
        Returns:
            학습 결과 정보
        """
        logger.info(f"모델 학습을 시작합니다. 에포크: {epochs}, 배치 크기: {batch_size}")
        
        # 학습 시작 시간
        start_time = datetime.now()
        
        try:
            # 모델 학습
            self.classifier.train(texts, labels, epochs=epochs, batch_size=batch_size)
            
            # 학습 완료 시간
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            # 학습 결과 정보
            training_info = {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'training_samples': len(texts),
                'unique_labels': len(set(labels)),
                'training_duration_seconds': training_duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'labels': list(set(labels))
            }
            
            # 학습 히스토리 저장
            self._save_training_history(training_info)
            
            # 모델 저장
            self._save_model()
            
            logger.info(f"모델 학습이 완료되었습니다. 소요 시간: {training_duration:.2f}초")
            return training_info
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류가 발생했습니다: {e}")
            raise
    
    def _save_model(self):
        """학습된 모델을 저장합니다."""
        # 버전별 모델 저장
        version = len(self.training_history)
        model_dir = self.studied_dir / f"model_v{version}"
        
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
        
        logger.info(f"모델이 저장되었습니다: {model_dir}")
    
    def get_model_info(self) -> Dict:
        """현재 모델 정보를 반환합니다."""
        return {
            'model_name': self.classifier.model_name,
            'device': str(self.classifier.device),
            'num_labels': len(self.classifier.label_to_id),
            'labels': list(self.classifier.label_to_id.keys()),
            'training_history_count': len(self.training_history),
            'last_training': self.training_history[-1] if self.training_history else None
        }
    
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
    
    def update_with_feedback(self, text: str, correct_label: str, is_correct: bool, memo: str = ""):
        """
        사용자 피드백을 바탕으로 모델을 업데이트합니다.
        
        Args:
            text: 원본 텍스트
            correct_label: 올바른 라벨
            is_correct: 예측이 올바른지 여부
            memo: 추가 메모
        """
        logger.info(f"피드백을 받았습니다: {correct_label} (정확: {is_correct})")
        
        # 모델 업데이트
        self.classifier.update_with_feedback(text, correct_label, is_correct, memo)
        
        # 업데이트된 모델 저장
        self._save_model()
        
        logger.info("피드백이 모델에 반영되었습니다")
    
    def get_training_history(self) -> List[Dict]:
        """학습 히스토리를 반환합니다."""
        return self.training_history
    
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
            backup_dir = self.studied_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.classifier.save_model(str(backup_dir))
            logger.info(f"기존 모델이 백업되었습니다: {backup_dir}")
        
        # 새 모델 로드
        self.classifier = XMLRoBERTaClassifier(model_dir=import_path)
        logger.info(f"모델이 가져와졌습니다: {import_path}")


def main():
    """메인 실행 함수"""
    try:
        # 로컬 학습 관리자 초기화
        study_manager = LocalStudyManager()
        
        # 모델 정보 출력
        model_info = study_manager.get_model_info()
        print("=== 모델 정보 ===")
        print(f"모델명: {model_info['model_name']}")
        print(f"디바이스: {model_info['device']}")
        print(f"레이블 수: {model_info['num_labels']}")
        print(f"학습 히스토리: {model_info['training_history_count']}회")
        
        # 학습 데이터 준비
        print("\n=== 학습 데이터 준비 ===")
        texts, labels = study_manager.prepare_training_data()
        
        # 모델 학습
        print("\n=== 모델 학습 시작 ===")
        training_result = study_manager.train_model(texts, labels, epochs=3, batch_size=8)
        
        print(f"\n=== 학습 완료 ===")
        print(f"학습 시간: {training_result['training_duration_seconds']:.2f}초")
        print(f"학습 샘플 수: {training_result['training_samples']}")
        print(f"고유 라벨 수: {training_result['unique_labels']}")
        
        # 테스트 예측
        if texts:
            test_text = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
            print(f"\n=== 테스트 예측 ===")
            print(f"테스트 텍스트: {test_text}")
            predictions = study_manager.predict_text(test_text)
            for pred in predictions:
                print(f"{pred['rank']}등: {pred['label']} ({pred['similarity']:.1f}%)")
        
    except Exception as e:
        logger.error(f"실행 중 오류가 발생했습니다: {e}")
        raise


if __name__ == "__main__":
    main()
