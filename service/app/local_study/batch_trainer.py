"""
배치 학습기 - RTX 4080 최적화

여러 설정으로 연속 학습을 실행하여 최적의 결과를 찾습니다.
"""

import os
import sys
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

# 상위 디렉토리의 모델 클래스 import
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.local_study.main import LocalStudyManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTX4080BatchTrainer:
    """RTX 4080 최적화 배치 학습기"""
    
    def __init__(self):
        self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        logger.info(f"🖥️ GPU 메모리: {self.gpu_memory_gb:.1f}GB")
        
        # RTX 4080 최적화 설정들
        self.training_configs = [
            # 초고속 대용량 배치
            {"name": "⚡초고속", "epochs": 30, "batch_size": 256, "learning_rate": 1e-4},
            {"name": "🚀고속1", "epochs": 25, "batch_size": 192, "learning_rate": 8e-5},
            {"name": "🚀고속2", "epochs": 35, "batch_size": 128, "learning_rate": 5e-5},
            
            # 최적화된 표준 설정
            {"name": "📊표준1", "epochs": 40, "batch_size": 96, "learning_rate": 3e-5},
            {"name": "📊표준2", "epochs": 45, "batch_size": 80, "learning_rate": 2.5e-5},
            {"name": "📊표준3", "epochs": 50, "batch_size": 64, "learning_rate": 2e-5},
            
            # 정밀 학습
            {"name": "🎯정밀1", "epochs": 60, "batch_size": 48, "learning_rate": 1.5e-5},
            {"name": "🎯정밀2", "epochs": 70, "batch_size": 32, "learning_rate": 1e-5},
        ]
    
    def run_single_training(self, config: Dict) -> Dict:
        """단일 학습 실행"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 {config['name']} 학습 시작")
            logger.info(f"   📊 Epochs: {config['epochs']}")
            logger.info(f"   📦 Batch Size: {config['batch_size']}")
            logger.info(f"   📚 Learning Rate: {config['learning_rate']}")
            logger.info(f"{'='*60}")
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            # 학습 시작 시간
            start_time = time.time()
            
            # 학습 관리자 초기화
            study_manager = LocalStudyManager()
            
            # 학습 데이터 준비
            texts, labels = study_manager.prepare_training_data()
            
            # 모델 학습
            training_result = study_manager.train_model(
                texts, 
                labels, 
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate']
            )
            
            # 학습 완료 시간
            end_time = time.time()
            duration = end_time - start_time
            
            # 간단한 테스트 예측
            test_result = None
            if texts:
                test_text = "Gasoline"  # 고정 테스트 텍스트
                predictions = study_manager.predict_text(test_text)
                test_result = {
                    "test_text": test_text,
                    "predictions": predictions[:3]  # 상위 3개만
                }
            
            result = {
                "name": config['name'],
                "config": config,
                "duration": duration,
                "training_result": training_result,
                "test_result": test_result,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            logger.info(f"✅ {config['name']} 완료: {duration:.1f}초 소요")
            if test_result:
                logger.info(f"   🧪 테스트 결과: {test_result['predictions'][0]['label']} ({test_result['predictions'][0]['similarity']:.1f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {config['name']} 실패: {e}")
            return {
                "name": config['name'],
                "config": config,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_batch_training(self, target_iterations: int = 5, cooldown_seconds: int = 30):
        """배치 학습 실행"""
        logger.info(f"🚀 RTX 4080 배치 학습 시작")
        logger.info(f"   📋 총 {len(self.training_configs)}개 설정")
        logger.info(f"   🔄 {target_iterations}회 반복")
        logger.info(f"   ⏱️ 쿨다운: {cooldown_seconds}초")
        
        all_results = []
        
        for iteration in range(1, target_iterations + 1):
            logger.info(f"\n🔄 ========== 반복 {iteration}/{target_iterations} ==========")
            
            iteration_results = []
            
            for i, config in enumerate(self.training_configs, 1):
                logger.info(f"\n📍 [{iteration}-{i}/{len(self.training_configs)}] {config['name']} 실행 중...")
                
                # 학습 실행
                result = self.run_single_training(config)
                iteration_results.append(result)
                all_results.append(result)
                
                # 중간 쿨다운 (마지막 설정이 아닌 경우)
                if i < len(self.training_configs):
                    logger.info(f"⏸️ {cooldown_seconds//2}초 쿨다운...")
                    time.sleep(cooldown_seconds // 2)
            
            # 반복 완료 후 결과 요약
            successful = [r for r in iteration_results if r['status'] == 'completed']
            failed = [r for r in iteration_results if r['status'] == 'failed']
            
            logger.info(f"\n📊 반복 {iteration} 결과:")
            logger.info(f"   ✅ 성공: {len(successful)}개")
            logger.info(f"   ❌ 실패: {len(failed)}개")
            
            if successful:
                fastest = min(successful, key=lambda x: x['duration'])
                logger.info(f"   ⚡ 가장 빠른 학습: {fastest['name']} ({fastest['duration']:.1f}초)")
            
            # 반복 간 쿨다운
            if iteration < target_iterations:
                logger.info(f"⏸️ 반복 간 쿨다운 {cooldown_seconds}초...")
                time.sleep(cooldown_seconds)
        
        # 최종 결과 요약
        self.print_final_summary(all_results)
        return all_results
    
    def print_final_summary(self, results: List[Dict]):
        """최종 결과 요약 출력"""
        logger.info(f"\n🎉 ========== 최종 결과 요약 ==========")
        
        successful = [r for r in results if r['status'] == 'completed']
        failed = [r for r in results if r['status'] == 'failed']
        
        logger.info(f"📊 전체 통계:")
        logger.info(f"   ✅ 성공: {len(successful)}개")
        logger.info(f"   ❌ 실패: {len(failed)}개")
        logger.info(f"   📈 성공률: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            # 최고 성능 찾기
            fastest = min(successful, key=lambda x: x['duration'])
            slowest = max(successful, key=lambda x: x['duration'])
            avg_duration = sum(r['duration'] for r in successful) / len(successful)
            
            logger.info(f"\n⚡ 성능 분석:")
            logger.info(f"   🏃 가장 빠른 학습: {fastest['name']} ({fastest['duration']:.1f}초)")
            logger.info(f"   🐌 가장 느린 학습: {slowest['name']} ({slowest['duration']:.1f}초)")
            logger.info(f"   📊 평균 학습 시간: {avg_duration:.1f}초")
            
            # 설정별 성공률
            config_stats = {}
            for result in results:
                config_name = result['name']
                if config_name not in config_stats:
                    config_stats[config_name] = {'total': 0, 'success': 0}
                config_stats[config_name]['total'] += 1
                if result['status'] == 'completed':
                    config_stats[config_name]['success'] += 1
            
            logger.info(f"\n📋 설정별 성공률:")
            for config_name, stats in config_stats.items():
                success_rate = stats['success'] / stats['total'] * 100
                logger.info(f"   {config_name}: {success_rate:.1f}% ({stats['success']}/{stats['total']})")
        
        logger.info(f"={'='*50}")


def main():
    """메인 실행 함수"""
    try:
        # GPU 확인
        if not torch.cuda.is_available():
            logger.error("❌ CUDA를 사용할 수 없습니다")
            return
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🖥️ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        if "4080" not in gpu_name and gpu_memory < 15:
            logger.warning("⚠️ RTX 4080이 아닙니다. 설정을 조정할 수 있습니다.")
        
        # 배치 학습기 초기화 및 실행
        trainer = RTX4080BatchTrainer()
        results = trainer.run_batch_training(target_iterations=3, cooldown_seconds=20)
        
        logger.info("🎉 모든 학습이 완료되었습니다!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중단되었습니다")
    except Exception as e:
        logger.error(f"❌ 실행 실패: {e}")
        raise


if __name__ == "__main__":
    main()
