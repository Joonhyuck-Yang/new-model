"""
비동기 병렬 학습 모듈

RTX 4080의 16GB VRAM을 최대한 활용하여 
여러 학습 프로세스를 병렬로 실행합니다.
"""

import asyncio
import os
import sys
import json
import torch
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import time

# 상위 디렉토리의 모델 클래스 import
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.ananke.model import XMLRoBERTaClassifier
from app.local_study.main import LocalStudyManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncTrainingManager:
    """비동기 병렬 학습 관리자"""
    
    def __init__(self, max_concurrent_jobs: int = None):
        """
        Args:
            max_concurrent_jobs: 최대 동시 실행 작업 수 (None이면 자동 계산)
        """
        self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        
        # RTX 4080 (16GB) 기준 최적화
        if max_concurrent_jobs is None:
            if self.gpu_memory_gb >= 15:  # RTX 4080
                # 각 학습 프로세스가 약 4-5GB 사용하므로 3개 병렬 실행
                self.max_concurrent_jobs = 3
            elif self.gpu_memory_gb >= 10:
                self.max_concurrent_jobs = 2
            else:
                self.max_concurrent_jobs = 1
        else:
            self.max_concurrent_jobs = max_concurrent_jobs
            
        logger.info(f"GPU 메모리: {self.gpu_memory_gb:.1f}GB, 최대 동시 작업: {self.max_concurrent_jobs}개")
        
        self.current_jobs = []
        self.completed_jobs = []
        self.job_lock = threading.Lock()
        
    async def train_single_batch_async(self, job_id: int, config: Dict) -> Dict:
        """단일 배치 비동기 학습"""
        try:
            logger.info(f"🚀 Job {job_id} 시작: epochs={config['epochs']}, batch_size={config['batch_size']}")
            
            # GPU 메모리 할당을 위한 지연
            await asyncio.sleep(job_id * 2)  # 각 작업마다 2초 간격으로 시작
            
            # 새 프로세스에서 학습 실행
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(
                    executor, 
                    self._run_training_process, 
                    job_id, 
                    config
                )
            
            logger.info(f"✅ Job {job_id} 완료: Loss={result.get('final_loss', 'N/A')}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Job {job_id} 실패: {e}")
            return {"job_id": job_id, "status": "failed", "error": str(e)}
    
    def _run_training_process(self, job_id: int, config: Dict) -> Dict:
        """별도 프로세스에서 실행되는 학습 함수"""
        try:
            # CUDA 디바이스 설정 (각 프로세스마다 다른 CUDA 컨텍스트)
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            
            # 학습 관리자 초기화
            study_manager = LocalStudyManager()
            
            # 학습 데이터 준비
            texts, labels = study_manager.prepare_training_data()
            
            # 학습 시작 시간
            start_time = time.time()
            
            # 모델 학습
            training_result = study_manager.train_model(
                texts, 
                labels, 
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                learning_rate=config.get('learning_rate', 2e-5)
            )
            
            # 학습 완료 시간
            end_time = time.time()
            duration = end_time - start_time
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            
            return {
                "job_id": job_id,
                "status": "completed",
                "duration": duration,
                "training_result": training_result,
                "config": config,
                "final_loss": "N/A",  # 실제 로스는 학습 로그에서 추출 가능
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"프로세스 {job_id} 오류: {e}")
            return {
                "job_id": job_id,
                "status": "failed", 
                "error": str(e),
                "config": config,
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_parallel_training(self, training_configs: List[Dict]) -> List[Dict]:
        """병렬 학습 실행"""
        logger.info(f"🎯 {len(training_configs)}개 작업을 {self.max_concurrent_jobs}개씩 병렬 실행")
        
        # 세마포어로 동시 실행 작업 수 제한
        semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        async def limited_train(job_id: int, config: Dict):
            async with semaphore:
                return await self.train_single_batch_async(job_id, config)
        
        # 모든 작업을 비동기로 실행
        tasks = [
            limited_train(i, config) 
            for i, config in enumerate(training_configs)
        ]
        
        # 모든 작업 완료까지 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 정리
        completed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"작업 실패: {result}")
                completed_results.append({"status": "failed", "error": str(result)})
            else:
                completed_results.append(result)
        
        return completed_results
    
    def generate_training_configs(self, target_loss: float = 2.0) -> List[Dict]:
        """다양한 하이퍼파라미터 조합으로 학습 설정 생성"""
        configs = []
        
        # RTX 4080 최적화된 설정들
        base_configs = [
            # 고속 학습 (작은 에포크, 큰 배치)
            {"epochs": 10, "batch_size": 128, "learning_rate": 5e-5, "name": "빠른학습"},
            {"epochs": 15, "batch_size": 96, "learning_rate": 3e-5, "name": "중간학습1"},
            {"epochs": 20, "batch_size": 64, "learning_rate": 2e-5, "name": "표준학습"},
            
            # 정밀 학습 (많은 에포크, 작은 배치)
            {"epochs": 30, "batch_size": 32, "learning_rate": 1e-5, "name": "정밀학습1"},
            {"epochs": 25, "batch_size": 48, "learning_rate": 1.5e-5, "name": "정밀학습2"},
            
            # 실험적 설정
            {"epochs": 40, "batch_size": 24, "learning_rate": 8e-6, "name": "느린정밀학습"},
        ]
        
        return base_configs
    
    async def continuous_training_until_target(self, target_loss: float = 2.0, max_iterations: int = 10):
        """목표 로스까지 계속 학습"""
        logger.info(f"🎯 목표 로스 {target_loss}까지 최대 {max_iterations}회 반복 학습")
        
        iteration = 0
        best_loss = float('inf')
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"\n🔄 반복 {iteration}/{max_iterations} 시작")
            
            # 학습 설정 생성
            training_configs = self.generate_training_configs(target_loss)
            
            # 병렬 학습 실행
            results = await self.run_parallel_training(training_configs)
            
            # 결과 분석
            successful_results = [r for r in results if r.get("status") == "completed"]
            
            if successful_results:
                logger.info(f"✅ {len(successful_results)}개 작업 완료")
                
                # 최고 성능 확인 (실제로는 로그 파일에서 로스 값을 읽어와야 함)
                for result in successful_results:
                    config_name = result['config'].get('name', 'Unknown')
                    duration = result.get('duration', 0)
                    logger.info(f"  📊 {config_name}: {duration:.1f}초 소요")
                
                # 잠시 대기 (GPU 쿨다운)
                await asyncio.sleep(10)
            else:
                logger.warning("⚠️ 모든 작업이 실패했습니다")
                break
        
        logger.info(f"🎉 총 {iteration}회 반복 학습 완료")


async def main():
    """메인 비동기 실행 함수"""
    try:
        logger.info("🚀 비동기 병렬 학습 시작")
        
        # GPU 확인
        if not torch.cuda.is_available():
            logger.error("❌ CUDA를 사용할 수 없습니다")
            return
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🖥️ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 비동기 학습 관리자 초기화
        training_manager = AsyncTrainingManager()
        
        # 지속적 학습 실행
        await training_manager.continuous_training_until_target(target_loss=2.0, max_iterations=5)
        
    except Exception as e:
        logger.error(f"실행 중 오류: {e}")
        raise


def run_async_training():
    """비동기 학습 실행 wrapper"""
    try:
        # Windows에서 multiprocessing 이슈 해결
        if sys.platform.startswith('win'):
            mp.set_start_method('spawn', force=True)
        
        # 이벤트 루프 실행
        asyncio.run(main())
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다")
    except Exception as e:
        logger.error(f"실행 실패: {e}")


if __name__ == "__main__":
    run_async_training()
