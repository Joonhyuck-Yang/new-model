'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';

interface Prediction {
  rank: number;
  label: string;
  similarity: number;
}

interface ModelInfo {
  status: string;
  model_name?: string;
  num_labels?: number;
  labels?: string[];
  device?: string;
}

interface TrainingStatus {
  status: string;
  message: string;
  progress: number;
}

export default function Home() {
  const [inputText, setInputText] = useState('');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo>({ status: 'loading' });
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({ status: 'idle', message: '', progress: 0 });
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackLabel, setFeedbackLabel] = useState('');
  const [feedbackMemo, setFeedbackMemo] = useState('');
  const [isCorrect, setIsCorrect] = useState(true);

  // 모델 정보 로드
  useEffect(() => {
    loadModelInfo();
    const interval = setInterval(loadModelInfo, 5000); // 5초마다 업데이트
    return () => clearInterval(interval);
  }, []);

  // 학습 상태 모니터링
  useEffect(() => {
    const interval = setInterval(loadTrainingStatus, 1000); // 1초마다 업데이트
    return () => clearInterval(interval);
  }, []);

  const loadModelInfo = async () => {
    try {
      const response = await axios.get('/api/model-info');
      setModelInfo(response.data);
    } catch (error) {
      console.error('모델 정보 로드 실패:', error);
    }
  };

  const loadTrainingStatus = async () => {
    try {
      const response = await axios.get('/api/training-status');
      setTrainingStatus(response.data);
    } catch (error) {
      console.error('학습 상태 로드 실패:', error);
    }
  };

  const handlePredict = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    try {
      const response = await axios.post('/api/predict', { text: inputText });
      setPredictions(response.data.predictions);
      setShowFeedback(true);
    } catch (error) {
      console.error('예측 실패:', error);
      alert('예측 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  // 학습 시작 함수 제거 - 피드백 시 자동 학습

  const handleFeedback = async () => {
    if (!feedbackLabel.trim()) {
      alert('올바른 라벨을 입력해주세요.');
      return;
    }

    try {
      // 피드백 제출
      await axios.post('/api/feedback', {
        text: inputText,
        correct_label: feedbackLabel,
        is_correct: isCorrect,
        memo: feedbackMemo
      });

      // 즉시 학습 완료까지 대기
      let attempts = 0;
      const maxAttempts = 30; // 최대 30초 대기
      
      const waitForTraining = async () => {
        while (attempts < maxAttempts) {
          try {
            const statusResponse = await axios.get('/api/training-status');
            if (statusResponse.data.status === 'completed') {
              alert('피드백이 반영되고 즉시 학습이 완료되었습니다! 🎉');
              break;
            } else if (statusResponse.data.status === 'error') {
              alert('학습 중 오류가 발생했습니다.');
              break;
            }
            // 1초 대기
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
          } catch (error) {
            console.error('학습 상태 확인 실패:', error);
            break;
          }
        }
        
        if (attempts >= maxAttempts) {
          alert('학습이 시간 초과되었습니다. 잠시 후 다시 시도해주세요.');
        }
      };

      // 즉시 학습 완료 대기 시작
      waitForTraining();

      setShowFeedback(false);
      setFeedbackLabel('');
      setFeedbackMemo('');
      setPredictions([]);
      
      // 모델 정보 새로고침
      setTimeout(() => {
        loadTrainingStatus();
        loadModelInfo();
      }, 1000);
      
    } catch (error) {
      console.error('피드백 제출 실패:', error);
      alert('피드백 제출에 실패했습니다.');
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
        XML-RoBERTa 학습 및 추론 시스템
      </h1>

      {/* 모델 정보 */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-2xl font-semibold mb-4">모델 정보</h2>
        {modelInfo.status === 'loading' ? (
          <div className="loading-spinner mx-auto"></div>
        ) : modelInfo.status === 'ready' ? (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p><strong>모델명:</strong> {modelInfo.model_name}</p>
              <p><strong>레이블 수:</strong> {modelInfo.num_labels}</p>
              <p><strong>디바이스:</strong> {modelInfo.device}</p>
            </div>
            <div>
              <p><strong>레이블:</strong></p>
              <div className="max-h-32 overflow-y-auto">
                {modelInfo.labels?.map((label, index) => (
                  <span key={index} className="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm mr-2 mb-1">
                    {label}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <p className="text-red-600">모델이 초기화되지 않았습니다.</p>
        )}
      </div>

      {/* 학습 상태 */}
      {trainingStatus.status !== 'idle' && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-2xl font-semibold mb-4">
            {trainingStatus.status === 'training' ? '🔄 학습 진행 중' : '✅ 학습 완료'}
          </h2>
          <div className="mb-2">
            <p className="text-gray-700">{trainingStatus.message}</p>
          </div>
          {trainingStatus.status === 'training' && (
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
                style={{ width: `${trainingStatus.progress}%` }}
              ></div>
            </div>
          )}
          <p className="text-sm text-gray-500 mt-2">
            {trainingStatus.status === 'training' ? `진행률: ${trainingStatus.progress}%` : '학습이 완료되었습니다'}
          </p>
        </div>
      )}

      {/* 학습 시작 섹션 제거 - 피드백 시 자동 학습 */}

      {/* 텍스트 입력 및 예측 */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h2 className="text-2xl font-semibold mb-4">텍스트 분류</h2>
        <div className="mb-4">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="분류할 텍스트를 입력하세요..."
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={4}
          />
        </div>
        <button
          onClick={handlePredict}
          disabled={!inputText.trim() || isLoading || modelInfo.status !== 'ready'}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-2 px-4 rounded transition-colors"
        >
          {isLoading ? '분석 중...' : '분류하기'}
        </button>
      </div>

      {/* 예측 결과 */}
      {predictions.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-2xl font-semibold mb-4">분류 결과</h2>
          <div className="space-y-3">
            {predictions.map((prediction) => (
              <div key={prediction.rank} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <span className="bg-blue-600 text-white px-3 py-1 rounded-full text-sm font-bold mr-3">
                    {prediction.rank}등
                  </span>
                  <span className="text-lg font-medium">{prediction.label}</span>
                </div>
                <span className="text-blue-600 font-bold">
                  {prediction.similarity.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 피드백 입력 */}
      {showFeedback && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-semibold mb-4">피드백</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                예측이 올바른가요?
              </label>
              <div className="flex space-x-4">
                <label className="flex items-center">
                  <input
                    type="radio"
                    checked={isCorrect}
                    onChange={() => setIsCorrect(true)}
                    className="mr-2"
                  />
                  맞습니다
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    checked={!isCorrect}
                    onChange={() => setIsCorrect(false)}
                    className="mr-2"
                  />
                  틀렸습니다
                </label>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                올바른 라벨
              </label>
              <input
                type="text"
                value={feedbackLabel}
                onChange={(e) => setFeedbackLabel(e.target.value)}
                placeholder="올바른 라벨을 입력하세요"
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {!isCorrect && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  메모 (선택사항)
                </label>
                <textarea
                  value={feedbackMemo}
                  onChange={(e) => setFeedbackMemo(e.target.value)}
                  placeholder="맥락이나 설명을 입력하세요..."
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={3}
                />
              </div>
            )}

            <div className="flex space-x-3">
              <button
                onClick={handleFeedback}
                className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition-colors"
              >
                피드백 제출
              </button>
              <button
                onClick={() => setShowFeedback(false)}
                className="bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded transition-colors"
              >
                취소
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
