#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.ananke.model import XMLRoBERTaClassifier
import torch

def test_prediction():
    print("=== 저장된 모델로 예측 테스트 ===")
    
    # 저장된 모델 로드
    model_dir = "../data/studied/model_v1"
    classifier = XMLRoBERTaClassifier(model_dir=model_dir)
    
    # 테스트 예측
    test_texts = [
        "Gasoline Petrol Motor fuel",
        "Coal Tar",
        "Iron Ore Hematite",
        "Natural Gas Methane"
    ]
    
    for text in test_texts:
        print(f"\n테스트 텍스트: {text}")
        try:
            predictions = classifier.predict(text)
            print(f"예측 결과 수: {len(predictions)}")
            
            for pred in predictions:
                print(f"  {pred['rank']}등: {pred['label']} ({pred['similarity']:.1f}%)")
        except Exception as e:
            print(f"예측 오류: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
