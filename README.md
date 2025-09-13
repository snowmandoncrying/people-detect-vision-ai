# people-detect-vision-ai
이미지에서 사람 감지 (성별, 연령대, 동작, 감정)

# 비전 AI 프로젝트

## 목표
사진 속 여러 사람의 속성을 분석하여 자연어로 설명하는 AI 시스템

**입력:** 사진 (여러 사람)
**출력:** "성인 남성 1명이 서서 만세를 하며 웃고 있고, 아동 여성 1명이 앉아서 팔짱을 끼고 화를 내고 있습니다."

## 기술 스택
- YOLO11 (사람 감지 & 포즈 추출)
- PyTorch (커스텀 분류 모델)
- OpenCV (이미지 처리)
- Intel Arc GPU (17.9GB VRAM)

## 개발 진행도
- ✅ 사람 감지 테스트 완료
- ⏳ 포즈 키포인트 추출 테스트
- 🔜 동작 분류 로직 개발
- 🔜 속성 분류 모델 학습
- 🔜 통합 파이프라인 구축

## 실행 방법
```bash
pip install ultralytics opencv-python
python yolo_test.py
