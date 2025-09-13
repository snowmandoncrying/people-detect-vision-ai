from ultralytics import YOLO
import numpy as np

print("=== 포즈 키포인트 테스트 ===")

# 포즈 모델 로드
pose_model = YOLO('yolo11n-pose.pt')

# 테스트 이미지들 (사람이 다양한 포즈)
test_images = [
    'https://ultralytics.com/images/bus.jpg',  # 여러 사람
    'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b',  # 운동하는 사람
]

for i, img_url in enumerate(test_images):
    print(f"\n--- 이미지 {i+1} 분석 ---")
    results = pose_model(img_url)

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        print(f"감지된 사람 수: {len(keypoints)}")

        # 각 사람별 키포인트 확인
        for person_idx, person_keypoints in enumerate(keypoints):
            print(f"사람 {person_idx + 1}:")
            print(f"  코 위치: {person_keypoints[0]}")      # 0: 코
            print(f"  왼쪽 어깨: {person_keypoints[5]}")     # 5: 왼쪽 어깨
            print(f"  오른쪽 어깨: {person_keypoints[6]}")   # 6: 오른쪽 어깨
            print(f"  왼쪽 손목: {person_keypoints[9]}")     # 9: 왼쪽 손목
            print(f"  오른쪽 손목: {person_keypoints[10]}")  # 10: 오른쪽 손목

    # 결과 이미지 표시
    results[0].show()

print("\n=== 키포인트 테스트 완료 ===")