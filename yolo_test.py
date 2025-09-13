from ultralytics import YOLO

# 모델 자동 다운로드됨
model = YOLO('yolo11n.pt')

# YOLO에서 제공하는 테스트 이미지 (여러 사람 있음)
test_url = 'https://ultralytics.com/images/bus.jpg'

# 바로 실행!
results = model(test_url)
results[0].show()  # 결과 창 자동으로 뜸

print(f"감지된 객체 수: {len(results[0].boxes)}")