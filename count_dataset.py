import os
from collections import defaultdict

root_dir = '/home/ohjihoon/바탕화면/app/datasets'

# 모든 클래스 폴더 순회
for class_name in sorted(os.listdir(root_dir)):
    class_path = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f'\n{class_name}')  # 클래스 이름 출력

    # 불량 유형별 이미지 개수를 저장할 딕셔너리
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)

    # 훈련 이미지 수 세기
    train_dir = os.path.join(class_path, 'train', 'images')
    if os.path.exists(train_dir):
        for fname in os.listdir(train_dir):
            if not fname.endswith('.png'):
                continue
            defect_type = '_'.join(fname.split('_')[:-1])  # 예: broken_large
            train_counts[defect_type] += 1

    # 테스트 이미지 수 세기
    test_dir = os.path.join(class_path, 'test', 'images')
    if os.path.exists(test_dir):
        for fname in os.listdir(test_dir):
            if not fname.endswith('.png'):
                continue
            defect_type = '_'.join(fname.split('_')[:-1])
            test_counts[defect_type] += 1

    # 모든 불량 유형 출력
    all_defect_types = set(train_counts.keys()) | set(test_counts.keys())
    for defect_type in sorted(all_defect_types):
        train_count = train_counts.get(defect_type, 0)
        test_count = test_counts.get(defect_type, 0)
        print(f'{defect_type} : ({train_count}), ({test_count})')