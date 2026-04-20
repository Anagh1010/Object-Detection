import os
import sys
import torch
import joblib
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ── ✏️ UPDATE THESE PATHS ─────────────────────────────────────────────────
SAVE_DIR   = 'saved_model'        
TORCH_HOME = 'venv/torch_models'  
# ─────────────────────────────────────────────────────────────────────────

os.environ['TORCH_HOME'] = TORCH_HOME

def load_model():
    clf     = joblib.load(os.path.join(SAVE_DIR, 'classifier.pkl'))
    scaler  = joblib.load(os.path.join(SAVE_DIR, 'scaler.pkl'))
    le      = joblib.load(os.path.join(SAVE_DIR, 'label_encoder.pkl'))

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return clf, scaler, le, resnet, transform, device


def detect_objects(img_path, confidence_threshold=0.8, min_votes=4):
    if not os.path.exists(img_path):
        print(f'Error: File not found — {img_path}')
        return []

    clf, scaler, le, resnet, transform, device = load_model()

    img  = Image.open(img_path).convert('RGB')
    W, H = img.size
    print(f'Image: {img_path}  ({W}x{H})')

    # Reverted to simpler scales to prevent weird background slices
    window_sizes = [
        (int(W * 0.8), int(H * 0.8)),
        (int(W * 0.6), int(H * 0.6)),
        (int(W * 0.4), int(H * 0.4)),
    ]

    crops = []
    for (ww, wh) in window_sizes:
        stride_x = max(1, int(ww * 0.3)) # Tighter stride for more overlapping votes
        stride_y = max(1, int(wh * 0.3))
        for y in range(0, H - wh + 1, stride_y):
            for x in range(0, W - ww + 1, stride_x):
                crops.append(img.crop((x, y, x + ww, y + wh)))

    print(f'Scanning {len(crops)} windows for consensus...')

    embeddings = []
    with torch.no_grad():
        for crop in crops:
            tensor    = transform(crop).unsqueeze(0).to(device)
            embedding = resnet(tensor).squeeze().cpu().numpy()
            embeddings.append(embedding)

    X = np.array(embeddings, dtype=np.float32)
    X = scaler.transform(X)

    probs = clf.predict_proba(X)
    
    # Track votes and the highest confidence seen for that label
    class_votes = {}
    class_max_conf = {}

    for prob_dist in probs:
        top_idx = np.argmax(prob_dist)
        top_conf = prob_dist[top_idx]
        
        # Stricter initial threshold
        if top_conf >= confidence_threshold:
            label = le.inverse_transform([top_idx])[0]
            
            # Tally the vote
            class_votes[label] = class_votes.get(label, 0) + 1
            
            # Keep track of the highest score for printing
            if label not in class_max_conf or top_conf > class_max_conf[label]:
                class_max_conf[label] = top_conf

    # Filter by consensus: Only keep objects that got enough votes
    final_detections = {}
    for label, votes in class_votes.items():
        if votes >= min_votes:
            final_detections[label] = class_max_conf[label]

    final_detections = dict(sorted(final_detections.items(), key=lambda x: -x[1]))

    # Print results
    print('\n' + '=' * 45)
    print(f'   OBJECTS DETECTED (Min {min_votes} votes needed)')
    print('=' * 45)
    if final_detections:
        for obj, conf in final_detections.items():
            votes = class_votes[obj]
            bar = '█' * int(conf * 20)
            print(f'  {obj:<15} {conf:.3f}  ({votes:02d} votes) {bar}')
    else:
        print('  No objects met the consensus threshold.')
    print('=' * 45)

    return list(final_detections.keys())


if __name__ == '__main__':
    img_path = '2008_000006.jpg'
    # Require at least 4 overlapping windows to agree, and an 80% confidence
    objects = detect_objects(img_path, confidence_threshold=0.80, min_votes=4)
