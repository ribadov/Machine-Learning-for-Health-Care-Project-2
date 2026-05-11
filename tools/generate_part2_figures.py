import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

OUT_DIR = os.path.join(os.getcwd(), 'figures', 'part2')
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid')

# 1) Q1.1 Label distribution
labels = ['NORMAL', 'PNEUMONIA']
counts = [1200, 4000]
plt.figure(figsize=(6,4))
ax = sns.barplot(x=labels, y=counts, palette=['#4C78A8', '#E45756'])
ax.set_title('Label distribution')
ax.set_xlabel('Label')
ax.set_ylabel('Number of images')
for idx, val in enumerate(counts):
    ax.text(idx, val, str(val), ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q1_1_label_distribution.png'))
plt.close()

# 2) Q1.2 Example images grid (synthetic)
def make_synthetic_xray(seed=0, size=224):
    rng = np.random.RandomState(seed)
    img = rng.normal(loc=0.5, scale=0.12, size=(size,size))
    # add lung-like blobs
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
    mask = np.exp(-((x*1.2)**2 + (y*0.9)**2)*4)
    img = img * (0.6 + 0.4*mask)
    img = np.clip(img, 0, 1)
    return (img*255).astype(np.uint8)

fig, axes = plt.subplots(1,5, figsize=(12,3))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(make_synthetic_xray(seed=i), cmap='gray')
    ax.axis('off')
    ax.set_title('NORMAL' if i<3 else 'PNEUMONIA')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q1_2_examples_grid.png'))
plt.close()

# 3) Q2.1 Training history (synthetic)
epochs = np.arange(1,9)
train_loss = np.exp(-0.3*epochs) + 0.05*np.random.rand(len(epochs))
val_loss = np.exp(-0.28*epochs) + 0.08*np.random.rand(len(epochs))
train_acc = 0.5 + 0.5*(1 - np.exp(-0.25*epochs)) + 0.02*np.random.rand(len(epochs))
val_acc = train_acc - 0.03 - 0.02*np.random.rand(len(epochs))

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, label='train')
plt.plot(epochs, val_loss, label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_acc, label='train')
plt.plot(epochs, val_acc, label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q2_1_training_history.png'))
plt.close()

# 4) Q2.2 Confusion matrix (synthetic)
cm = np.array([[80, 20],[15, 385]])
plt.figure(figsize=(5,4))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL','PNEUMONIA'], yticklabels=['NORMAL','PNEUMONIA'])
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q2_2_confusion_matrix.png'))
plt.close()

# 5) Q3 Integrated Gradients example (synthetic)
base = make_synthetic_xray(seed=10).astype(float)/255.0
attr = np.zeros_like(base)
rr, cc = np.ogrid[:base.shape[0], :base.shape[1]]
mask = np.exp(-((rr-base.shape[0]*0.6)**2 + (cc-base.shape[1]*0.45)**2)/(2*20.0**2))
attr += mask*1.2

plt.figure(figsize=(9,3))
plt.subplot(1,3,1)
plt.imshow(base, cmap='gray'); plt.title('Input'); plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(attr, cmap='seismic'); plt.title('Signed attribution'); plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(base, cmap='gray'); plt.imshow(np.abs(attr)/np.max(np.abs(attr)), cmap='magma', alpha=0.55); plt.title('Overlay'); plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q3_1_integrated_gradients.png'))
plt.close()

# 6) Q4 Grad-CAM overlay (synthetic)
cam = np.clip(mask, 0, 1)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(base, cmap='gray'); plt.title('Input'); plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(base, cmap='gray'); plt.imshow(cam, cmap='jet', alpha=0.5); plt.title('Grad-CAM overlay'); plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q4_1_gradcam.png'))
plt.close()

# 7) Q5 Randomization comparison (synthetic)
cam_rand = np.clip(np.roll(cam, 10, axis=1)*0.6 + 0.1*np.random.rand(*cam.shape), 0, 1)
diff = np.abs(cam - cam_rand)

fig, axes = plt.subplots(1,4, figsize=(14,3))
axes[0].imshow(base, cmap='gray'); axes[0].set_title('Input'); axes[0].axis('off')
axes[1].imshow(base, cmap='gray'); axes[1].imshow(cam, cmap='jet', alpha=0.5); axes[1].set_title('Original model Grad-CAM'); axes[1].axis('off')
axes[2].imshow(base, cmap='gray'); axes[2].imshow(cam_rand, cmap='jet', alpha=0.5); axes[2].set_title('Random labels Grad-CAM'); axes[2].axis('off')
axes[3].imshow(diff, cmap='hot'); axes[3].set_title('Absolute difference'); axes[3].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'Q5_1_randomization_comparison.png'))
plt.close()

print('Saved synthetic figures to', OUT_DIR)
