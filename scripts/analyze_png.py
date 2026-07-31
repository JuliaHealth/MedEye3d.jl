from PIL import Image
import numpy as np

img = Image.open('example_quad_decathlon.png').convert('RGB')
arr = np.array(img)
h, w, c = arr.shape
print(f"Image shape: {w}x{h}")

q1 = arr[:h//2, :w//2] # Top-left (Panel 1)
q2 = arr[:h//2, w//2:] # Top-right (Panel 2)
q3 = arr[h//2:, :w//2] # Bottom-left (Panel 3)
q4 = arr[h//2:, w//2:] # Bottom-right (Panel 4)

def analyze(name, q):
    total = q.shape[0] * q.shape[1]
    black = np.sum(np.all(q == [0, 0, 0], axis=2))
    red = np.sum((q[:,:,0] > 100) & (q[:,:,1] < 50) & (q[:,:,2] < 50))
    grey = np.sum((q[:,:,0] > 50) & (q[:,:,0] < 200) & (np.abs(q[:,:,0]-q[:,:,1]) < 10) & (np.abs(q[:,:,1]-q[:,:,2]) < 10))
    print(f"{name}: Black: {black/total*100:.1f}%, Red: {red/total*100:.1f}%, Grey: {grey/total*100:.1f}%")

analyze("Panel 1 (TL)", q1)
analyze("Panel 2 (TR)", q2)
analyze("Panel 3 (BL)", q3)
analyze("Panel 4 (BR)", q4)
