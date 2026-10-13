import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from sklearn.cluster import KMeans

print("Worksheet 10.6 - Exercise 2: Image compression using K-Means")
image = load_sample_image("china.jpg")
img = image[10:406, 120:516]
height, width, _ = img.shape
pixels = img.reshape(-1,3).astype(float)/255
results, compressed = [], {}
for k in [2,4,8,16,32]:
    model = KMeans(n_clusters=k,n_init=1,random_state=42).fit(pixels)
    reconstructed = model.cluster_centers_[model.labels_].reshape(height,width,3)
    compressed[k] = reconstructed
    mse = np.mean((img.astype(float)/255-reconstructed)**2)
    original_bits = height*width*24
    compressed_bits = height*width*math.ceil(math.log2(k))+k*24
    results.append({"K colours":k,"bits/pixel":math.ceil(math.log2(k)),
                    "PSNR (dB)":10*np.log10(1/max(mse,1e-12)),
                    "Compression ratio":original_bits/compressed_bits})
print(pd.DataFrame(results).round(2).to_string(index=False))
fig, axes = plt.subplots(2,3,figsize=(12,8))
axes[0,0].imshow(img); axes[0,0].set_title("Original")
for ax,(k,im) in zip(axes.ravel()[1:],compressed.items()):
    ax.imshow(np.clip(im,0,1)); ax.set_title(f"K = {k} colours")
for ax in axes.ravel(): ax.axis("off")
plt.tight_layout(); plt.show()
