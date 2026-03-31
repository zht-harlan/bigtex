import pandas as pd  
import numpy as np  
from sklearn.manifold import TSNE  
import matplotlib.pyplot as plt  
from load_data import *
import torch
import torch.nn.functional as F





# data = Data(
#     n_id=dataset.node_map,
#     x=dataset.x_original,
#     edge_index=dataset.edge_index,
#     y=dataset.label_map,
# )
def load_and_preprocess_embeddings(filename, data):
    data1 = pd.read_csv(filename, encoding='utf-8', engine='python')
    embeddings = np.array([np.fromstring(embedding, sep=',') for embedding in data1['Embedding']])
    ids = data1['ID'].values
    
    # مرتب‌سازی امبدینگ‌ها
    embeddings_df = pd.DataFrame(embeddings, index=ids)
    embeddings_sorted = embeddings_df.loc[data.n_id.numpy()]
    
    # تبدیل به tensor و نرمال‌سازی
    embeddings_tensor = torch.tensor(embeddings_sorted.values, dtype=torch.float)
    normalized_embeddings = F.normalize(embeddings_tensor, p=2, dim=1)
    
    return normalized_embeddings


dataset_name = "arxiv"
if dataset_name == 'arxiv':
        data, _=load_arxiv(dataset_name=dataset_name)
elif dataset_name=='arxiv_2023':
        data, _=load_arxiv_2023()
x='orig'
# x='GT'
# x='tape'
# x='simteg'
# لود و پردازش امبدینگ‌ها
if     x == "GT":
    if dataset_name == "arxiv":
        filename = 'embeddings/embeddings_arxiv_GT_SAge_fine.csv'
    elif dataset_name == "arxiv_2023":
        filename = 'embeddings/embeddings_arxiv_2023_GT_SAGE_fine.csv'
    embeddings = load_and_preprocess_embeddings(filename, data)
   
elif x=='simteg':
    if dataset_name == "arxiv":
        path = 'embeddings/simteg_arxiv.pt'  # یا مسیر کامل فایل
    elif dataset_name == "arxiv_2023":
        path = 'embeddings/sim_arxiv_2023.pt'
    embeddings = torch.load(path)
elif x=='tape':
        if dataset_name == "arxiv":
            embedding_dim = 384
            emb = np.fromfile("embeddings/tape_arxiv.emb", dtype=np.float32)
            emb = emb.reshape(-1, embedding_dim)
            embeddings = torch.tensor(emb)
        elif dataset_name == "arxiv_2023":
            embedding_dim = 384
            emb = np.fromfile("embeddings/tape_arxiv2023.emb", dtype=np.float32)
            emb = emb.reshape(-1, embedding_dim)
            embeddings = torch.tensor(emb)  
else:
    embeddings=data.x
labels = data.y

#----------------انتخاب رندوم 100 نمونه از هر کلاسس

import numpy as np

num_per_class = 100
selected_indices = []

labels = np.array(labels)  # اطمینان از اینکه labels از نوع numpy array هست

unique_labels = np.unique(labels)

for label in unique_labels:
    indices = np.where(labels == label)[0]
    if len(indices) >= num_per_class:
        selected = np.random.choice(indices, num_per_class, replace=False)
    else:
        selected = np.random.choice(indices, len(indices), replace=False)  # اگر کمتر از 100 تا داشت
    selected_indices.extend(selected)

selected_indices = np.array(selected_indices)
#----------------انتخاب رندوم 100 نمونه از هر کلاسس




# بارگذاری داده‌ها از فایل CSV  
# filename = 'GPU@/embeddings_arxiv_GT_SAge_fine.csv'  
# data = pd.read_csv(filename)  

# استخراج امبدینگ‌ها و لیبل‌ها  
# embeddings = np.array([np.fromstring(embedding, sep=',') for embedding in data['Embedding']])  
# labels = data['Label'].values  
embeddings=embeddings[selected_indices]
labels=labels[selected_indices]
# بررسی شکل داده‌ها  
print("شکل امبدینگ‌ها:", embeddings.shape)  # باید (2708, 768) باشد  

# اجرای t-SNE برای کاهش ابعاد  
tsne = TSNE(n_components=2, random_state=42)  
embeddings_2d = tsne.fit_transform(embeddings)  

# ایجاد دیکشنری برای نقشه رنگی  
unique_labels = np.unique(labels)  
colors = plt.cm.get_cmap('hsv', len(unique_labels))  # نقشه رنگی HSV با تعداد کلاس‌ها  

# تصویرسازی  
plt.figure(figsize=(10, 8))  

# رسم نقاط با رنگ‌های مربوط به هر کلاس  
for i, label in enumerate(unique_labels):  
    indices = np.where(labels == label)  
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],   
                color=colors(i), label=label, alpha=0.6)  



# markers = ['o', 's', '^', 'v', '*', 'P', 'X', '+', 'D', 'h']  # هر کلاس یک شکل
# colors = plt.cm.tab10  # یا هر colormap دلخواه

# for i, label in enumerate(unique_labels):  
#     indices = np.where(labels == label)
#     plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],   
#                 color=colors(i), marker=markers[i % len(markers)],
#                 label=label, alpha=0.6)





# نمایش اسطوره  
# plt.legend(title='Classes', loc='best', fontsize='small')  

plt.title('t-SNE Visualization of Embeddings')  
plt.xlabel('t-SNE Component 1')  
plt.ylabel('t-SNE Component 2')  
# plt.grid()  
plt.show()
f_name="tsne_"+dataset_name+"_"+x+".png"
plt.savefig(f_name, dpi=300)