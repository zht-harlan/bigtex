import csv
import torch
import torch.nn as nn 

# class DimensionalityReductionModel(nn.Module):  
#     def __init__(self, input_dim, output_dim):  
#         super(DimensionalityReductionModel, self).__init__()  
#         self.linear = nn.Linear(input_dim, output_dim)  
    
#     def forward(self, x):  
#         return self.linear(x)

def SaveEmbeddings(model,train_loader,valid_loader,test_loader, dataset_name, model_name):
    #get embbeding of all nodes
    # model.load_state_dict(torch.load('model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # مدل را در حالت ارزیابی قرار می‌دهیم
    all_embeddings = {}
    
    # reduction_model = DimensionalityReductionModel(input_dim=embedding_dim, output_dim=128)
    # reduction_model = reduction_model.to(device)
    # reduction_model.eval()
    # reduction_model.to(device)
    # پردازش داده‌ها به صورت mini-batches
    with torch.no_grad():  # در حالت پیش‌بینی نیازی به محاسبه گرادیان نیست
        for batch in train_loader:
            batch = batch.to(device)
            # فرض می‌کنیم که `batch` شامل ویژگی‌ها و ایندکس‌های گراف برای یک mini-batch است
            _,_, embeddings,_,_ = model(batch.x, batch.edge_index, batch.n_id, batch.x, batch.batch_size)
            # reduced_embeddings = reduction_model(embeddings)
            
            # انتقال به CPU و تبدیل به NumPy
            # reduced_embeddings = reduced_embeddings.cpu().numpy()
            # print("embeddings",embeddings.shape)
            for i in range(batch.batch_size):
                all_embeddings[batch.n_id[i].item()] = {  
                                                            'embedding': [round(value, 4) for value in embeddings[i].tolist()],  
                                                            'label': batch.y[i].item()  
                                                        }   
        for batch in valid_loader:
            batch = batch.to(device)
            # فرض می‌کنیم که `batch` شامل ویژگی‌ها و ایندکس‌های گراف برای یک mini-batch است
            _,_, embeddings,_,_ = model(batch.x, batch.edge_index, batch.n_id, batch.x, batch.batch_size)
            # reduced_embeddings = reduction_model(embeddings)
            # reduced_embeddings = reduced_embeddings.cpu().numpy()
            # print("embeddings",embeddings.shape)
            for i in range(batch.batch_size):
                all_embeddings[batch.n_id[i].item()] = {  
                                                            'embedding': [round(value, 4) for value in embeddings[i].tolist()],  
                                                            'label': batch.y[i].item()  
                                                        }
        for batch in test_loader:
            batch = batch.to(device)
            # فرض می‌کنیم که `batch` شامل ویژگی‌ها و ایندکس‌های گراف برای یک mini-batch است
            _,_,embeddings,_,_ = model(batch.x, batch.edge_index, batch.n_id, batch.x, batch.batch_size)
            # reduced_embeddings = reduction_model(embeddings)
            # reduced_embeddings = reduced_embeddings.cpu().numpy()
            # print("embeddings",embeddings.shape)
            for i in range(batch.batch_size):
                all_embeddings[batch.n_id[i].item()] = {  
                                                            'embedding': [round(value, 4) for value in embeddings[i].tolist()],  
                                                            'label': batch.y[i].item()  
                                                        }

                

    # تبدیل به numpy و ذخیره امبدینگ‌ها
    # all_embeddings = np.concatenate(all_embeddings, axis=0)
    # np.save('embeddings.npy', all_embeddings)
    print("len(all_embeddings)",len(all_embeddings))
    # print(all_embeddings)
    # first_item = next(iter(all_embeddings.items()))  

    # print("اولین عنصر دیکشنری:")  
    # print(first_item)  

    # save embeddings in CSV
      

    # نام فایل برای ذخیره داده‌ها  
    filename = 'embeddings_'+ dataset_name+"_"+ model_name+ '.csv'  
    # ذخیره در CSV  
    # filename = 'embeddings_with_labels.csv'  
    with open(filename, mode='w', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow(['ID', 'Label', 'Embedding'])  # سرستون‌ها  
        for key, value in all_embeddings.items():  
            # تبدیل امبدینگ به یک رشته  
            embedding_str = ",".join(map(str, value['embedding']))  
            writer.writerow([key, value['label'], embedding_str])  
    

    print(f"succsesfully Saved embeddings as {filename}")
