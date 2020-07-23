import pandas as pd
import cv2
import os


from sklearn.utils import shuffle

# temp_data/mask
# 231.04761904761904 231.04761904761904
# temp_data/no_mask
# 228.82738095238096 228.82738095238096

path = "temp_data/"
folder_list = os.listdir(path)
df = pd.DataFrame(columns = ['filename', 'label'])


for i in range(len(folder_list)):

  if folder_list[i] != '.DS_Store':
    sub_fol_list = os.listdir(path+folder_list[i])
    print(path + folder_list[i])

    for images in sub_fol_list:
      img_path = path + folder_list[i] + "/" + images
      img = cv2.imread(img_path,0)
      resize_img = cv2.resize(img, (222,222))

      if folder_list[i] == 'mask':
        img_name = "m" + images
        df = df.append({'filename' : img_name, 'label' : 1}, ignore_index = True)
        cv2.imwrite("dataset/" + img_name,resize_img)
      if folder_list[i] == 'no_mask':
        img_name = "n" + images
        df = df.append({'filename' : img_name, 'label' : 0}, ignore_index = True)
        cv2.imwrite("dataset/" + img_name,resize_img)


df = shuffle(df)
df.to_csv("mnm_dataset.csv")
print(df.head())

#222.86319218241042