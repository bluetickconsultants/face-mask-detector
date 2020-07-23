import os
import cv2

path = 'temp_data'
avg_fol1_col = 0
folders = os.listdir(path)
for folder in folders:
  row = 0
  col = 0
  files = os.listdir(os.path.join(path, folder))
  print(folder, len(files))
  for file in files:
    filepath = os.path.join(path, folder, file)
    image = cv2.imread(filepath, 0)
    row += image.shape[0]
    col += image.shape[1]
  avg_fol1_row = row / len(files)
  avg_fol1_col += col / len(files)
  print(folder, avg_fol1_row)
print(avg_fol1_col / 2)