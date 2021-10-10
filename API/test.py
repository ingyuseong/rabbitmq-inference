import os

male_dir = "/Users/macbook/Google Drive/프로젝트/bidi_ai/API/male"

for i, filename in enumerate(os.listdir(male_dir)):
  changed = 'male_style'
  if len(str(i)) < 2:
    changed += '0'
  changed += str(i)
  print(changed)
  os.rename(os.path.join(male_dir, filename), male_dir + "/" + changed + ".jpg")