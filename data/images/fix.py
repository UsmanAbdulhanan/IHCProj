import csv
import os
import re

patch = 'NKX3'
# with open(f'data/data_splits/i2i/{patch}_splits.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     elems = []
#     for itm in spamreader:
#         if itm :
#             cleaned_items = [i for i in itm if i.strip() != '']
#             elems.extend(cleaned_items)



pics = os.listdir(f'data/bbox_info/HE_NKX3/{patch}')
files = sorted(pics)
files_no_ext = [f.removesuffix(".npz") for f in files]



with open(f'data/images/{patch}_splitsn.csv', 'w') as out:
    writer = csv.writer(out)
    for elem in files_no_ext:
        writer.writerow([elem])
    
# with open(f'data/data_splits/i2i/{patch}_splits.csv', newline='') as inp, open(f'data/images/{patch}_splitsn.csv', 'w') as out:
#     writer = csv.writer(out)
#     for row in csv.reader(inp, delimiter=',', quotechar='|'):
#         skip = False
#         for itm in row:
#             if itm and itm not in good_names:         
#                 skip = True
#                 break
#         if not skip:        
#             writer.writerow(row)