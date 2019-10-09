# Import useful libraries
import os
import shutil
import random
import argparse

parser = argparse.ArgumentParser(description='Creates train/val(/test) folders given the percentage of samples for each folder')
parser.add_argument('--percentages', nargs='*', default=[], type=int, 
                   help='Percentage for training/validation/test. If two integers are placed there will not be test folder')

args = parser.parse_args()
percentages = args.percentages
if (len(percentages) not in [2,3]) or sum(percentages) != 100:
    parser.error(message= 'Add two or three percentages values such that the sum is 100')
    exit(1)


#Get current cwd and path for raw dataset and the new dataset
cwd = os.getcwd()
raw_path = cwd + '/raw_dataset'
data_path = cwd + '/dataset'

# Check if raw_dataset data is available
if not os.path.exists(raw_path):
    parser.error(message='file "raw_dataset" not found in current directory')
    exit(1)

#If the folder dataset already exists it deletes it 
if os.path.exists(data_path) and os.path.isdir(data_path):
    shutil.rmtree(data_path)

# fire and nonfire images filenames and shuffled 
fire_img_fn= os.listdir(raw_path +'/fire')
nonfire_img_fn = os.listdir(raw_path + '/nonfire')

shuffled_fire_img_fn = fire_img_fn[:]
shuffled_nonfire_img_fn = nonfire_img_fn[:]
random.shuffle(shuffled_fire_img_fn)
random.shuffle(shuffled_nonfire_img_fn)

n_samples = len(shuffled_fire_img_fn) + len(shuffled_nonfire_img_fn)
n_fire_img = len(shuffled_fire_img_fn)
n_nonfire_img = len(shuffled_nonfire_img_fn)

if len(percentages) == 2:
    data_type = {
        'train': percentages[0],
        'validation': percentages[1]
        }
else:
    data_type = {
        'train': percentages[0],
        'validation': percentages[1],
        'test': percentages[2]
        }

for type in data_type.keys():
    os.makedirs(data_path + '/{}'.format(type), exist_ok=True)
    os.makedirs(data_path + '/{}/fire'.format(type), exist_ok=True)
    os.makedirs(data_path + '/{}/nonfire'.format(type), exist_ok=True)

    # fire/nonfire  upper bounds
    f_upper_bound = data_type[type] * n_fire_img // 100 
    nf_upper_bound = data_type[type] * n_nonfire_img // 100
    for i, filename in enumerate(shuffled_fire_img_fn[: f_upper_bound], start=1):
        ext = filename.split('.')[-1]
  
        src = raw_path + '/fire/{}'.format(filename)
        dst = data_path + '/{}/fire/fire_{}.{}'.format(type, i, ext) 
        shutil.copyfile(src, dst)

    for i, filename in enumerate(shuffled_nonfire_img_fn[: nf_upper_bound], start=1):
        ext = filename.split('.')[-1]
        
        src = raw_path + '/{}/{}'.format('nonfire',filename)
        dst = data_path + '/{}/nonfire/nonfire_{}.{}'.format(type, i, ext)
        shutil.copyfile(src, dst)

    del shuffled_fire_img_fn[:f_upper_bound]
    del shuffled_nonfire_img_fn[:nf_upper_bound]


