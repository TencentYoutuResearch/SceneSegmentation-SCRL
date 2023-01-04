import os
import cv2
import numpy as np
import json

# Concate 16 shot images into a single image, 
# the concated images are used for speeding up pre-training. 
# Matrix size of the concated image: [16x3]
def concate_pic(shot_info, img_path, save_path, row=16):
    for imdb, shot_num in shot_info.items():
        pic_num = shot_num // row
        for item in range(pic_num):
            img_list = []
            for idx in range(row):
                shot_id = item * row + idx
                img_name_0 = f"{img_path}/{imdb}/shot_{str(shot_id).zfill(4)}_img_0.jpg"
                img_name_1 = f"{img_path}/{imdb}/shot_{str(shot_id).zfill(4)}_img_1.jpg"
                img_name_2 = f"{img_path}/{imdb}/shot_{str(shot_id).zfill(4)}_img_2.jpg"
                img_0 = cv2.imread(img_name_0)
                img_1 = cv2.imread(img_name_1)
                img_2 = cv2.imread(img_name_2)
                img = np.concatenate([img_0,img_1,img_2],axis=1)
                img_list.append(img)
            full_img = np.concatenate(img_list,axis=0)
            # print(img.shape)
            # print(full_img.shape)
            new_pic_dir = f"{save_path}/{imdb}/"
            if not os.path.isdir(new_pic_dir):
                os.makedirs(new_pic_dir)
            filename = new_pic_dir + str(item).zfill(4) + '.jpg'
            cv2.imwrite(filename, full_img)

# Number of shot in each movie
def _generate_shot_num(new_shot_info='./MovieNet_shot_num.json'):
    shot_info = './MovieNet_1.0_shotinfo.json'
    shot_split = './movie1K.split.v1.json'
    with open(shot_info, 'rb') as f:
        shot_info_data = json.load(f)
    with open(shot_split, 'rb') as f:
        shot_split_data = json.load(f)
    new_shot_info_data = {}
    _type = ['train','val','test']
    for _t in _type:
        new_shot_info_data[_t] = {}
        _movie_list = shot_split_data[_t]
        for idx, imdb_id in enumerate(_movie_list):
            shot_num = shot_info_data[_t][str(idx)]
            new_shot_info_data[_t][imdb_id] = shot_num
    with open(new_shot_info, 'w') as f:
        json.dump(new_shot_info_data, f, indent=4)

    
def process_raw_label(_T = 'train', raw_root_dir = './'):
    split = 'movie1K.split.v1.json'
    data_dict = json.load(open(os.path.join(raw_root_dir,split)))

    # print(data_dict.keys())
    # dict_keys(['train', 'val', 'test', 'full'])
    # print(len(data_dict['train'])) # 660
    # print(len(data_dict['val']))  # 220
    # print(len(data_dict['test'])) # 220
    # print(len(data_dict['full'])) # 1100

    data_list = data_dict[_T]

    # annotation
    annotation_path = 'annotation'
    count = 0
    video_list = []
    # all annotations
    for index,name in enumerate(data_list):
        # print(name)
        annotation_file = os.path.join(raw_root_dir, annotation_path, name+'.json')
        data = json.load(open(annotation_file))
        # only need sence seg labels
        if data['scene'] is not None:
            video_list.append({'name':name,'index':index})
            count += 1
    print(f'scene annotations num: {count}')
    return video_list



# GT generation
def process_scene_seg_lable(scene_seg_path = './CVPR20SceneSeg/data/scene318/label318',
    scene_seg_label_json_name = './movie1K.scene_seg_318_name_index_shotnum_label.v1.json',
    raw_root_dir = './MovieNet'):
    def _process(data):
        seg_label = []
        for i in data:
            name = i['name']
            index = i['index']
            label = []
            with open (os.path.join(scene_seg_path,name+'.txt'), 'r') as f:
                shotnum_label = f.readlines()
            for i in shotnum_label:
                if ' ' in i:
                    shot_id = i.split(' ')[0].strip()
                    l = i.split(' ')[1].strip()
                    label.append((shot_id,l))
            shot_count = len(label) + 1
            seg_label.append({"name":name, "index":index, "shot_count":shot_count, "label":label })
        return seg_label

    train_list = process_raw_label('train',raw_root_dir)
    val_list = process_raw_label('val',raw_root_dir)
    test_list = process_raw_label('test',raw_root_dir)
    data = {'train':train_list, 'val':val_list, 'test':test_list}

    # CVPR20SceneSeg GT
    train = _process(data['train'])
    test = _process(data['test'])
    val = _process(data['val'])
    d_all = {'train':train, 'val':val, 'test':test}
    
    with open(scene_seg_label_json_name,'w') as f:
        f.write(json.dumps(d_all))
   


if __name__ == '__main__':
    # Path of movienet images
    img_path = '/MovieNet_unzip/240P'

    # Shot number
    shot_info = './MovieNet_shot_num.json'
    _generate_shot_num(shot_info)

    # GT label
    scene_seg_label_json_name = './movie1K.scene_seg_318_name_index_shotnum_label.v1.json'
    ## Download LGSS Annotation from: https://github.com/AnyiRao/SceneSeg/blob/master/docs/INSTALL.md
    ## 'scene_seg_path' is the path of the downloaded annotations
    scene_seg_path = './CVPR20SceneSeg/data/scene318/label318'
    ## Path of raw MovieNet 
    raw_root_dir = './MovieNet/MovieNet_Ori'
    process_scene_seg_lable(scene_seg_path ,scene_seg_label_json_name, raw_root_dir)

    # Concate images
    save_path = './compressed_shot_images'
    with open(shot_info, 'rb') as f:
        shot_info_data = json.load(f)
    concate_pic(shot_info_data['train'], img_path, save_path)
            
