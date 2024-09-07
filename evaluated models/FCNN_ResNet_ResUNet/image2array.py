import SimpleITK as sitk
import json
import h5py
import numpy as np
import os
import pickle
import random

#Base = '/nfs/project/zhanj7/brats'
#Base = 'data/test_data/'
Base = '../../../testing_dataset/testing_data/'


def write2file(data, labels, save_path):

    writeFile = open(save_path, 'wb')
    d = {}
    d['X'] = data
    d['y'] = labels
    pickle.dump(d, writeFile)
    writeFile.close()
    print ('save success')


def write2h5(data, labels, save_path):

    writeFile = h5py.File(save_path, 'w')
    writeFile.create_dataset('X', data=data)
    writeFile.create_dataset('y', data=labels)
    writeFile.close()
    print ('save h5 success')

def get_data(filelists, label):

    train_np = np.empty([0, 155, 240, 240, 4], dtype = 'int16')
    temp_np = np.empty([0, 155, 240, 240], dtype = 'int16')
    train_label_np = np.empty([0, 155, 240, 240], dtype = 'int16')

    for filelist in filelists:
        print ('processing data tanjida', filelist[0].split('/')[-3])
        for filename in filelist:
            print(filename)
            img = sitk.ReadImage(filename)
            imgdata = sitk.GetArrayFromImage(img)
            temp_np = np.append(temp_np, imgdata.reshape(1, 155, 240, 240), axis = 0)
            print("Debug 1",temp_np.shape)
        temp_np = temp_np.transpose((1, 2, 3, 0))
        train_np = np.append(train_np, temp_np.reshape(1, 155, 240, 240, 4), axis = 0)
        print(train_np.shape)
        temp_np = np.empty([0, 155, 240, 240], dtype = 'int16')

    for filename in label:
        print ('processing label tanjida', filename.split('/')[-3])
        img = sitk.ReadImage(filename)
        imgdata = sitk.GetArrayFromImage(img)
        train_label_np = np.append(train_label_np, imgdata.reshape(1, 155, 240, 240), axis = 0)
        print(train_label_np.shape)

    return train_np, train_label_np


def get_file_lists(topdir):

    '''
    return the file format we want
    '''

    file_lists = []
    temp_list = []
    label_list = []

    if not os.path.exists(topdir):
        print('file path does not exist')

    i = 0
    for dirpath, dirname, filenames in os.walk(topdir):
        print('I am here',filenames)
        if filenames:
            print("WHY????//")
            for filename in filenames:
                #print(filename)
                #if filename.endswith('.mha') and filename.startswith('VSD.Brain_'):
                #if (filename.endswith('_seg.nii.gz') and filename.startswith('BraTS19_')) :
                #if filename.endswith('_Seg.nii.gz') :
                if filename.endswith('_seg.nii.gz') :
                    print("I am here 2222222")
                    label_list.append(dirpath + '/' +  filename)
                    i = i + 1
                #elif filename.endswith('.mha'):
                #elif (filename.endswith('_flair.nii.gz') and filename.startswith('BraTS19_')) or (filename.endswith('_t1.nii.gz') and filename.startswith('BraTS19_')) or (filename.endswith('_t1ce.nii.gz') and filename.startswith('BraTS19_')) or (filename.endswith('_t2.nii.gz') and filename.startswith('BraTS19_')):
                elif (filename.endswith('_flair.nii.gz')) or (filename.endswith('_t1.nii.gz') ) or (filename.endswith('_t1ce.nii.gz') ) or (filename.endswith('_t2.nii.gz') ):
                #elif (filename.endswith('_flair.nii')) or (filename.endswith('_t1.nii') ) or (filename.endswith('_t1ce.nii') ) or (filename.endswith('_t2.nii') ):
                    temp_list.append(dirpath + '/' + filename)
                    i = i + 1
            if i == 5:
                temp_list.sort()
                file_lists.append(temp_list)
                temp_list = []
                i = 0

    return file_lists, label_list


def Array2Image(from_path, to_path, array):
    lists, _ = get_file_lists(from_path)
    for i in xrange(len(lists)):
        num = lists[i][0].split('.')[-2]
        img = sitk.GetImageFromArray(array[i])
        sitk.WriteImage(img, to_path + '/VSD.Seg_zjc.' + num + '.mha')

def single2Image(save_path, name, array):
    print(array.shape)
    print(array.dtype)
    img = sitk.GetImageFromArray(array.astype(np.int64))
    print(save_path+'/'+name+'.mha')
    # help(img)
    sitk.WriteImage(img, save_path+'/'+name+'.mha')
    print('saving '+name+'finished')

def save_pred(array, save_path, name_file):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    N, D, W, H = array.shape
    # d = json.loads(open(name_file, 'r').readline())
    l = json.loads(open(name_file, 'r').readline())
    print("LLLLLL printing",array.shape)
    #for i in range(220):
    for i in range(1):
        name = save_path+'/'+np.str(i)+'.mha'
        img = sitk.GetImageFromArray(array[i])
        sitk.WriteImage(img, name)
        print(name+' has been saved')

    
    for i, name in enumerate(l):
        file_name = 'VSD.Seg_HG_zjc.'+name[0].split('/')[-2].split('.')[-1]
        os.rename(save_path+'/'+np.str(i)+'.mha', save_path+'/'+file_name+'.mha')
        print('rename '+file_name+' succeed')

    


def five2four(array):
    N, D, W, H = array.shape
    result = np.empty(array.shape, array.dtype)
    result[np.where(array==4)] = 3
    result[np.where(np.logical_or(array==1, array==3))] = 2
    result[np.where(array == 2)] = 1
    return result




if __name__ == '__main__':

    f_list, l_list = get_file_lists(Base)

    print("DEBUG 3",f_list)
    print("DEBUG 4",l_list)

    HGG_train_np, HGG_train_label_np = get_data(f_list, l_list)

    print('>>>>>>>>>>>>>>',l_list)

    print("DEBUG 3",HGG_train_np.shape)
    print("DEBUG 4",HGG_train_label_np.shape)
    print('Start Writing HGG_train.npz...')
    np.savez('HGG_train_only_45_patients.npz', HGG_train_np, HGG_train_label_np)





    #a = np.ones((220, 4, 4, 4), dtype = np.uint16)
    #save_pred(a, Base+'/prediction/tmp', './HGG_train.json')

    




    '''
    f_list, l_list = get_file_lists(Base+'/BRATS2015_Training/HGG')
    json.dump(f_list, open('./HGG_train.json', 'w'))
    f_np, l_np = get_data(f_list, l_list)
    exit(0)

    write2h5(f_np, l_np, Base+'/HGG_train.h5')
    '''

    '''
    file_list, label_list = get_file_lists('/home/ff/data/Brain_Tumor/BRATS2015_Training/LGG')
    name_list =  [ele.split('/')[-3] for ele in label_list]
    d = {}
    d['file_list'] = file_list
    d['label_list'] = label_list
    d['name_list'] = name_list
    with open('LGG_train.json', 'w') as f:
        json.dump(d, f)
    exit(0)

    # Get HGG and LGG traing list with shape (n, 4) and label list with shape (n, )
    HGG = []
    LGG = []
    HGG_label = []
    LGG_label = []

    HGG_temp = []
    LGG_temp = []

    i = 0
    for dirpath, dirname, filenames in os.walk(HGG_base):
        if filenames:
            for filename in filenames:
                if filename.endswith('.mha') and filename.startswith('VSD.Brain_'):
                    HGG_label.append(dirpath + '/' +  filename)
                    i = i + 1
                elif filename.endswith('.mha'):
                    HGG_temp.append(dirpath + '/' + filename)
                    i = i + 1
            if i == 5:
                HGG_temp.sort()
                HGG.append(HGG_temp)
                HGG_temp = []
                i = 0

    i = 0
    for dirpath, dirname, filenames in os.walk(LGG_base):
        if filenames:
            for filename in filenames:
                if filename.endswith('.mha') and filename.startswith('VSD.Brain_'):
                    LGG_label.append(dirpath + '/' +  filename)
                    i = i + 1
                elif filename.endswith('.mha'):
                    LGG_temp.append(dirpath + '/' + filename)
                    i = i + 1
            if i == 5:
                LGG_temp.sort()
                LGG.append(LGG_temp)
                LGG_temp = []
                i = 0

    print(len(HGG_label), len(HGG), len(LGG_label), len(LGG))

    # print 'HGG_label: ', HGG_label
    # print ''
    # print 'HGG: ', HGG
    # print ''
    # print 'LGG_label: ', LGG_label
    # print ''
    # print 'LGG: ', LGG


    # Shuffle to get the training data and validation data
    HGG_val = []
    HGG_val_label = []
    HGG_train = []
    HGG_train_label = []

    index = np.arange(len(HGG))
    np.random.shuffle(index)
    HGG_val_index = index[:len(HGG) / 10 * 3]
    HGG_val_index.sort()
    HGG_train_index = np.delete(np.arange(len(HGG)), HGG_val_index)

    for i in HGG_val_index:
        HGG_val.append(HGG[i])
        HGG_val_label.append(HGG_label[i])
    for i in HGG_train_index:
        HGG_train.append(HGG[i])
        HGG_train_label.append(HGG_label[i])

    # HGG_val = HGG_train[HGG_val_index]
    # HGG_val_label = HGG_label[HGG_val_index]
    # HGG_train = HGG_train[HGG_train_index]
    # HGG_train_label = HGG_label[HGG_train_index]

    LGG_val = []
    LGG_val_label = []
    LGG_train = []
    LGG_train_label = []

    index = np.arange(len(LGG))
    np.random.shuffle(index)
    LGG_val_index = index[:len(LGG) / 10 * 3]
    LGG_val_index.sort()
    LGG_train_index = np.delete(np.arange(len(LGG)), LGG_val_index)

    for i in LGG_val_index:
        LGG_val.append(LGG[i])
        LGG_val_label.append(LGG_label[i])
    for i in LGG_train_index:
        LGG_train.append(LGG[i])
        LGG_train_label.append(LGG_label[i])


    print(len(HGG_val), len(HGG_val_label), len(HGG_train), len(HGG_train_label))
    print(len(LGG_val), len(LGG_val_label), len(LGG_train), len(LGG_train_label))




    # Read data from file lists and labels
    # try:
    #     HGG_train_np, HGG_train_label_np = get_data(HGG_train, HGG_train_label)
    #     print 'Start Writing HGG_train.pkl...'
    #     write2File(HGG_train_np, HGG_train_label_np, 'HGG_train.pkl')

    #     HGG_val_np, HGG_val_label_np = get_data(HGG_val, HGG_val_label)
    #     print 'Start Writing HGG_val.pkl...'
    #     write2File(HGG_val_np, HGG_val_label_np, 'HGG_val.pkl')

    #     LGG_train_np, LGG_train_label_np = get_data(LGG_train, LGG_train_label)
    #     print 'Start Writing LGG_train.pkl...'
    #     write2File(LGG_train_np, LGG_train_label_np, 'LGG_train.pkl')

    #     LGG_val_np, LGG_val_label_np = get_data(LGG_val, LGG_val_label)
    #     print 'Start Writing LGG_val.pkl...'
    #     write2File(LGG_val_np, LGG_val_label_np, 'LGG_val.pkl')

    #     print 'Success!'
    # except:
    #     print 'Failed, ZannNenn'

    HGG_val_np, HGG_val_label_np = get_data(HGG_val, HGG_val_label)
    print('Start Writing HGG_val.npz...')
    np.savez(data_base + 'HGG_val.npz', HGG_val_np, HGG_val_label_np)
    # write2File(HGG_val_np, HGG_val_label_np, 'HGG_val.pkl')

    HGG_train_np, HGG_train_label_np = get_data(HGG_train, HGG_train_label)
    print('Start Writing HGG_train.npz...')
    np.savez(data_base + 'HGG_train.npz', HGG_train_np, HGG_train_label_np)
    # write2File(HGG_train_np, HGG_train_label_np, 'HGG_train.pkl')

    LGG_val_np, LGG_val_label_np = get_data(LGG_val, LGG_val_label)
    print('Start Writing LGG_val.npz...')
    np.savez(data_base + 'LGG_val.npz', LGG_val_np, LGG_val_label_np)
    # write2File(LGG_val_np, LGG_val_label_np, 'LGG_val.pkl')

    LGG_train_np, LGG_train_label_np = get_data(LGG_train, LGG_train_label)
    print('Start Writing LGG_train.npz...')
    np.savez(data_base + 'LGG_train.npz', LGG_train_np, LGG_train_label_np)
    # write2File(LGG_train_np, LGG_train_label_np, 'LGG_train.pkl')



    





    # print len(HGG_label), len(HGG_train), len(LGG_label), len(LGG_train)

    # print 'HGG_label: ', HGG_label
    # print ''
    # print 'HGG_train: ', HGG_train
    # print ''
    # print 'LGG_label: ', LGG_label
    # print ''
    # print 'LGG_train: ', LGG_train

    # HGG_train_np = np.empty([0, 4, 240, 240, 155], dtype = 'int16')
    # HGG_temp_np = np.empty([0, 240, 240, 155], dtype = 'int16')
    # HGG_train_label_np = np.empty([0, 240, 240, 155], dtype = 'int16')

    # for filelist in HGG_train:
    #     for filename in filelist:
    #         img = sitk.ReadImage(filename)
    #         imgdata = sitk.GetArrayFromImage(img) 
    #         # print imgdata.shape
    #         imgdata = imgdata.transpose((2, 1, 0))
    #         HGG_temp_np = np.append(HGG_temp_np, imgdata.reshape(1, 240, 240, 155), axis = 0)
    #         # print HGG_temp_np.shape
    #     HGG_train_np = np.append(HGG_train_np, HGG_temp_np.reshape(1, 4, 240, 240, 155), axis = 0)
    #     print HGG_train_np.shape
    #     HGG_temp_np = np.empty([0, 240, 240, 155], dtype = 'int16')

    # for filename in HGG_train_label:
    #     img = sitk.ReadImage(filename)
    #     imgdata = sitk.GetArrayFromImage(img)
    #     imgdata = imgdata.transpose((2, 1, 0))
    #     HGG_label_np = np.append(HGG_train_label_np, imgdata.reshape(1, 240, 240, 155), axis = 0)
    #     print HGG_label.shape


    # print 'Start Pickling Training Data...'
    # write2File(HGG_train_np, HGG_train_label_np, 'HGG_train.pkl')
    # print 'Training Data Saved!'


    # LGG_train_np = np.empty([0, 4, 240, 240, 155], dtype = 'int16')
    # LGG_temp_np = np.empty([0, 240, 240, 155], dtype = 'int16')
    # LGG_train_label_np = np.empty([0, 240, 240, 155], dtype = 'int16')

    # for filelist in LGG_train:
    #     for filename in filelist:
    #         img = sitk.ReadImage(filename)
    #         imgdata = sitk.GetArrayFromImage(img) 
    #         # print imgdata.shape
    #         imgdata = imgdata.transpose((2, 1, 0))
    #         LGG_temp_np = np.append(LGG_temp_np, imgdata.reshape(1, 240, 240, 155), axis = 0)
    #         # print HGG_temp_np.shape
    #     LGG_train_np = np.append(LGG_train_np, LGG_temp_np.reshape(1, 4, 240, 240, 155), axis = 0)
    #     print LGG_train_np.shape
    #     LGG_temp_np = np.empty([0, 240, 240, 155], dtype = 'int16')

    # for filename in LGG_train_label:
    #     img = sitk.ReadImage(filename)
    #     imgdata = sitk.GetArrayFromImage(img)
    #     imgdata = imgdata.transpose((2, 1, 0))
    #     LGG_label_np = np.append(LGG_train_label_np, imgdata.reshape(1, 240, 240, 155), axis = 0)
    #     print LGG_train_label_np.shape

    # print 'Start Pickling Training Data...'
    # write2File(HGG_train_np, HGG_train_label_np, 'HGG_train.pkl')
    # print 'Training Data Saved!'



    # np.save('/home/ff/data/Brain_Tumor/npy/HGG_train', HGG_train_np)    
    # print HGG_train_np.shape
    '''
