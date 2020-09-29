#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
��MNIST�ж�ȡԭʼͼƬ�����桢��ȡ��ǩ���ݲ����档
MNIST�ļ��ṹ�������Բο���https://blog.csdn.net/justidle/article/details/103149253
"""
"""
ʹ�÷�����
1����MNIST���ļ����ص����ء�
2����py�ļ�����Ŀ¼�£�����mnist_dataĿ¼��Ȼ��MNIST���ĸ��ļ�������mnist_dataĿ¼������ѹ
3����py�ļ�����Ŀ¼�£�����testĿ¼����Ŀ¼���ڴ�Ž�ѹ����ͼƬ�ļ��ͱ�ǩ�ļ�
"""

import struct
import numpy as np
import PIL.Image

def read_image(filename):
    #���ļ�
    f = open(filename, 'rb')
    
    #��ȡ�ļ�����
    index = 0
    buf = f.read()
    
    #�ر��ļ�
    f.close()
    
    #�����ļ�����
    #>IIII ��ʾʹ�ô�˹��򣬶�ȡ�ĸ�����
    magic, numImages, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    
    for i in range(0, numImages):
        # L����Ҷ�ͼƬ
        image = PIL.Image.new('L', (columns, rows))
        
        for x in range(rows):
            for y in range(columns):
                # ��>B' ��ȡһ���ֽ�
                image.putpixel((y,x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')
                
        print('save ' + str(i) + 'image')
        image.save('mnist_data/test/'+str(i)+'.png')
        
def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    
    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    
    labelArr = [0] * labels
    
    for x in range(labels):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
    
    save = open(saveFilename, 'w')
    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')
    save.close()
    print('save labels success')

if __name__ == '__main__':
    #ע��t10k-images-idx3-ubyte����һ����10,000��ͼƬ
    read_image('mnist_data/t10k-images-idx3-ubyte')
    read_label('mnist_data/t10k-labels-idx1-ubyte', 'mnist_data/test/label.txt')
    