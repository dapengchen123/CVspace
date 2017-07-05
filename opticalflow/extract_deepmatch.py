import argparse
import cv2
import os
import os.path as osp
import scipy.io
import errno
import subprocess
import time


info_dir = '../info'

def main(args):
    data_flag = args.mode
    margin = args.margin
    imgname_file = osp.join(info_dir, data_flag + '_name.txt')
    # get the image files
    imgfo = open(imgname_file, "r")
    namelist = imgfo.read().splitlines()

    # get the matfile about tracklet
    trackletfo = osp.join(info_dir, 'tracks_' + data_flag + '_info.mat')
    trackletmat = scipy.io.loadmat(trackletfo)
    tracklet_lists = trackletmat['track_' + data_flag + '_info']
    trackletdir = osp.join('../../Results/deepmatches', data_flag)
    if not os.path.isdir(trackletdir):
        os.makedirs(trackletdir)
    original_imgdir = osp.join('..', data_flag)

    trackletnum = len(tracklet_lists)


    for i in range(trackletnum):
        tracklet = tracklet_lists[i]
        start_frame = tracklet[0]
        end_frame = tracklet[1]

        i_tracklet_name = "tracklet-{:04d}".format(i)
        i_tracklet_dir = osp.join(trackletdir, i_tracklet_name)
        if not os.path.isdir(i_tracklet_dir):
            os.makedirs(i_tracklet_dir)

        baseimages = namelist[start_frame-1:end_frame]
        img_len = len(baseimages)
        images = []
        for _, img in enumerate(baseimages):
            images.append(osp.join(original_imgdir, img))

        tic = time.time()
        for ind, img_path in enumerate(images):
            img = images[ind]
            ref_imgs = []

            if ind + margin < img_len:
                ref_imgs.append(images[ind + margin])
            if ind - margin >= 0:
                ref_imgs.append(images[ind - margin])

            basename = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            basepath = os.path.join(i_tracklet_dir, basename)

            # execution
            bashline = "./deepmatching" + " " + img + " " + ref_imgs[0] + " " + "-png_settings" + \
                       " " + "-out" + " " + " " + basepath

            process = subprocess.Popen(bashline.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
        toc = time.time()
    print("{:.2f} min, {:.2f} fps".format((toc - tic) / 60., 1. * len(images) / (toc - tic)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=int, default=2, help='margin of frame to compute the boundary.')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    args = parser.parse_args()
    main(args)
