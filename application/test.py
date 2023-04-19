from overlap import overlap
import argparse
import os
import cv2


def test(test_dir, output, down_scale, frame_count, save_image):
    video_path = os.sep.join([test_dir, 'videos'])
    for scene, folder in enumerate(os.listdir(video_path)):
        if not os.path.exists(os.sep.join([output, f'{folder}'])):
            os.mkdir(os.sep.join([output, f'{folder}']))
        cam_type = folder[5:-2]
        cam = len(os.listdir(os.sep.join([video_path, folder])))
        cap = []
        img = [None] * cam
        for camera in os.listdir(os.sep.join([video_path, folder])):
            cap.append(cv2.VideoCapture(
                os.sep.join([video_path, folder, camera])))
        k = 0
        for i in range(cam):
            with open(os.sep.join([output, f'{folder}', f'CAM_{i+1}.txt']), 'w') as f:
                pass
        while True and k < frame_count:
            k += 1
            for i in range(cam):
                _, img[i] = cap[i].read()
            imgs, t, poly = overlap(img, downscale=down_scale, log=True)
            for i in range(cam):
                if save_image:
                    cv2.imwrite(os.sep.join(
                        [output, f'{folder}', f'frame_{k}.jpg']), imgs[i])
                with open(os.sep.join([output, f'{folder}', f'CAM_{i+1}.txt']), 'a+t') as f:
                    f.write(
                        f'frame_{k}.jpg, {tuple(int(x) for x in poly[i].flatten())}, {t}\n')
            print(f"Time: {t}s")


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--test_dir', required=True,
                      help="Path to test folder")
    args.add_argument('-o', '--output', default='./HANAH',
                      help="Output test folder")
    args.add_argument('-s', '--down_scale', default=1, type=int,
                      help="How many time shoud be reduce in width and height of the frame")
    args.add_argument('-f', '--frame_count', default=15,
                      type=int, help="Maximum number of frame to detect")
    args.add_argument('-i', '--save_image', default=False,
                      type=bool, help="Save each frame in detect or not")
    args.add_argument('-a', '--ground_truth_folder', default='',
                      type=str, help="Groundtruth folder to calculate the IoU")
    args = args.parse_args()
    test(args.test_dir, args.output, args.down_scale,
         args.frame_count, args.save_image)


if __name__ == '__main__':
    main()
