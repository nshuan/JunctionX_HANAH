from overlap import overlap
import argparse, os
import cv2

def test(test_dir, output, down_scale, frame_count):
    video_path = os.sep.join([test_dir, 'videos'])
    for scene, folder in enumerate(os.listdir(video_path)):
        if not os.path.exists(os.sep.join([output, f'SCENE_{scene+1}'])):
            os.mkdir(os.sep.join([output, f'SCENE_{scene+1}']))
        cam_type = folder[5:-2]
        cam = len(os.listdir(os.sep.join([video_path, folder])))
        cap = []
        img = [None] * cam
        for camera in os.listdir(os.sep.join([video_path, folder])):
            cap.append(cv2.VideoCapture(os.sep.join([video_path, folder,camera])))
            if not os.path.exists(os.sep.join([output, f'SCENE_{scene+1}',camera[:-3]])):
               os.mkdir(os.sep.join([output, f'SCENE_{scene+1}',camera[:-3]]))
        k = 0
        while True and k < frame_count:
            k += 1
            for i in range(cam):
                _, img[i] = cap[i].read()
            imgs, t, poly = overlap(img, downscale=down_scale, log=True)
            for i in range(cam):
                cv2.imwrite(os.sep.join([output, f'SCENE_{scene+1}',f'CAM_{i+1}', f'frame_{k}.jpg']), imgs[i])
                with open(os.sep.join([output, f'SCENE_{scene+1}',f'CAM_{i+1}', f'CAM_{i+1}.txt']), 'a+t') as f:
                    f.write(f'frame_{k}.jpg, {tuple(x for x in poly[i].flatten())}, {t}\n')
            print(f"Time: {t}s")

def main():
    args = argparse.ArgumentParser()
    args.add_argument( '-d','--test_dir', required=True, help="Path to test folder")
    args.add_argument( '-o','--output', default='./HANAH')
    args.add_argument( '-s','--down_scale', default=1, type=int)
    args.add_argument( '-f','--frame_count', default=15, type=int)
    args = args.parse_args()
    test(args.test_dir, args.output, args.down_scale, args.frame_count)

if __name__ == '__main__':
    main()
