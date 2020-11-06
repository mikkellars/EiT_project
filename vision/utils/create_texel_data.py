"""Scripts for creating texel image to use for binary classification
"""
import os
import numpy as np
import cv2 
import glob

class createTexel():
    def __init__(self, load_path, save_path):
        assert os.path.exists(load_path)
        assert os.path.exists(save_path)
        self.load_path = load_path
        self.save_path = save_path
        self.show_resolution = 512

        self.patch_num_texel, self.patch_num_no_texel = self.__get_cur_texel_n_notexel_idx()

        self.window_size = 128
        stride = self.stride = 4

        self.imgs = self.__load_imgs()

    # ------------------
    #  Public functions
    # ------------------
    def create_patch(self):
        winW = self.window_size
        winH = self.window_size

        x = 0
        y = 0

        btn_pressed = -1

        for img_path in self.imgs:
            ori_img = cv2.imread(self.load_path + "/" + img_path)
            ori_img =  cv2.resize(ori_img, (self.show_resolution,self.show_resolution))
            btn_pressed = -1
            print("Current image: ", img_path)
            while(btn_pressed != 110): # btn n --> next image
                img = ori_img.copy()
                cv2.rectangle(img, (x, y), (x + winW, y + winH), (0, 255, 0), 2) # patch retangle on img

                patch = ori_img[y:y+winH, x:x+winW]
                show_patch =  cv2.resize(patch, (self.show_resolution,self.show_resolution))
                cv2.drawMarker(show_patch, (show_patch.shape[1]//2, show_patch.shape[0]//2), (0, 0, 255), 1, 40, 4)# Red center

                total_img = np.hstack((img, show_patch))
                cv2.imshow("Window", total_img)
                btn_pressed = cv2.waitKey()
                
                if btn_pressed == 113: # btn q --> quit 
                    return

                x, y = self.__window_pos(btn_pressed, x, y)

                if btn_pressed == 49: # Key 1 --> Texel in middle of image
                    print(f'Img_{self.patch_num_texel} saved as Texel')
                    self.__save_patch(patch, self.patch_num_texel, texel = True)
                    self.patch_num_texel += 1
                elif btn_pressed == 48: # Key 0 --> No texel in middle of image
                    print(f'Img_{self.patch_num_no_texel} saved as Non-Texel')
                    self.__save_patch(patch, self.patch_num_no_texel, texel = False)
                    self.patch_num_no_texel += 1

    # ------------------
    # Private functions
    # ------------------
    def __load_imgs(self):
        imgs = os.listdir(os.path.abspath(self.load_path))
        imgs = list(filter(lambda file: file.endswith('.jpg'), imgs))
        return imgs


    def __save_patch(self, patch, patch_num, texel):
        if texel:
            filename = f'{self.save_path}/true/patch_{patch_num}.jpg'
            cv2.imwrite(filename, patch) 
        else:
            filename = f'{self.save_path}/false/patch_{patch_num}.jpg'
            cv2.imwrite(filename, patch) 


    def __get_cur_texel_n_notexel_idx(self):
        patch_num_texel = 0
        patch_num_no_texel = 0
        path_texel = f'{self.save_path}/true/'
        path_no_texel = f'{self.save_path}/false/'

        patch_num_texel = len(glob.glob1(path_texel,"*.jpg"))
        patch_num_no_texel = len(glob.glob1(path_no_texel,"*.jpg"))

        return patch_num_texel, patch_num_no_texel

    def __window_pos(self, btn_pressed:int, x:int, y:int):
        up_btn = 82
        down_btn = 84
        left_btn = 81
        right_btn = 83
        winW = self.window_size
        winH = self.window_size
        stride = self.stride

        if btn_pressed == up_btn and y > 0:
            if y - stride < 0:
                y = y - 1
            else:
                y = y - stride
        elif btn_pressed == down_btn and y + winH < self.show_resolution:
            if y + winH + stride > self.show_resolution:
                y = y + 1
            else:
                y = y + stride

        elif btn_pressed == left_btn and x > 0:
            if x - stride < 0:
                x = x - 1
            else:
                x = x - stride
        elif btn_pressed == right_btn and x + winW < self.show_resolution:
            if x + winW + stride > self.show_resolution:
                x = x + 1
            else:
                x = x + stride

        return x, y
        

def main():
    l_path = "/home/mikkel/Documents/experts_in_teams_proj/vision/data/fence_data/train_set/images"
    s_path = "/home/mikkel/Documents/experts_in_teams_proj/vision/data/fence_data/texel_data"

    create_data = createTexel(l_path, s_path)
    create_data.create_patch()
    print('works')


if __name__ == '__main__':
    main()