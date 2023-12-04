import cv2
import matplotlib.pylab as plt
import matplotlib.cm as cm
from .model import StyleTransfer
"""
"gogh 1~4"
"kimhongdo 1"
"oil_paint 1~2"
"cartoon 1" 
"custom"

"k_means"
"in"
"black_and_white"
"bit"
"""
class Main():
  def __init__(self,filterName="gogh", image_path="D:/kivy/CANVAS_Project/data/content/con11.jpg"):
    self.filterName = filterName
    self.contentPath=str(image_path)
    self.content_img = cv2.imread(self.contentPath)
    self.content_img = cv2.cvtColor(self.content_img,cv2.COLOR_BGR2RGB)
    self.filterType = 1
    fmodel = StyleTransfer(filter = self.filterName, 
                            filterType = self.filterType,
                            non_filtering_area = [[0,0],[0,0]]) # 0~500 y,x
    res = fmodel.forward(self.content_img)
    
    if self.filterName == "black_and_white":
        plt.imshow(res,cmap = cm.gray)
        # plt.show()
        plt.savefig(f'CANVAS_Project/log/{self.filterName}.png',bbox_inches='tight', pad_inches=0)
    else:
        plt.imshow(res)
        plt.axis('off')
        # plt.show()
        plt.savefig(f'D:/kivy/CANVAS_Project/log/{self.filterName}.png',bbox_inches='tight', pad_inches=0)

# Main()