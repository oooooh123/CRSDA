from ultralytics import YOLO

model=YOLO(r"D:\pycharm_object\ultralytics-main\ours_best.pt") #权重路径
results=model.val(source=r"D:\pycharm_object\ultralytics-main\datasets\test",data=r'D:\pycharm_object\ultralytics-main\datasets\bdz.yaml',)
#测试图片路径


