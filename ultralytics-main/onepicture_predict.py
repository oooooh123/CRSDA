from ultralytics import YOLO

# 加载模型
model = YOLO(r"D:\pycharm_object\ultralytics-main\runs\duibistudy\ours_best.pt ") #权重路径
model.predict(r"D:\pycharm_object\ultralytics-main\datasets\predict\images\biandian_00332.jpg" , save=True, line_width=10,) #预测图片路径