from roboflow import Roboflow
rf = Roboflow(api_key="nB0npeehP3uCXLFaKcq1")
project = rf.workspace("lahari-s").project("Helmet Detection_YOLOv8")
dataset = project.version(1).download("yolov8")