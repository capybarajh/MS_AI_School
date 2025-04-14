#### cvat
import os
import glob
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO
"""
<annotations>
    <image id="0" name="IMG_4913_JPG_jpg.rf.4f67c223e9cbf0ed07236bfe142aaaee.jpg" width="1920" height="1080">
    <box label="bond_1" source="manual" occluded="0" xtl="1026.81" ytl="324.65" xbr="1309.74" ybr="479.46" z_order="0"> </box>
    </image>
    <image id="1" name="IMG_4913_JPG_jpg.rf.4f67c223e9cbf0ed07236bfe142aaaee.jpg" width="1920" height="1080">
    <box label="bond_1" source="manual" occluded="0" xtl="1026.81" ytl="324.65" xbr="1309.74" ybr="479.46" z_order="0"> </box>
    </image>
</annotations>

"""
# model load
model = YOLO("./runs/detect/train11/weights/best.pt")
data_path = "./data/test/"
data_path_list = glob.glob(os.path.join(data_path, "*.jpg"))

tree = ET.ElementTree()
root = ET.Element("annotations")
"""
1. 
<annotations>
</annotations>
"""
id_number = 0
xml_path = "./test.xml"

for path in data_path_list :
    names = model.names
    results = model.predict(path, save=False, imgsz=640, conf=0.5)
    boxes = results[0].boxes
    box_info = boxes
    box_xyxy = box_info.xyxy
    cls = box_info.cls
    image = cv2.imread(path)
    img_height, img_width, _ = image.shape
    file_name = os.path.basename(path)
    print(file_name, img_height, img_width)

    xml_frame = ET.SubElement(root, "image", id="%d" % id_number, name=file_name, width="%d" % img_width, height="%d" % img_height)
    """
      2. 
      <annotations>
          <image id="0" name="IMG_4913_JPG_jpg.rf.4f67c223e9cbf0ed07236bfe142aaaee.jpg" width="1920" height="1080">
      </annotations>
     """
    for bbox, class_number in zip(box_xyxy, cls) :
        class_number = int(class_number.item())
        class_name_temp = names[class_number]
        print(class_name_temp)
        """
          3. 
          <annotations>
              <image id="0" name="IMG_4913_JPG_jpg.rf.4f67c223e9cbf0ed07236bfe142aaaee.jpg" width="1920" height="1080">
                <box label="bond_1" source="manual" occluded="0" xtl="1026.81" ytl="324.65" xbr="1309.74" ybr="479.46" z_order="0"> </box>
          </annotations>
         """
        x1 = int(bbox[0].item())
        y1 = int(bbox[1].item())
        x2 = int(bbox[2].item())
        y2 = int(bbox[3].item())
        ET.SubElement(xml_frame, "box", label=str(class_name_temp), source="manual",occluded="0",
                      xtl=str(x1), ytl=str(y1), xbr=str(x2), ybr=str(y2), z_order="0")

    id_number +=1
    tree._setroot(root)
    tree.write(xml_path, encoding="utf-8")



