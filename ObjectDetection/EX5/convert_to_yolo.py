import os 
from xml.etree.ElementTree import parse

class InfraredImageProcessor :
    def __init__(self, xml_folder_path) : 
        self.xml_folder_path = xml_folder_path
        self.label_dict = {'bond' : 0}
    
    def find_xml_file(self) : 
        all_root = []
        for (path, dir, files) in os.walk(self.xml_folder_path) : 
            print(path, dir, files)
            for filename in files : 
                ext = os.path.splitext(filename)[-1]
                # ext -> .xml
                if ext == '.xml' :
                    root = os.path.join(path, filename)
                    all_root.append(root)

        return all_root
    
    def process_images(self): 
        xml_dirs = self.find_xml_file()
        for xml_dir in xml_dirs : 
            tree = parse(xml_dir)
            root = tree.getroot()
            img_metas = root.findall('image')
            for img_meta in img_metas :
                try : 
                    # 키포인트 라벨별 리스트 
                    head = []
                    tail = []

                    # xml 이미지 이름 가져오기 
                    image_name = img_meta.attrib['name']
                    image_name = image_name.replace('.jpg', '.txt')

                    # Box Meta 
                    box_metas = img_meta.findall('box')
                    point_metas = img_meta.findall('points')
                    img_width = int(img_meta.attrib['width'])
                    img_height = int(img_meta.attrib['height'])

                    for point_meta in point_metas : 
                        point_label = point_meta.attrib['label']
                        point_x = float(point_meta.attrib['points'].split(',')[0])
                        point_y = float(point_meta.attrib['points'].split(',')[1])

                        if point_label == 'head' : 
                            head = point_x, point_y
                        elif point_label == 'tail' : 
                            tail = point_x, point_y

                    for box_meta in box_metas : 
                        box_label = box_meta.attrib['label']
                        box = [int(float(box_meta.attrib['xtl'])), 
                               int(float(box_meta.attrib['ytl'])),
                               int(float(box_meta.attrib['xbr'])),
                               int(float(box_meta.attrib['ybr']))]
                        print(box)
                        if box_label == 'ignore' : 
                            pass
                        
                        box_x = round(((box[0] + box[2]) / 2) / img_width, 6)
                        box_y = round(((box[1] + box[3]) / 2) / img_height, 6)
                        box_w = round((box[2] - box[0]) / img_width, 6)
                        box_h = round((box[3] - box[1]) / img_height, 6)

                        head_x_temp = round(head[0] / img_width, 6)
                        head_y_temp = round(head[1] / img_height, 6)
                        tail_x_temp = round(tail[0] / img_width, 6)
                        tail_y_temp = round(tail[1] / img_height, 6)

                        label_number = self.label_dict[box_label]

                        change_fore_ocuure_temp = 2 
                        os.makedirs("./dataset/val/labels/", exist_ok=True)
                        with open(f"./dataset/val/labels/{image_name}" , 'a' ) as f :
                            f.write(
                                f"{label_number} {box_x:.6f} {box_y:.6f} {box_h:.6f} {box_w:.6f} {head_x_temp:.6f} {head_y_temp:.6f} {change_fore_ocuure_temp} {tail_x_temp:.6f} {tail_y_temp:.6f} {change_fore_ocuure_temp}\n" 
                            )


                except Exception as e : 
                    pass

test = InfraredImageProcessor("./annotation/val")
test = test.process_images()