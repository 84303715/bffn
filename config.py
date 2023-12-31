from collections import OrderedDict
from bidict import bidict


IMG_SIZE = (224, 224)  # width, height, cv2.resize will use this




#key is ROI number
ROI_LANDMARK = OrderedDict({"1": ["17u","19u","19","17"], #eye brow
                "2": ["19u","21u","21","19"],
                "3": ["21u", "27uu","27","21"],
                "4": ["27uu","22u","27","22"],
                "5": ["22u","24u","24","22"],
                "6": ["24u","26u","26","24"],
                #eye and temple
                "7": ["1", "17", "36"],
                "8": ["17","19","37","36"],
                "9": ["19","38","39","21"],
                "10":["21","27","28","39"],
                "11":["22","27","28", "42"],
                "12":["22","42","43","24"],
                "13":["24","44","45","26"],
                "14":["26","16","15","45"],
                #middle
                "15":["2","41~2","3~29","3"],
                "16":["41","2~41","3~29","39"],
                "17":["39","3~29","29","28"],
                "18":["28","29","13~29","42"],
                "19":["42","46","14~46","13~29"],
                "20":["14~46","14","13","13~29"],
                #middle down
                "21":["3","3~29","4~33","4"],
                "22":["4~33","3~29","29","33"],
                "23":["33","29","13~29","12~33"],
                "24":["13~29","13","12","12~33"],
                #mouse
                "25":["4","4~33","5~59","5"],
                "26":["5~59","4~33","33","59"],
                "27":["33","55","11~55","12~33"],
                "28":["11~55","12~33","12","11"],
                #below mouse
                "29":["59","58","6~58","5~59"],
                "30":["6~58","58","57","8~57"],
                "31":["57","8~57","10~56","56"],
                "32":["56","10~56","11~55","55"],
                #chin
                "33":["5","5~59","6~58","6"],
                "34":["6","6~58","8~57","8"],
                "35":["8~57","10~56","10","8"],
                "36":["10~56","10","11","11~55"],
                #miss part mouse
                "37":["33","55","56","57","58","59"],
                "38":["1","2","36","37"],
                "39":["14","15","44","45"],
                "40":["19","37","38"],
                "41":["24","43","44"],
                "42": ["36","37", "38", "39", "40", "41"], # left eye
                "43": ["42","43","44","45","46","47"]
                })

LABEL_FETCH = {
    ("4",) :[("1","2","5","7"),],
    ("6",):[("9", ),],
    ('10', '11', '12', '13', '14', '15'): [('18', '22', '23', '24', '28'),('16', '20', '25', '26', '27')],
    ('16', '20', '25', '26', '27'): [('10', '11', '12', '13', '14', '15'),],
}

AU_ROI = OrderedDict({"1":[1, 2, 8, 9, 5, 6, 12, 13, 40, 41, 42, 43],  # 修改，增加眼睛部分，与5和7一致
                      "2":[1, 2, 8, 9, 5, 6, 12, 13, 40, 41, 42, 43],  # 修改与1一样
                      "4":[1, 2, 3, 4, 8, 9, 5, 6, 12, 13, 40, 41],  # 增加3和4：皱眉头的眉头部分
                      "5":[1, 2, 8, 9, 5, 6, 12, 13, 40, 41, 42, 43], # 整个不对，改掉，应该是眼睛和眉毛部分
                      "6":[42,43,16,17,18,19],# 修改，增加眼睛和颧骨部分，颧骨外侧删掉
                      "7":[1, 2, 8, 9, 5, 6, 12, 13, 40, 41, 42, 43], # 修改
                      "12":[21,22,23,24,25,26,27,28,37], # 重定义
                      "15":[21,22,23,24,25,26,27,28,37], # 修改为与12一致
                      "16":[25,26,27,28,29,30,31,32,33,34,35,36,37], # 删掉22和23
                      "20":[25,26,27,28,29,30,31,32,33,34,35,36,37],
                      "23":[26,27,37,29,30,31,32],
                      "26":[25,26,27,28,29,30,31,32,33,34,35,36,37], # 修改为27号合并，与AU25的区别是带下巴
                    })

AU_SQUEEZE = bidict({idx : str(AU) for idx, AU in enumerate(sorted(map(int, list(AU_ROI.keys()))))})


use_AU = ["1", "2", "4", "5", "6", "7", "9", "12", "15", "16", "20", "23", "26"]


