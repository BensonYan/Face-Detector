from tqdm import tqdm

try:
    import cv2
except Exception:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
import dlib
from math import ceil, floor
from os import mkdir, listdir, rename, makedirs, system, remove, walk
from os.path import exists, getsize, join
import argparse
from random import uniform
import os
from config import config_dict


class FaceExt:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(config_dict["faceProcessor"]["modelPath"])
        self.name_list = []
        self.cwd = os.getcwd()

    def extract(self, videoPath, num, savePath="data"):
        if not exists(savePath):
            mkdir(savePath)
        video = cv2.VideoCapture(videoPath)
        videoName = videoPath.split('/')[-1].split('.')[0]
        print("videoName: "+videoName)
        indicator = tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))))
        if video.isOpened():
            videoHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            videoWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            while video.isOpened():
                ret, frame = video.read()
                if ret:
                    indicator.update()
                else:
                    break
                image = self.extractFromPic2(frame, videoWidth, videoHeight,videoName)
                if image is not None:
                    cv2.imshow(args.type, image)
                    cv2.imwrite(savePath + "/" + videoName + "_" + str(indicator.n) + ".png", image)
                    cv2.waitKey(1)
                    if indicator.n == num:
                        break
                else:
                    continue
            indicator.close()
        else:
            print("Fail to open video -> " + videoPath)

    def get_boundingbox(self, face, width, height, scale=1.3, minsize=None):
        """
        Expects a dlib face to generate a quadratic bounding box.
        :param face: dlib face class
        :param width: frame width
        :param height: frame height
        :param scale: bounding box size multiplier to get a bigger face region
        :param minsize: set minimum bounding box size
        :return: x, y, bounding_box_size in opencv form
        """
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        if minsize:
            if size_bb < minsize:
                size_bb = minsize
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Check for out of bounds, x-y top left corner
        x1 = max(int(center_x - size_bb // 2), 0)
        y1 = max(int(center_y - size_bb // 2), 0)
        # Check for too big bb size for given x, y
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        return x1, y1, size_bb

    def extractFromPic_All(self, image):
        height, width, channels = image.shape
        return_face = []
        if image is not None:
            faceImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.detector(faceImage, 1)
            for face in faces:
                x, y, size = self.get_boundingbox(face, width, height)
                faceImage = cv2.resize(image[y:y + size, x:x + size],
                                              config_dict["faceProcessor"]["imageSize"])

                return_face.append(cv2.cvtColor(faceImage, cv2.COLOR_RGB2BGR))
            return return_face
        else:
            return None

    def extractFromPic2(self, image, width, height,v_name):
        if image is not None:
            faceImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            try:
                rect = self.detector(faceImage)[0]
            except IndexError as reason:
                print(reason)
# =============================================================================
#                 print(name)
#                 self.name_list.append(name)
#                 f= open(self.cwd+"/name_v3.txt", "w")
#                 for element in self.name_list:
#                     f.write(element + "\n")
#                 f.close()
# =============================================================================
                rect = None
                pass			
            if rect != None:
                try:                
	                faces = self.detector(faceImage, 1)
# =============================================================================
# 	                if len(faces) > 1:
# 	                    faceArea = []
# 	                    for face in faces:
# 	                        area = (face.bottom() - face.top()) * (face.right() - face.left())
# 	                        faceArea.append(area)
# 	                        face = faces.index(max(faceArea))
# 	                else:
# 	                    face = faces[0]
# =============================================================================
	                face = faces[0]
	                x, y, size = self.get_boundingbox(face, width, height)
	                return cv2.resize(image[y:y + size, x:x + size], config_dict["faceProcessor"]["imageSize"])
                except Exception as reason:
	                print(reason)
	                if v_name not in self.name_list:
	                    self.name_list.append(v_name)
	                    f= open(self.cwd+"/c40_face2face_videoname.txt", "w")
	                    for element in self.name_list:
	                        f.write(element + "\n")
	                    f.close()
	                return None
            else:
                return None
        else:
            return None

    def extractFromPic(self, image):
        if image is not None:
            faceImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            try:
                faces = self.detector(faceImage, 1)

                if len(faces) > 1:
                    faceArea = []
                    for face in faces:
                        area = (face.bottom() - face.top()) * (face.right() - face.left())
                        faceArea.append(area)
                        face = faces.index(max(faceArea))
                else:
                    face = faces[0]
                if face.bottom() - face.top() > 40:
                    shape = self.predictor(image, face)

                    nosePoint = (shape.part(30).x, shape.part(30).y)
                    leftPoint = (shape.part(17).x, shape.part(17).y)
                    rightPoint = (shape.part(26).x, shape.part(26).y)
                    chinPoint = (shape.part(8).x, shape.part(8).y)

                    nose2top = int(2.3 * (nosePoint[1] - leftPoint[1]))
                    nose2bottom = int(1.02 * chinPoint[1] - nosePoint[1])
                    height = nose2top + nose2bottom

                    nose2left = int(1.5 * (nosePoint[0] - leftPoint[0]))
                    nose2right = int(1.5 * (rightPoint[0] - nosePoint[0]))
                    width = nose2left + nose2right

                    if height > width:
                        diff = height - width
                        nose2left += ceil(diff / 2)
                        nose2right += floor(diff / 2)

                        image = image[nosePoint[1] - nose2top:nosePoint[1] + nose2bottom,
                                nosePoint[0] - nose2left:nosePoint[0] + nose2right]
                        return cv2.resize(image, config_dict["faceProcessor"]["imageSize"])

            except Exception:
                return None
        else:
            return None

class FaceData:

    def __init__(self):
        self.faceExt = FaceExt()

    def createDir(self, path):
        if not exists(path):
            makedirs(path)

    def extract(self, extractType):
        self.createDir(config_dict["faceProcessor"]["out"][extractType])
        for videoPath in config_dict["faceProcessor"]["rawVideo"][extractType]:
            print("Extracting " + extractType)
            indicator = tqdm(listdir(videoPath))
            for video in listdir(videoPath):
                indicator.update()
                if video.split('.')[-1] == "mp4":
                    self.faceExt.extract(videoPath=videoPath + video,
                                         num=config_dict["faceProcessor"]["amountEachVideo"],
                                         savePath=config_dict["faceProcessor"]["out"][extractType])

    def ex(self, extractType,name):
        #self.createDir(config_dict["faceProcessor"]["out"][extractType])
        for videoPath in config_dict["faceProcessor"]["rawVideo"][extractType]:
            print("Extracting " + extractType)
            indicator = tqdm(listdir(videoPath))
            indicator.update()
            if video.split('.')[-1] == "mp4":
                self.faceExt.extract(videoPath=videoPath + name,
                num=config_dict["faceProcessor"]["amountEachVideo"],
                savePath=config_dict["faceProcessor"]["out"][extractType])
                print(config_dict["faceProcessor"]["amountEachVideo"])

    def extractFolder(self, inFolder, outFolder):
        rejectFolder = "../test"
        # if not exists(outFolder):
        #     makedirs(outFolder)
        # if not exists(rejectFolder):
        #     makedirs(rejectFolder)
        # for each in listdir(inFolder):
        #     image = cv2.imread(join(inFolder, each))
        #     height, width, channels = image.shape
        #     eImage = self.faceExt.extractFromPic2(image, width, height)
        #     if eImage is not None:
        #         cv2.imwrite(join(outFolder, each), eImage)
        #     else:
        #         cv2.imwrite(join(rejectFolder, each), image)

        for each in listdir(rejectFolder):
            image = cv2.imread(join(inFolder, each))
            height, width, channels = image.shape
            eImage = self.faceExt.extractFromPic2(image, width, height)
            if eImage is not None:
                cv2.imwrite(join("../", each), eImage)
            else:
                # print(each)
                pass


    def countImages(self, folder):
        count = 0
        size = 0
        if exists(folder):
            for root, directories, images in walk(folder):
                if len(directories) > 0:
                    for d in directories:
                        for image in listdir(folder+d):
                            if image.split('.')[-1] == "png":
                                count += 1
                                size += getsize(folder + join(d,image))
                else:
                    for image in listdir(folder):
                        if image.split('.')[-1] == "png":
                            count += 1
                            size += getsize(folder + image)
        return count, size * 1e-9  # size in GB
    def checkDuplicate(self, datapath, Type,name_list):

        if Type == 'fake':
	        for _, _, files in walk(datapath):				
		        for image in files:
		            name = image.split('_')[:3]
		            if name in name_list:
		               continue
		            else:
		               name_list.append(name)
        else:
	        for _, _, files in walk(datapath):				
		        for image in files:
		            name = image.split('_')[:2]
		            if name in name_list:
		               continue
		            else:
		               name_list.append(name)				   
        return name_list
    def recover(self):
        print("Recovering...")
        self.__recover(setType="train", recoverType=config_dict["dataset"]["target"][0])
        self.__recover(setType="train", recoverType=config_dict["dataset"]["target"][1])
        self.__recover(setType="test", recoverType=config_dict["dataset"]["target"][0])
        self.__recover(setType="test", recoverType=config_dict["dataset"]["target"][1])
        self.sum()

    def __recover(self, setType, recoverType):		
        if recoverType == "fake":
            if config_dict["faceProcessor"]["out"][recoverType].split('/')[-2] != "fake":
                recoverType = join(recoverType,config_dict["faceProcessor"]["out"][recoverType].split('/')[-2])
        print(recoverType)
        datasetPath = join(config_dict["dataset"][setType], recoverType)
        self.createDir(datasetPath)
        print(datasetPath)
        if recoverType!='fake' and recoverType!='real' :
            recoverType = 'fake'
# =============================================================================
# 
#         for image in listdir(datasetPath):
#             if image.split('.')[-1] == "png":
#                 rename(join(datasetPath, image),
#                        join(config_dict["faceProcessor"]["out"][recoverType], image))
# =============================================================================

    def split(self, random):
        self.recover()
        #self.__split(random, organizeType=config_dict["dataset"]["target"][0])
        self.__split(random, organizeType=config_dict["dataset"]["target"][1])
        self.sum()

    def __split(self, random, organizeType):
        #rootImagepath = '../../Bosheng/ff_c40/combine/train/' + organizeType + "Deepfakes"
        #print(rootImagepath)
        setType = 'train'
        #print("Classifying " + organizeType)
        imageFolder = config_dict["faceProcessor"]["out"][organizeType]
        folderType = imageFolder.split('/')[-2]
        if organizeType == "fake":
            if folderType == "Deepfakes":
                organizeType = join(organizeType,folderType)
                self.createDir(join(config_dict["dataset"][setType], organizeType))
            elif folderType == "Face2Face":
                organizeType = join(organizeType,folderType)
                self.createDir(join(config_dict["dataset"][setType], organizeType))
            elif folderType == "FaceSwap":
                organizeType = join(organizeType,folderType)
                self.createDir(join(config_dict["dataset"][setType], organizeType))				
            elif folderType == "NeuralTextures":
                organizeType = join(organizeType,folderType)
                self.createDir(join(config_dict["dataset"][setType], organizeType))		
            rootImagepath = '../../Bosheng/ff_c40/combine/' + setType + '\\' + organizeType
        else:
            rootImagepath = '../../Bosheng/ff_c40/combine/' + setType + '\\' + organizeType
        print("root Image path:", rootImagepath)
        total = self.countImages(imageFolder)[0]
        indicator = tqdm(range(total))
        #duplicateName = self.checkDuplicate(rootImagepath, organizeType)
        name_list = []
        for image in listdir(imageFolder):
            #duplicateName = self.checkDuplicate(rootImagepath, organizeType,name_list)
            if organizeType ==  'fake' or folderType == "Deepfakes" or folderType == "Face2Face" or folderType == "FaceSwap" or folderType == "NeuralTextures" :
            	checkName = image.split('_')[:2]
            else:			
            	checkName = image.split('_')[0]
            indicator.update()
            imagePath = join(imageFolder, image)
            if not self.verifyImage(imagePath):
            	continue
            number = uniform(0, 1) if random else indicator.n / total
            if number >= config_dict["dataset"]["train_proportion"] and image.split('.')[-1] == "png":
            	if checkName not in name_list:            	
            	    rename(imagePath, join(config_dict["dataset"]["test"], organizeType, image))
            	    #None
            	else:
            	    continue   
            	#break
            else:
	            name_list.append(checkName)									
	            rename(imagePath, join(config_dict["dataset"]["train"], organizeType, image))

    def verifyImage(self, path2Image):
        if cv2.imread(path2Image) is None:
            print("This image is problematic and will be deleted -> " + path2Image)
            remove(path2Image)
            return False
        else:
            return True

    def combineLink(self, dataset: list, ouput_path):
        if not exists(ouput_path):
            makedirs(ouput_path)
        count = 0
        for each in dataset:
            for idx, image in enumerate(listdir(each)):
                ramNum = uniform(0, 1)
                proportion = 1 / len(dataset)
                if ramNum < proportion:
                    cmd = "ln -s " + str(join(each, image)) + " " + str(join(ouput_path, str(count)+".png"))
                    system(cmd)
                    count += 1
                if idx % 5000 == 0:
                    print(str(join(each, image)))
                    print(str(join(ouput_path, str(count)+".png")))
                    print("Loop " + str(idx) + ", linked " + str(count) + " images.")

    def sum(self):
        print("********************************************")
        print("*               Amount   Size (GB)")
        amount, size = self.countImages(join(config_dict["dataset"]["train"], "real/"))
        print("* Train real:   {}       {:.3f}".format(amount, size))
        amount, size = self.countImages(join(config_dict["dataset"]["train"], "fake/"))
        print("* Train fake:   {}       {:.3f}".format(amount, size))
        amount, size = self.countImages(join(config_dict["dataset"]["test"], "real/"))
        print("* Test  real:   {}       {:.3f}".format(amount, size))
        amount, size = self.countImages(join(config_dict["dataset"]["test"], "fake/"))
        print("* Test  fake:   {}       {:.3f}".format(amount, size))
        amount, size = self.countImages(config_dict["faceProcessor"]["out"]["real"])
        print("* origin  real:   {}       {:.3f}".format(amount, size))
        amount, size = self.countImages(config_dict["faceProcessor"]["out"]["fake"])
        print("* origin  fake:   {}       {:.3f}".format(amount, size))
        print("********************************************")

    def putLabel(self):
        image = cv2.imread("../fake.png")
        height, width, channels = image.shape
        if image is not None:
            faceImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.faceExt.detector(faceImage, 1)
            for face in faces:
                x, y, size = self.faceExt.get_boundingbox(face, width, height)
                # faceImage = cv2.resize(image[y:y + size, x:x + size],
                #                               config_dict["faceProcessor"]["imageSize"])

                cv2.rectangle(image, (x, y), (x + size, y + size), (0, 0, 255), 5)
                cv2.putText(image,
                            "Fake",
                            (x, y-20),
                            cv2.FONT_HERSHEY_COMPLEX,
                            2,
                            (0, 0, 255)
                            ,5)
        else:
            return None

        cv2.imwrite("../FakeLabel.png", image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--extractFolder', action='store_true')
    parser.add_argument('--type', type=str, default=config_dict["dataset"]["target"][0])  # default real
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--sum', action='store_true')
    parser.add_argument('--putLabel', action='store_true')
    parser.add_argument('--combine', action="store_true")
    parser.add_argument('-l', '--list', nargs='+')
    parser.add_argument('-o', type=str, default="../data/combine/train/fake")
    args = parser.parse_args()
    video = "000.mp4"
    faceData = FaceData()
    if args.sum:
        faceData.sum()
    elif args.extract:
        assert args.type != config_dict["dataset"]["target"][0] or \
               args.type != config_dict["dataset"]["target"][1]
        faceData.extract(extractType=args.type)
    elif args.split:
        faceData.split(random=args.random)
    elif args.recover:
        faceData.recover()
    if args.combine:
        faceData.combineLink(args.list, args.o)
    if args.extractFolder:
        faceData.extractFolder(inFolder="../faceforensics_benchmark_images/",
                               outFolder="../out")

    if args.putLabel:
        faceData.putLabel()
    #faceData.ex(extractType="fake",name='011_805.mp4' )
    #faceData.split(random=False)