




import os
import numpy as np
import cv2
from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
from PIL import Image, ImageTk


root = Tk()
root.geometry('500x570')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.title('Predict Cam')
frame.config(background='white')

label = Label(frame, text="Predict Cam", bg='white', font=('Time 35 bold'))
label.pack(side=TOP)

file = 'ab.jpeg'
filename = Image.open(file)
filename = ImageTk.PhotoImage(filename.resize((1550, 1000)))
background_label = Label(frame, image=filename)
background_label.pack(side=TOP)


def hel():
    help(cv2)


def Contri():
    tkinter.messagebox.showinfo("Contributors", "\n1.Anchal Jain\n2. Ansh Jain \n3. Jigar Joshi \n")


def anotherWin():
    tkinter.messagebox.showinfo("About",
                                'Predict Cam version v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3')


menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools", menu=subm1)
subm1.add_command(label="Open CV Docs", command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About", menu=subm2)
subm2.add_command(label="Predict Cam", command=anotherWin)
subm2.add_command(label="Contributors", command=Contri)


def exitt():
    exit()


def webcam():

    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # video_capture = cv2.VideoCapture("/dev/video1")

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



subjects = ["", "AnshJain", "AnchalJain"]

def detect_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area

    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]

def detect_face1(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    g = []
    for xv in faces:
        (x, y, w, h) = xv
        g.append(gray[y:y + w, x:x + h])

    # return only the face part of the image
    return g, faces


def prepare_training_data(data_folder_path):

    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;


        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 0, 0), 3)



def face_count():
    file_path = filedialog.askopenfilename()
    file_path.replace('/', '//')


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=10)

    print
    type(faces)

    if len(faces) == 0:
        print
        "No faces found"

    else:
        print
        faces
        print
        faces.shape
        print
        "Number of faces detected: " + str(faces.shape[0])

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (350, 0, 255), 2)

        image = cv2.resize(image, (900, 600))
        cv2.rectangle(image, ((0, image.shape[0] - 25)), (270, image.shape[0]), (255, 255, 255), -1)

        cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow('Image with faces',image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def predict(test_img):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    print(img.shape)
    # detect face from the image
    face, rect = detect_face1(img)
    print(face)
    # predict the image using our face recognizer
    k = 0
    for x in face:
        label, confidence = face_recognizer.predict(face[k])
        print('Confidence = ', confidence)
        print('Label = ', label)
        # get name of respective label returned by face recognizer
        if confidence > 75.0:
            label_text = subjects[label]
        else:
            label_text = "Undefined"

        # draw a rectangle around face detected
        draw_rectangle(img, rect[k])
        # draw name of predicted person
        draw_text(img, label_text, rect[k][0], rect[k][1] - 5)
        k = k + 1

    return img



traincounter = 0

face_recognizer = "global"

def input_image():

    file_path = filedialog.askopenfilename()
    file_path.replace('/', '//')
    global traincounter
    global face_recognizer

    if traincounter == 0:
        traincounter = 1

        print("Preparing data...")
        faces, labels = prepare_training_data('training-data')
        print("Data prepared")

        # print total faces and labels
        print("Total faces: ", len(faces))
        print("Total labels: ", len(labels))


        face_recognizer = cv2.face.LBPHFaceRecognizer_create()



        # train our face recognizer of our training faces
        face_recognizer.train(faces, np.array(labels))




    print("Predicting images...")

    # load test images
    test_img1 = cv2.imread(file_path)

    # perform a prediction

    # test_img1 = cv2.resize(test_img1, (800, 1200))
    # test_img2 = cv2.resize(test_img2, (800, 1200))

    predicted_img1 = predict(test_img1)

    print("Prediction complete")

    # display both images
    cv2.imshow("Picture 1", cv2.resize(predicted_img1, (600, 800)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def age_gender():

    file_path = filedialog.askopenfilename()
    file_path.replace('/', '//')

    test_img1 = cv2.imread(file_path)
    # url = 'https://www.youtube.com/watch?v=nD1jhw6F-J4'
    # vPafy = pafy.new(url)
    # play = vPafy.getbest(preftype="mp4")
    # cap = cv2.VideoCapture(play.url)
    #
    # cap.set(3, 480)  # set width of the frame
    # cap.set(4, 640)  # set height of the frame

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Female', 'Male']

    # user defined functions
    def load_caffe_models():
        age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
        gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
        return (age_net, gender_net)

    def video_detector(age_net, gender_net):
        font = cv2.FONT_HERSHEY_SIMPLEX

    if __name__ == "__main__":
        age_net, gender_net = load_caffe_models()
        video_detector(age_net, gender_net)



    image = test_img1
    cv2.resize(image, (600, 1000))
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if (len(faces) > 0):
        print("Found {} faces".format(str(len(faces))))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Get Face
        face_img = image[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)
        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)
        overlay_text = "%s %s" % (gender, age)
        cv2.putText(image, overlay_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Picture", cv2.resize(image, (500, 900)))

        # 0xFF is a hexadecimal constant which is 11111111 in binary.
    cv2.waitKey(0)
    cv2.destroyAllWindows()



but1 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=webcam, text='Open Cam',
              font=('helvetica 15 bold'))
but1.place(x=5, y=104)

but2 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=input_image,
              text='Face Recognition', font=('helvetica 15 bold'))
but2.place(x = 5,y=176)

but3 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=age_gender,
              text='Detect Age and Gender', font=('helvetica 15 bold'))
but3.place(x=5, y=250)

but4 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=face_count,
              text='Face Count', font=('helvetica 15 bold'))
but4.place(x=5, y=322)

# but5 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=   ,
#               text='  ', font=('helvetica 15 bold'))
# but5.place(x=5, y=400)

but5 = Button(frame, padx=5, pady=5, width=5, bg='white', fg='black', relief=GROOVE, text='EXIT', command=exitt,
              font=('helvetica 15 bold'))
but5.place(x=210, y=478)

root.mainloop()


