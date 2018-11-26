import cv2
import numpy as np
import os


subjects=["","A","B","C","D","E","F","G"]
temp=[]

def predict(test_img):


# make a copy of the image as we don't want to change original image
    img = test_img.copy()
# detect face from the image
    face, rect = detect_face(img)


# predict the image using our face recognizer
    label =face_recognizer.predict(face)
    temp=label

# get name of respective label returned by face recognizer
    label_text = subjects[temp[0]]

# draw a rectangle around face detected
    draw_rectangle(img, rect)
# draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('D:\ImageRecognition\lbpcascade_frontalface.xml')
    print(face_cascade)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data(data_folder_path):


    dirs = os.listdir(data_folder_path)
    for i in dirs:
        print(i)

    faces = []
    labels = []
    images=[]

    for dir_name in dirs:

        if not dir_name.startswith("P"):
            continue


        label = int(dir_name.replace("P", ""))
        print(label)


        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)
        for i in subject_images_names:
            images.append(i)

        for image_name in images:

            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)

            if image is not None:


                cv2.imshow("Training on image...", image)
                cv2.waitKey(100)

                face, rect = detect_face(image)


                if face is not None:
                    faces.append(face)
                    labels.append(label)
                for i in labels:
                    print(i)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.destroyAllWindows()


    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("/ImageRecognition/train")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


print("Predicting images...")

# load test images
test_img1 = cv2.imread("try.jpg")


# perform a prediction
predicted_img1 = predict(test_img1)
print("Prediction complete")

# display both images
cv2.imshow(subjects[1], predicted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()