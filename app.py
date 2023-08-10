import cv2
import os
from flask import Flask,request,render_template,redirect
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib


app = Flask(__name__)



datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)



if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('username,password,time')
if not os.path.isdir('user_database'):
    os.makedirs('user_database')
if f'user_database.csv' not in os.listdir('user_database'):
    with open(f'user_database/user_database.csv','w') as f:
        f.write('username','password')


def totalreg():
    return len(os.listdir('static/faces'))


#Face Segmentation
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#Face Identification
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


def extract():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['username']
    rolls = df['password']
    times = df['time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user
def add(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    
    with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
        f.write(f'\n{username},{userid},{current_time}')


def verfiy_User(username,password,identified_person):
    un = identified_person.split('_')
    if username == un[0]:
        df = pd.read_csv(f'user_database/user_database.csv')
        dbPass=df[df['username']==username]
        dbPass.reset_index(inplace = True)
        if len(dbPass.index) != 0:
            if dbPass["password"][0] == password:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html') 

@app.route('/enroll')
def enroll():
    return render_template('/sign_up.html')

@app.route('/tryAgain')
def tryagain():
    return render_template('home.html')

# Verification / Login Page
@app.route('/start',methods=['GET','POST'])
def start():
    username = request.form['username']
    password = request.form['userid']
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 
    userimagefolder = 'static/faces/'+username+'_'+str(password)
    
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            un = identified_person.split('_')
            df = pd.read_csv(f'user_database/user_database.csv')
            dbPass=df[df['username']==username]
            dbPass.reset_index(inplace = True)
            isVerfied = True
            if len(dbPass.index) != 0:
                if username == un[0]:
                    if un[1] == password:
                        isVerfied = True
            if isVerfied:
                cv2.putText(frame,f'{un[0]},VERIFIED PRESS ESC TO CONTINUE',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if cv2.waitKey(1)==27:
                    add(identified_person)
                    return render_template("sucess.html")
                    break        
            else:
                cv2.putText(frame,'FAILED VERIFIED PRESS ESC TO CONTINUE',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if cv2.waitKey(1)==27:
                    return render_template("fail.html")
                    break                  
        cv2.imshow('Verify',frame)
        if cv2.waitKey(1)==27:
            return render_template("fail.html")
            break
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract()    
    return render_template('home.html') 

#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    with open(f'user_database/user_database.csv','a') as f:
        f.write(f'\n{newusername},{newuserid}')
    
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html') 




#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
