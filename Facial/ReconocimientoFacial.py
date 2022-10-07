import cv2
import os
from utils.metrics.Timer import Timer
from utils.metrics.measuring import tracing_start, tracing_mem

def write_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line)
def Recon_Facial():
	n = 2
	run = 'run_1_1'
	timer = Timer()
	timer.start()
	tracing_start()
	base_path = os.path.join('Facial/', run)
	dataPath = 'Facial/Data' #Cambia a la ruta donde hayas almacenado Data
	imagePaths = os.listdir(dataPath)
	print('imagePaths=',imagePaths)

	face_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Leyendo el modelo

	face_recognizer.read('Facial/modeloLBPHFace.xml')

	#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
	cap = cv2.VideoCapture('Facial/Angel.mp4')

	faceClassif = cv2.CascadeClassifier('Facial/haarcascade_frontalface_default.xml')

	while True:
		ret,frame = cap.read()
		if ret == False: break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = gray.copy()

		faces = faceClassif.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
			result = face_recognizer.predict(rostro)

			cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

			# LBPHFace
			if result[1] < 70:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			else:
				cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

		cv2.imshow('frame',frame)
		k = cv2.waitKey(1)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
	timer.end()
	print(f'Processing time": {timer.calculate_time()}')
	peak = tracing_mem()
	write_txt(os.path.join(base_path, f'results_{n}.txt'),
			  [f'Processing time: {timer.calculate_time()}', f'Peak size in MB: {peak}'])

