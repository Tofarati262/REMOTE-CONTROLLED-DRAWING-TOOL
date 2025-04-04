import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import threading
import serial.tools.list_ports
import queue

#-------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
print("Camera Opened:", cap.isOpened())

#-------------------------------------------------------

def draw_rects(screen, rects):
    for rect in rects:
        pygame.draw.rect(screen, pygame.Color(0,255,0), rect)

def run_face_tracking(cap, mp_face_mesh, face_mesh, drawing_spec, message_queue):
    rects = []
    noise_tunning = 0.5

    pygame.init()

    WIDTH, HEIGHT = 1900, 1000

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Test")

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)

    rect_width, rect_height = 50, 50

    old_x, old_y = 0, 0

    edgesx = [-13,11]
    edgesy  = [8,-10]
    offset_x = 1
    offset_y = 4
    drawing = False

    running = True
    while cap.isOpened() and running == True:
        moved = False
        success, image = cap.read()

        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_a:
                    drawing = True
                if event.key == pygame.K_c:
                    rects = []
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    drawing = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                
                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                message = ""
                if y < -10:
                    text = "Looking left"
                    if(message_queue.empty()):
                        message += "left"
                elif y > 10:
                    text = "Looking Right"
                    if(message_queue.empty()):
                        message += "right"
                if x < -5:
                    text = "Looking Down"
                    if(message_queue.empty()):
                        message += "down"    
                elif x > 16:
                    text = "Looking Up"
                    if(message_queue.empty()):
                        message += "up"
                if message == "":
                    text = "Forward"
                    if(message_queue.empty()):
                        message = "Forward"
                if(message_queue.empty()):
                    message_queue.put(message)
                
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255,0,0), 3)

                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
            
            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime

            offset_x = 1
            offset_y = 4
            x = x + offset_y
            y = y + offset_x
            distancex = abs(x - old_x)
            distancey = abs(y - old_y)

            if distancex <= noise_tunning:
                rect_y = 1000/2 + (1000/18 * np.round(-old_x, 2))
            else:
                rect_y = 1000/2 + (1000/18 * np.round(-x, 2))
                movedx = True
                temp1 = old_x
                old_x = x

            if distancey <= noise_tunning:
                rect_x = 1900/2 + (1900/24 * np.round(old_y, 2))
            else:
                rect_x = 1900/2 + (1900/24 * np.round(y, 2))
                movedy = True
                temp2 = old_y
                old_y = y

            if rect_x <= 0:
                rect_x = 0
            elif rect_x + rect_width >= WIDTH:
                rect_x = WIDTH - rect_width
            if rect_y <= 0:
                rect_y = 0
            elif rect_y + rect_height >= HEIGHT:
                rect_y = HEIGHT - rect_height

            if (movedy or movedx) and drawing == True:
                if movedx and movedy:
                    posx = 1900/2 + (1900/24 * np.round(y, 2))
                    posy = 1000/2 + (1000/18 * np.round(-x, 2))
                    rects.append(pygame.Rect(posx, posy, abs(posx - (1900/2 + (1900/24 * np.round(temp2, 2)))) + rect_width, abs(posy - (1000/2 + (1000/18 * np.round(-temp1, 2)))) + rect_height))
                elif movedx:
                    posx = 1900/2 + (1900/24 * np.round(y, 2))
                    rects.append(pygame.Rect(posx, posy, abs(posx - (1900/2 + (1900/24 * np.round(temp1, 2)))) + rect_width, posy + rect_height))
                elif movedy:
                    posy = 1000/2 + (1000/18 * np.round(-x, 2))
                    rects.append(pygame.Rect(posx, posy, posx + rect_width, abs(posy - (1000/2 + (1000/18 * np.round(-temp1, 2)))) + rect_height))

            cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            pygame.draw.rect(screen, RED, (rect_x, rect_y, rect_width, rect_height))
            draw_rects(screen, rects)
            pygame.display.flip()

        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    pygame.quit()

    # Sending a message via the queue to signal the Arduino connection
    message_queue.put("Face Tracking Stopped")

    return running

def connect_arduino(message_queue):
    ports = serial.tools.list_ports.comports()
    serialInst = serial.Serial()
    portsList = []

    for one in ports:
        portsList.append(str(one))
        print(str(one))

    com = input("Select Com Port for Arduino #: ")

    for i in range(len(portsList)):
        if portsList[i].startswith("COM" + str(com)):
            use = "COM" + str(com)
            print(use)

    serialInst.baudrate = 9600
    serialInst.port = use
    serialInst.open()
    print("Connected to arduino")
    lastCommand = ""
    while True:
        if not message_queue.empty():
            message = message_queue.get()
            if message == "Forward" and message != lastCommand:
                command = "F"
                print("sent " + command)
                serialInst.write(command.encode('utf-8'))
                lastCommand = message
            elif message == "up" and message != lastCommand:
                command = "up"
                print("sent " + command)
                serialInst.write(command.encode('utf-8'))
                lastCommand = message
            elif message == "leftup" and message != lastCommand:
                command = "upLeft"
                print("sent " + command)
                serialInst.write(command.encode('utf-8'))
                lastCommand = message
            elif message == "rightup" and message != lastCommand:
                command = "upRight"
                print("sent " + command)
                serialInst.write(command.encode('utf-8'))
                lastCommand = message
            elif message == "left" and message != lastCommand:
                command = "left"
                print("sent " + command)
                serialInst.write(command.encode('utf-8'))
                lastCommand = message
            elif message == "right" and message != lastCommand:
                command = "right"
                print("sent " + command)
                serialInst.write(command.encode('utf-8'))
                lastCommand = message
            elif message == "down" and message != lastCommand:
                command = "down"
                print("sent " + command)
                serialInst.write(command.encode('utf-8'))
                lastCommand = message
            elif message == "leftdown" and message != lastCommand:
                command = "downLeft"
                print("sent " + command)
                serialInst.write(command.encode('utf-8'))
                lastCommand = message
            elif message == "rightdown" and message != lastCommand:
                command = "downRight"
                print("sent " + command)
                serialInst.write(command.encode('utf-8'))
                lastCommand = message
            if message == "Face Tracking Stopped":
                print("Shutting down Arduino connection.")
                serialInst.close()
                break

# Create a queue for message passing between threads
message_queue = queue.Queue()

# Create and start threads
t1 = threading.Thread(target=connect_arduino, args=(message_queue,))
t2 = threading.Thread(target=run_face_tracking, args=(cap, mp_face_mesh, face_mesh, drawing_spec, message_queue))

print(cap.isOpened())

t1.start()
t2.start()

