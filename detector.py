import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils

class DrowsinessDetector:
    def __init__(self, thresh=0.25, frame_check=20, model_path="models/shape_predictor_68_face_landmarks.dat"):
        self.thresh = thresh
        self.frame_check = frame_check
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.flag = 0

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        """
        Takes a BGR frame, returns (output_frame, alert_flag)
        alert_flag == True means visual alert should be shown.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        alert = False

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape_np = face_utils.shape_to_np(shape)
            leftEye = shape_np[self.lStart:self.lEnd]
            rightEye = shape_np[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # drawing eye hulls
            leftHull = cv2.convexHull(leftEye)
            rightHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftHull], -1, (0,255,0), 1)
            cv2.drawContours(frame, [rightHull], -1, (0,255,0), 1)

            if ear < self.thresh:
                self.flag += 1
                if self.flag >= self.frame_check:
                    alert = True
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(frame, "************VIDEO STOPS!************", (10,750),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                # reset when eyes reopen
                self.flag = 0

        return frame, alert
