import cv2
import dlib
from utils import get_eye_landmarks, calculate_ear, Plotter, draw_eye_landmarks 

def main():
    capture = cv2.VideoCapture(0)
    #------------------------------------------------------------------------------
    # 1. Initializations.
    #------------------------------------------------------------------------------

    # Initialize counter for the number of blinks detected.
    BLINK = 0
    frame_count = 0
    ear_sum = 0
    frame_caliberation = 30
    upper = 0.9
    lower = 0.7
    ear_calibrated = 0
    plot = Plotter()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    cap = cv2.VideoCapture(0)
    state_prev = state_curr = 'open'
    #------------------------------------------------------------------------------

    while True:
        ret, frame = capture.read()
        frame_count += 1
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(frame, face)
            landmarks = landmarks.parts()
            eye_landmarks = get_eye_landmarks(landmarks)
            ear = calculate_ear(eye_landmarks)
            draw_eye_landmarks(frame, eye_landmarks)
            plot.update(frame_count, ear)

            if frame_count < frame_caliberation:
                ear_sum += ear
            elif frame_count == frame_caliberation:
                ear_calibrated = ear_sum / frame_caliberation
            else:
                if ear < ear_calibrated * lower:
                    state_curr = 'close'
                if ear > ear_calibrated * upper:
                    state_curr = 'open'
                if state_curr == 'open' and state_prev == 'close':
                    BLINK += 1
                state_prev = state_curr

        cv2.putText(frame, f"Blinks: {BLINK}", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()