from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import PoseModule as pm
import os

app = Flask(__name__)

@app.route('/curls', methods=['POST'])
def curl():
    video_base64 = request.json['video']
    video_bytes = base64.b64decode(video_base64)

    # Save the video to a temporary file
    temp_video_path = 'temp_video.mp4'
    with open(temp_video_path, 'wb') as file:
        file.write(video_bytes)

    output_video_name = os.path.splitext(os.path.basename(temp_video_path))[0] + "_processed.mp4"
    output_video_path = os.path.join('video', output_video_name)

    cap = cv2.VideoCapture(temp_video_path)
    detector = pm.poseDetector()
    count = 0
    dir = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (width, height))
    frame_number = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 12, 14, 16)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0

            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)
            # Draw Curl Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 25)

        out.write(img)
        frame_number += 1

    out.release()
    cap.release()

    # Read the processed video and encode it as base64
    with open(output_video_path, 'rb') as file:
        output_video_bytes = file.read()
        output_video_base64 = base64.b64encode(output_video_bytes).decode('utf-8')

    # Remove the temporary video file
    os.remove(temp_video_path)

    return jsonify({"output_video": output_video_base64})


if __name__ == '__main__':
    app.run()




@app.route('/squats', methods=['GET'])
def squats():
    video = request.args.get('video')
    output_video_name = os.path.splitext(os.path.basename(video))[0] + "_processed.mp4"
    output_video_path = os.path.join('video', output_video_name)

    cap = cv2.VideoCapture(video)
    detector = pm.poseDetector()
    count = 0
    dir = 0
    knee_error = 0
    knee_error_final = 0
    accuracy = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (width, height))
    frame_number = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 24, 26, 28)
            per = np.interp(angle, (185, 260), (0, 100))
            bar = np.interp(angle, (185, 260), (650, 100))
            if per == 100:
                if dir == 0:
                    count += 0.5
                    knee_error_final += knee_error
                    knee_error = 0
                    dir = 1
            if per == 0:
                if dir == 1:
                    count += 0.5
                    dir = 0

            # Check if knee extends beyond the tips of the feet
            uncertainty_margin = 30
            knee_beyond_foot = lmList[25][1] > (lmList[31][1] + uncertainty_margin)

            # Draw lines from hip to knee, and from knee to ankle
            hip_knee_color = (0, 255, 0)  # Green
            knee_ankle_color = (0, 255, 0) if not knee_beyond_foot else (0, 0, 255)  # Green if not beyond, red if beyond
            img = detector.draw_lines(img, [23, 25, 27], [hip_knee_color, knee_ankle_color])
            img = detector.draw_lines(img, [24, 26, 28], [hip_knee_color, knee_ankle_color])

            if knee_beyond_foot: 
                knee_error = 1

            # # Draw Bar
            # cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            # cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            # cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
            #             color, 4)
            
            # Draw Squat Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 25)
        
        if count != 0:
            percentPerCount = 100 / count;
        else:
            percentPerCount = 0
        accuracy = percentPerCount  * (count - knee_error_final)

        out.write(img)
        frame_number += 1

    out.release()
    cap.release()

    return jsonify({
        "output_video": os.path.abspath(output_video_path),
        "count": count,
        "accuracy": accuracy,
        "error": [{
            "field": "knee",
            "message": "Don't stick your knees out in front of the tips of your feet",
            "localizedMessage": "Не выставляйте колени перед кончиками стоп",
            "count": knee_error_final
        }],
    })



@app.route('/pushups', methods=['GET'])
def pushups():
    video = request.args.get('video')
    output_video_name = os.path.splitext(os.path.basename(video))[0] + "_processed.mp4"
    output_video_path = os.path.join('video', output_video_name)

    cap = cv2.VideoCapture(video)
    detector = pm.poseDetector()
    count = 0
    back_error_final = 0
    back_error = 0
    dir = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (width, height))
    frame_number = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Back angle from vertical (estimated using points from hip to shoulder)
            back_angle_1 = detector.findAngle(img, 11, 23, 27)
            back_angle_2 = detector.findAngle(img, 12, 24, 28)
            
            # Incorrect form if the back_angle deviates from 180 (vertical) by more than a threshold
            incorrect_form = (back_angle_1 + back_angle_2) / 2 - 180 > 10
            
            left_hand_angle = detector.findAngle(img, 11, 13, 15)
            right_hand_angle = detector.findAngle(img, 12, 14, 16)
            per = np.interp(left_hand_angle, (205, 270), (0, 100))
            bar = np.interp(left_hand_angle, (205, 270), (650, 100))
            hands_color = (0, 255, 0)
            back_color = (0, 255, 0)

            if incorrect_form:
                back_color = (0, 0, 255)
            if incorrect_form and per > 0:
                back_error = 1

            img = detector.draw_lines(img, [11, 23, 27], [back_color, back_color])
            img = detector.draw_lines(img, [12, 24, 28], [back_color, back_color])
            img = detector.draw_lines(img, [11, 13, 15], [hands_color, hands_color])
            img = detector.draw_lines(img, [12, 14, 16], [hands_color, hands_color])
            
            if per == 100:
                if dir == 0:
                    count += 1
                    back_error_final += back_error
                    back_error = 0
                    dir = 1
            if per == 0:
                if dir == 1:
                    dir = 0

            # Draw Squat Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 25)

        out.write(img)
        frame_number += 1

    if count != 0:
        percentPerCount = 100 / count;
    else:
        percentPerCount = 0
    accuracy = percentPerCount  * (count - back_error_final)

    out.release()
    cap.release()

    return jsonify({
        "output_video": os.path.abspath(output_video_path),
        "count": count,
        "accuracy": accuracy,
        "error": [{
            "field": "back",
            "message": "Keep your back straight",
            "localizedMessage": "Держите вашу спишу ровнее",
            "count": back_error_final
        }],
    })



@app.route('/abs_legs', methods=['GET'])
def abs_legs():
    video = request.args.get('video')
    output_video_name = os.path.splitext(os.path.basename(video))[0] + "_processed.mp4"
    output_video_path = os.path.join('video', output_video_name)

    cap = cv2.VideoCapture(video)
    detector = pm.poseDetector()
    count = 0
    dir = 1
    error = 0
    errorFinal = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (width, height))
    frame_number = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 11, 23, 27)  # Angle between hip, torso, and leg
            per = np.interp(angle, (90, 180), (0, 100))  # Change these values according to the expected leg raise angle

            leg_angle = detector.findAngle(img, 23, 25, 27)
            incorrect_form = leg_angle - 180 > 40

            # check if angle is 90
            if per == 100:
                if dir == 0:
                    count += 1
                    errorFinal += error
                    error = 0
                    dir = 1

            if per == 0:
                if dir == 1:
                    count += 0
                    dir = 0

            # Draw lines from hip to torso, and from torso to leg
            hip_torso_color = (0, 255, 0)  # Green
            img = detector.draw_lines(img, [11, 23], [hip_torso_color])

            leg_color = (0, 255, 0) 
            if incorrect_form:
                error = 1
                leg_color = (0, 0, 255)
            leg_color = (0, 255, 0) if not incorrect_form else (0, 0, 255) # Red for incorrect form
            img = detector.draw_lines(img, [23, 25, 27], [leg_color, leg_color])

            # Draw Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 25)

        out.write(img)
        frame_number += 1
    
    if count != 0:
        percentPerCount = 100 / count;
    else:
        percentPerCount = 0
    accuracy = percentPerCount  * (count - errorFinal)

    out.release()
    cap.release()

    return jsonify({
        "output_video": os.path.abspath(output_video_path),
        "count": count,
        "accuracy": accuracy,
        "error": [
            {
                "field": "legs",
                "message": "Keep your legs straight",
                "localizedMessage": "Держите ваши ноги ровнее",
                "count": errorFinal
            }
        ],
    })



@app.route('/lateral_raise', methods=['GET'])
def lateral_raise():
    video = request.args.get('video')
    output_video_name = os.path.splitext(os.path.basename(video))[0] + "_processed.mp4"
    output_video_path = os.path.join('video', output_video_name)

    cap = cv2.VideoCapture(video)
    detector = pm.poseDetector()
    count = 0
    dir = 1
    error = 0
    upper_error = 0

    perPrev = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (width, height))

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            angle = detector.findAngle(img, 16, 12, 24)  # Angle between shoulder, elbow and wrist
            per = np.interp(angle, (260, 340), (0, 100))  # Change these values according to the expected arm raise angle

            if per == 100:
                if dir == 0:
                    count += 1
                    dir = 1
                    upper_error += error
                    error = 0
                    perPrev = 0

            if per == 0:
                if dir == 1:
                    count += 0
                    dir = 0

            arm_color = (0, 255, 0)  # Green
            if per > 10 and per > perPrev and dir == 1:
                error = 1
                arm_color = (0, 0, 255)
            img = detector.draw_lines(img, [16, 12, 24], [arm_color, arm_color])

            # Draw Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 25)
            
        perPrev = per

        if count != 0:
            percentPerCount = 100 / (count + upper_error);
        else:
            percentPerCount = 0
        accuracy = percentPerCount  * (count)

        out.write(img)

    out.release()
    cap.release()

    return jsonify({
        "output_video": os.path.abspath(output_video_path),
        "count": count,
        "accuracy": accuracy,
        "error": [
            {
                "field": "shouder",
                "message": "Raise your hands higher",
                "localizedMessage": "Поднимайте руки выше",
                "count": upper_error
            }
        ],
    })



if __name__ == '__main__':
    app.run()
