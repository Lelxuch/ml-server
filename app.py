from flask import Flask, request, jsonify
import cv2
import numpy as np
import PoseModule as pm
import os

app = Flask(__name__)

@app.route('/curl', methods=['GET'])
def curl():
    video = request.args.get('video')
    output_video_name = os.path.splitext(os.path.basename(video))[0] + "_processed.mp4"
    output_video_path = os.path.join('video', output_video_name)
    skip_frames = 2

    cap = cv2.VideoCapture(video)
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

        if frame_number % skip_frames == 0:
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

    return jsonify({"output_video": os.path.abspath(output_video_path)})


if __name__ == '__main__':
    app.run()
