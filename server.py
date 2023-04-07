import asyncio
import copy
import cv2
import itertools
import mediapipe as mp

from libs.emotion_recognition import KeyPointClassifier
from websockets.server import serve


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


category = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    static_image_mode=True)

keypoint_classifier = KeyPointClassifier()


def get_expression(image):
    image = cv2.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, face_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)
            facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
            return category[facial_emotion_id]
    else:
        return category[2]


async def echo(websocket):
    async for message in websocket:
        if isinstance(message, bytes):
            with open('image.png', 'wb') as f:
                f.write(message)
            img = cv2.imread('image.png')
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            expression = get_expression(img)
            # print(expression)
            result = '{"event": "@ExpressionResult", "result": "' + expression + '"}'
            await websocket.send(result)
        else:
            result = '{"event": "@ExpressionResult"}'
            await websocket.send(result)


async def main():
    async with serve(echo, "localhost", 8765):
        await asyncio.Future()  # run forever


asyncio.run(main())
