import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class Video(object):
    video = None
    number_frame = 0
    key_points_array = []
    des_array = []
    matching_rate_array = []
    avg_rate_array = []
    array_shots = []
    array_boundary_position = []
    minimun_frames_in_shot = 10
    key_frame_array = []

    def __init__(self, path_to_video):
        self.video = cv2.VideoCapture(path_to_video)
        self.read_frames_and_sift()

    def segment_to_frames(self):
        print 'Start segment to frames'
        success = True
        while success:
            success,image = self.video.read()
            if success:
                print 'Read a new frame: ', success
                cv2.imwrite('./test_frame/frame%d.jpg' % self.number_frame, image)
                self.number_frame += 1

    def read_frames_and_sift(self):
        print 'Start read and sift'
        if self.number_frame == 0:
            self.segment_to_frames()

        sift = cv2.xfeatures2d.SIFT_create()
        for i in range(0, self.number_frame):
            img = cv2.imread('./test_frame/frame{}.jpg'.format(i))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray,None)
            self.key_points_array.append(kp)
            self.des_array.append(des)

    def compute_matching_rate(self, des_1, des_2):
        bf = cv2.BFMatcher()
        matching_rate = 0

        try:
            len_des_1 = len(des_1)
        except:
            len_des_1 = 0

        try:
            len_des_2 = len(des_2)
        except:
            len_des_2 = 0


        if len_des_1 > 1 and len_des_2 > 1:
            matches = bf.knnMatch(des_1, des_2, k = 2)
            number_matching_kp = 0
            for a, b in matches:
                if a.distance < 0.75 * b.distance:
                    number_matching_kp += 1
            matching_rate = 2.0 * number_matching_kp / (len(des_1) + len(des_2))
        else:
            if abs( len_des_1 - len_des_2 ) <= 10:
                matching_rate = 1.0
            else:
                matching_rate = 0.0

        return float(matching_rate)

    def compute_matching_rate_array(self):
        print 'Start compute matching rate array'
        self.matching_rate_array = []
        self.matching_rate_array.append(0)
        for i in range(1, self.number_frame):
            matching_rate = self.compute_matching_rate(self.des_array[i-1], self.des_array[i])
            self.matching_rate_array.append(matching_rate)

    def comptute_avg_rate_array(self, number_frame_before):
        print 'Start compute averange rate array'
        self.avg_rate_array = []
        if len(self.matching_rate_array) == 0:
            self.compute_matching_rate_array()
            self.matching_rate_array[0] = self.matching_rate_array[1]

        self.avg_rate_array.append(0)

        for i in range(1, self.number_frame):
            if i < number_frame_before:
                avg_rate = np.mean(self.matching_rate_array[0:i])
            else:
                start_position = i - number_frame_before
                avg_rate = np.mean(self.matching_rate_array[start_position:i])
            self.avg_rate_array.append(avg_rate)

    def detect_shot_boundary_1(self, threshold_top, threshold_bottom, sum_threshold, number_frame_after):
        self.comptute_avg_rate_array(1)

        subject_rate_array = []
        boundary = []
        boundary.append(0)
        self.array_boundary_position.append(0)
        for i in range(0, self.number_frame):
            subject_rate = self.avg_rate_array[i] - self.matching_rate_array[i]
            subject_rate_array.append(subject_rate)


        for i in range(0, self.number_frame):
            subject_rate = subject_rate_array[i]
            if (subject_rate >= threshold_top):
                self.array_boundary_position.append(i)
                print 'Cut boundary at', i

            elif subject_rate >= threshold_bottom:
                if i < self.number_frame - number_frame_after:
                    end_position = i + number_frame_after
                else:
                    end_position = self.number_frame
                sum_rate = subject_rate_array[i:end_position]
                if (sum_rate >= sum_threshold):
                    self.array_boundary_position.append(i)
                    print 'Gradual boundary at', i
        self.array_boundary_position.append(self.number_frame - 1)
        self.detect_shot()
        self.write_shot_and_key_frame()

        return

    def detect_shot_boundary_2(self, threshold_top, threshold_bottom, number_frame_after):
        self.compute_matching_rate_array()
        self.matching_rate_array[0] = 1.0
        i = 1
        self.array_boundary_position.append(0)
        while i < self.number_frame:
            if self.matching_rate_array[i] <= threshold_bottom:
                print 'Cut boundary at', i-1,' and ', i
                self.array_boundary_position.append(i)
            elif self.matching_rate_array[i] <= threshold_top:
                first_gradual_at = i
                while i < self.number_frame - 1:
                    if self.matching_rate_array[i] <= threshold_top:
                        i += 1
                    else:
                        break

                if self.compute_matching_rate(self.des_array[first_gradual_at], self.des_array[i]) < threshold_bottom:
                    if (i - first_gradual_at + 1) <= number_frame_after:
                        print 'Cut boundary at', first_gradual_at,' and ', i
                        self.array_boundary_position.append(i)
                    else:
                        print 'Gradual from ', first_gradual_at, ' to ', i
                        self.array_boundary_position.append(first_gradual_at)
                else:
                    i = first_gradual_at

            i += 1
        self.array_boundary_position.append(self.number_frame - 1)
        self.detect_shot()
        self.write_shot_and_key_frame()

        return

    def detect_shot(self):
        for i in range(0, len(self.array_boundary_position) - 1):
            start_shot_position = self.array_boundary_position[i]
            end_shot_position = self.array_boundary_position[i + 1] - 1
            number_frame = end_shot_position - start_shot_position + 1
            if number_frame < self.minimun_frames_in_shot and i > 0:
                last_shot_info = self.array_shots[-1]
                last_shot_info['end_frame_position'] = end_shot_position
            else:
                self.array_shots.append({
                    'start_frame_position': start_shot_position,
                    'end_frame_position': end_shot_position
                })
        fisrt_shot_number_frame = self.array_shots[0]['end_frame_position'] - self.array_shots[0]['start_frame_position'] + 1
        if fisrt_shot_number_frame < self.minimun_frames_in_shot:
            self.array_shots[0]['end_frame_position'] = self.array_shots[1]['end_frame_position']
            del self.array_shots[1]

    def detect_key_frame(self, start_frame_position, end_frame_position):
        max_kp = len(self.key_points_array[start_frame_position])
        key_frame_position = start_frame_position
        for i in range(start_frame_position, end_frame_position + 1):
            number_kp = self.key_points_array[i]
            if number_kp > max_kp:
                max_kp = number_kp
                key_frame_position = i
        return key_frame_position

    def write_shot_and_key_frame(self):
        if len(self.array_shots) == 0:
            self.detect_shot
        for i in range(0, len(self.array_shots)):
            self.write_shot(self.array_shots[i], i)
            key_frame_position = self.detect_key_frame(self.array_shots[i]['start_frame_position'],
                self.array_shots[i]['end_frame_position'])
            self.key_frame_array.append(key_frame_position)
            img = cv2.imread('./test_frame/frame{}.jpg'.format(key_frame_position))
            cv2.imwrite('./key_frame/shot%d.jpg'% i, img)

    def write_shot(self, shot_info, shot_number):
        first_frame = cv2.imread('./test_frame/frame{}.jpg'.format(shot_info['start_frame_position']))
        height , width , layers =  first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter('./shot/shot%d.avi' % shot_number,fourcc, 20.0, (width,height))
        for i in range(shot_info['start_frame_position'], shot_info['end_frame_position'] + 1):
            img = cv2.imread('./test_frame/frame{}.jpg'.format(i))
            video.write(img)

if __name__ == '__main__':
    video = Video('./data/7Up Vintage that tu nhien that sang khoai 15s.mp4')
    # video.detect_shot_boundary_1(0.5, 0.2, 1.5, 5)
    video.detect_shot_boundary_2(0.5, 0.2, 3)
    plt.plot(range(0,video.number_frame),video.matching_rate_array)
    plt.ylabel('')
    plt.show()
