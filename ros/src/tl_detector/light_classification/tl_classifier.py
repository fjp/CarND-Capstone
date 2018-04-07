import rospy
from styx_msgs.msg import TrafficLight
import tensorflow as tf
from os.path import join
import numpy as np
import cv2, time

class TLClassifier(object):
    def __init__(self):
        # Loading TF Model
        model_path = '/home/student/finalProject/Udacity-term3-final/share2team/frozen_sim/frozen_inference_graph.pb'
        detection_graph = tf.Graph()

        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:        
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        rospy.loginfo("[CSChen] loaded model.pb")

        self._image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self._detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self._detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self._detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self._num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        self._sess = tf.Session(graph=detection_graph)
        rospy.loginfo("[CSChen] Get tensor variables and create session")


    def color_analyzer(self,image):
        rospy.loginfo('[CSChen] np.max(image)={}'.format(np.max(image)))
        # I think image in ROS is stored in BGR not RGB !!
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        h, s, v = h.reshape(-1), s.reshape(-1), v.reshape(-1)
        rospy.loginfo('[CSChen] np.max(h)={}'.format(np.max(h)))
        redh = len([x for x in h if (x<25)])
        yellowh = len([x for x in h if (25<x<45)])
        greenh = len([x for x in h if (45<x<65)])
        light_hlist = [redh, yellowh, greenh]
        rospy.loginfo('[CSChen] light_hlist={}'.format(light_hlist))
        if np.argmax(light_hlist)==0:
            rospy.loginfo('[CSChen] RED')
            return TrafficLight.RED
        if np.argmax(light_hlist)==1:
            rospy.loginfo('[CSChen] YELLOW')
            return TrafficLight.YELLOW
        rospy.loginfo('[CSChen] GREEN')
        return TrafficLight.GREEN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # Should get the image data and it's shape first
        image_h_original, image_w_original, c_num = image.shape  # for simulator, 600, 800, 3
        rospy.loginfo('[CSChen] image.shape = {}'.format(image.shape))
        
        image_expanded = np.expand_dims(image, axis=0)
        rospy.loginfo('[CSChen] image_expanded.shape = {}'.format(image_expanded.shape))

        stime = time.time()
        # Actual detection.
        (boxes, scores, classes, num) = self._sess.run(
          [self._detection_boxes, self._detection_scores, self._detection_classes, self._num_detections],
          feed_dict={self._image_tensor: image_expanded})
        etime = time.time()
        boxes = np.squeeze(boxes)
        rospy.loginfo('[CSChen] After TensorFlow with {}! len(boxes)={}'.format(etime-stime,len(boxes)))
        # TrafficLight.[RED, YELLOW, GREEN, UNKNOWN] represents 0, 1, 2, 3
        recog_color_list = [0, 0, 0, 0]
        # We pick up first three boxes which are more confident
        for i in range(3):
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            xmin, xmax = int(xmin*800), int(xmax*800)
            ymin, ymax = int(ymin*600), int(ymax*600)
            rospy.loginfo('[CSChen] boxes (y,x) = from ({},{}) to ({},{})'.format(ymin,xmin,ymax,xmax))

            # TODO: Do color analyzer ...
            recog_color = self.color_analyzer(image[ymin:ymax,xmin:xmax,:])
            recog_color_list[recog_color] += 1

        final_predicted_light = np.argmax(recog_color_list)
        rospy.loginfo('[CSChen] final predicted light state = {}'.format(final_predicted_light))
        return final_predicted_light
        # return TrafficLight.UNKNOWN
