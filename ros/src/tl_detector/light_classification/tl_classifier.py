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
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            xmin, xmax = xmin*800, xmax*800
            ymin, ymax = ymin*600, ymax*600
            rospy.loginfo('[CSChen] boxes (y,x) = from ({},{}) to ({},{})'.format(ymin,xmin,ymax,xmax))

            # TODO: Do color analyzer ...

        
        return TrafficLight.UNKNOWN
