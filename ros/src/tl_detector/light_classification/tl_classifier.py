import rospy
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import helper
from os.path import join
import numpy as np
import cv2, time

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        model_dir = '/home/student/finalProject/Udacity-term3-final'
        vgg_path = join(model_dir,'vgg')
        tl_path = join(model_dir,'tl_model/semantic_model_epoch_10')
        self._num_classes = 2
        self._min_num_traffic_pixel_ratio = 0.005
        
        # vgg_path = './vgg'
        self._sess = tf.Session()
        self._input_image, self._keep_prob, l3out, l4out, l7out = helper.load_vgg(self._sess,vgg_path)
        global_encoder = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        rospy.loginfo('[CSChen] len(global_encoder)={}'.format(len(global_encoder)))
        with tf.variable_scope('decoder'):
            layer_output = helper.layers(l3out, l4out, l7out, self._num_classes)
        rospy.loginfo("[CSChen] layer_output.shape={}".format(layer_output.shape))
        
        vars_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='decoder')
        rospy.loginfo('[CSChen] len(vars_decoder)={}'.format(len(vars_decoder)))
        saver = tf.train.Saver(var_list = vars_decoder)
        saver.restore(self._sess, tl_path)
        rospy.loginfo("[CSChen] Model loaded")

        logits = tf.reshape(layer_output,(-1,self._num_classes))
        self._out_softmax = tf.nn.softmax(logits) # batch_idx x all_pixel x num_classes
        rospy.loginfo("[CSChen] self._out_softmax.shape={}".format(self._out_softmax.shape))

        # [Debug]:
        self._debug_counter = 0


    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        self._debug_counter += 1
        # Should get the image data and it's shape first
        image_h_original, image_w_original, c_num = image.shape  # for simulator, 600, 800, 3
        rospy.loginfo('[CSChen] Before resizing image.shape = {}'.format(image.shape))
        
        # image = np.concatenate([image, np.zeros(shape=(8,800,3))], axis=0)

        image_resize = cv2.resize(image,(412,316))
        
        image_h, image_w, c_num = image_resize.shape  # for simulator, 608, 800, 3
        rospy.loginfo('[CSChen] After resizing image.shape = {}'.format(image_resize.shape))

        # [Debug]
        # traffic_pixels = [image[100,100,:], image[150,150,:]]
        # traffic_pixels = np.array([traffic_pixels], dtype=np.uint8)
        # rospy.loginfo('[CSChen] traffic_pixels.shape={}; traffic_pixels={}'.format(traffic_pixels.shape, traffic_pixels))
        # hsv = cv2.cvtColor(traffic_pixels,cv2.COLOR_RGB2HSV)
        # rospy.loginfo('[CSChen] hsv={}'.format(hsv))

        # rospy.loginfo('[CSChen] image.shape={}'.format(image.shape))
        stime = time.time()
        out_softmax = self._sess.run(self._out_softmax,{self._keep_prob: 1.0, self._input_image: [image_resize]})
        etime = time.time()
        rospy.loginfo('[CSChen] After TensorFlow with {}! out_softmax.shape={}'.format(etime-stime,out_softmax.shape))
        
        im_softmax = out_softmax[:, 1].reshape(320, 416) # image_h, image_w
        hratio = float(image_h_original)/320.0
        wratio = float(image_w_original)/416.0
        # im_softmax = out_softmax[:, 1].reshape(image_h, image_w) # image_h, image_w
        ypixels, xpixels = np.nonzero(im_softmax>0.5)

        debug_image = np.copy(image)

        traffic_pixels = []
        for yidx,xidx in zip(ypixels,xpixels):
            yidx_original = int(yidx*hratio)
            xidx_original = int(xidx*wratio)
            traffic_pixels.append(image[yidx_original,xidx_original,:])
            debug_image[yidx_original,xidx_original,:] = [0,0,255]
            # traffic_pixels.append(image[yidx,xidx,:])

        # [Debug]: save debug image
        cv2.imwrite("/home/student/finalProject/pics_vgg/camera_img_"+str(self._debug_counter)+".jpg",debug_image)
        rospy.loginfo("[CSChen] dumped image camera_img_"+str(self._debug_counter)+".jpg")

        min_num_traffic_pixel = image_h*image_w*self._min_num_traffic_pixel_ratio
        if (len(traffic_pixels)<min_num_traffic_pixel):
            rospy.loginfo('[CSChen] UNKNOWN (no traffic light detected)')
            return TrafficLight.UNKNOWN
        traffic_pixels = np.array([traffic_pixels], dtype=np.uint8)
        rospy.loginfo('[CSChen] traffic_pixels.shape={}'.format(traffic_pixels.shape))
        hsv = cv2.cvtColor(traffic_pixels,cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0,:,0], hsv[0,:,1], hsv[0,:,2]
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
