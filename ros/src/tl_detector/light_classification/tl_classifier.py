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


    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        # Should get the image data and it's shape first
        image_h, image_w, c_num = image.shape  # for simulator, 600, 800, 3
        rospy.loginfo('[CSChen] Before resizing image.shape = {}'.format(image.shape))
        
        image = np.concatenate([image, np.zeros(shape=(8,800,3))], axis=0)

        # image = cv2.resize(image,(412,316))
        
        image_h, image_w, c_num = image.shape  # for simulator, 600, 800, 3
        rospy.loginfo('[CSChen] After resizing image.shape = {}'.format(image.shape))

        # rospy.loginfo('[CSChen] image.shape={}'.format(image.shape))
        stime = time.time()
        out_softmax = self._sess.run(self._out_softmax,{self._keep_prob: 1.0, self._input_image: [image]})
        etime = time.time()
        rospy.loginfo('[CSChen] After TensorFlow with {}! out_softmax.shape={}'.format(etime-stime,out_softmax.shape))
        
        im_softmax = out_softmax[:, 1].reshape(image_h, image_w) # image_h, image_w
        ypixels, xpixels = np.nonzero(im_softmax>0.5)
        rvalues = 0.0
        gvalues = 0.0
        for yidx,xidx in zip(ypixels,xpixels):
            rvalues += image[yidx,xidx,0]
            gvalues += image[yidx,xidx,1]

        if rvalues>gvalues:
            rospy.loginfo('[CSChen] RED')
        else:
            rospy.loginfo('[CSChen] GREEN')
        # im_softmax = out_softmax[:, 1].reshape(320, 416) # image_h, image_w

        # those points whoes prob>0.5 are our target pixels

        # segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        return TrafficLight.UNKNOWN
