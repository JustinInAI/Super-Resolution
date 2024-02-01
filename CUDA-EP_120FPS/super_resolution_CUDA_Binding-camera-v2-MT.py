# This source code is for downloading pytorch super resoultion model directly and coverting to onnx runtime
# Before doing following, please install onnxruntime-gpu and torch:
# pip install onnxruntime-gpu
# pip install torch
# Some standard imports
import io
import numpy as np

from torch import nn # include torch python packages
import torch.utils.model_zoo as model_zoo
import torch.onnx # include torch onnx runtime


# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init

# For onnxruntime
import onnx
import onnxruntime as ort
from win32api import GetSystemMetrics
import win32con,win32gui,win32print
#from PySide6 import QtGui
#screen=QtGui.QGuiApplication.primaryScreen()
#scaled_pixel_ratio = screen.scaled_pixel_ratio()
#print("SP:", scaled_pixel_ratio)
'''
hDC = win32gui.GetDC(0)
HOREZON=win32print.GetDeviceCaps(hDC,win32con.DESKTOPHORZRES)
VERTICAL=win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
print("H:",HOREZON)
print("Y:",VERTICAL)
'''
#model_path="C:\camerasrmodel\colorspace-bgr2ycrcb2bgr-SR-old.onnx"
model_path="C:\camerasrmodel\ycrcb-bgr-sr-2x-only1slice-FP16.onnx"
# Binding ONNX Runtime CUDA EP
#providers= [("CUDAExecutionProvider", {'device_id':0,'arena_extend_strategy': 'kNextPowerOfTwo','do_copy_in_default_stream':False})]

#providers = ['CUDAExecutionProvider','CPUExecutionProvider']
#providers=['DmlExecutionProvider']
#providers=['OpenVINOExecutionProvider']
#providers=['CPUExecutionProvider']
providers = ['CUDAExecutionProvider']
sess_options = ort.SessionOptions()
#sess_options.enable_profiling = True
#sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#sess_options.optimized_model_filepath = model_path
#sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
#print("input ", sess.get_inputs()[2].name)

#sess_cb = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
#sess_cr = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

#providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]

# Set graph optimization level
#sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#sess_options.optimized_model_filepath = model_path
#sess_options.enable_profiling = True
sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)



'''
# Binding ONNX Runtime DirectML EP
providers = ['DmlExecutionProvider']  # DirectML EP Providers source code
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("super_resolution.onnx", sess_options=sess_options, providers=providers)
'''

# Computing with CUDA EP or DirectML EP
'''
def to_numpy(tensor): ### how to use cuda???

    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
'''
# compute ONNX Runtime output prediction
#ort_inputs = {sess.get_inputs()[0].name: to_numpy(x)}
#ort_outs = sess.run(None, ort_inputs)


# compare ONNX Runtime and PyTorch results
#np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

#print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# Onnx model loading, binding CUDA EP and comuting with CUDA EP End






import cv2
#for one jpeg file validation
from PIL import Image
import os
import torchvision.transforms as transforms
import time
from time import perf_counter

import sys



#img = img.astype(np.uint8)
#resize = transforms.Resize([480, 640])
#img = resize(img)

#to_tensor = transforms.ToTensor()
#img = to_tensor(img)
#img.unsqueeze_(0)
#print("img shape: ", img.shape)

#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
#cv2.namedWindow("super resolution-640x480-1280x960",cv2.WINDOW_AUTOSIZE)
#cv2.resizeWindow("super resolution-640x480-1280x960", 640,480)
#cap.set(cv2.CAP_PROP_HW_DEVICE,0)
#cap.set(cv2.CAP_PROP_HW_ACCELERATION,cv2.VIDEO_ACCELERATION_D3D11)
t0 = time.perf_counter()

from threading import Thread

#cap.grab()
#ret, frame = cap.retrieve()
from ctypes import windll,wintypes
windll.user32.SetThreadDpiAwarenessContext(wintypes.HANDLE(-1))

Screen_Name="640x480-->1280x960"
class Frame_Get:
    def __init__(self,src):

        #self.cap.grab()
        #self.ret, self.frame = self.cap.retrieve()
        self.cap = cv2.VideoCapture(src, cv2.CAP_MSMF, params=[cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_D3D11])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.cap.set(cv2.CAP_PROP_FPS,120)
        
        #cv2.namedWindow(Screen_Name,cv2.WINDOW_NORMAL)
        #self.start()
       
        self.ret,self.frame = self.cap.read()
        #self.stopped = True 

        self.mt=Thread(target=self.update,args=())
        self.mt.daemon=True
    def start(self):
        #print("thread start")
        #self.stopped = False
        self.mt.start()
        #return self
    def update(self):
        #print("frame update start!!!!!")
        while True:
           #if self.stopped is True :
            #    break
           self.ret,self.frame = self.cap.read()
           #if self.ret is False:
            #   self.stopped = True
             #  break 
        self.cap.release()      

       
    def read(self):
        #print("get new frame")
        return self.frame
        
    def stop(self):
        self.stopped = True 




#from multiprocessing.pool import ThreadPool
#pool = ThreadPool(processes=5)
#import concurrent.futures
#with concurrent.futures.ThreadPoolExecutor(max_workers=1) as mt:
        #nf=mt.submit(SR_Presentation)
FG = Frame_Get(0)
FG.start()
#mt = threading(target=Frame_Get.SR_Presentation,args=())
fps = 0
f=0
start_time=time.time()
while (True):   
    new_frame=FG.read()
    #cv2.imshow("test",new_frame)

    #nf=mt.submit(SR_Presentation)


    #nf = pool.apply_async(SR_Presentation)
    
    #X_ortvalue = ort.OrtValue.ortvalue_from_numpy(frame,'cuda', 0)
    io_binding = sess.io_binding()
    #io_binding.bind_input(name=sess.get_inputs()[0].name, device_type=X_ortvalue.device_name(), device_id=0, element_type=np.uint8, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
    #t4 = time.perf_counter()
 
    io_binding.bind_cpu_input("input",new_frame)
    io_binding.bind_output("output")
    
    #print("Latency for conver to numpy:", (t4-t3))
    #Y_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(sess.get_outputs()[0].shape, np.uint8, 'cuda', 0) 
    

    #io_binding.bind_input(name=sess.get_inputs()[0].name, device_type=X_ortvalue.device_name(), device_id=0, element_type=np.uint8, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
    #io_binding.bind_cpu_input(sess.get_inputs()[0].name,frame)
    #io_binding.bind_output(name=sess.get_outputs()[0].name, device_type=Y_ortvalue.device_name(), device_id=0, element_type=np.uint8, shape=Y_ortvalue.shape(), buffer_ptr=Y_ortvalue.data_ptr())
    #io_binding.bind_output(sess.get_outputs()[0].name)
    t4 = time.perf_counter()
    sess.run_with_iobinding(io_binding)
    t5 = time.perf_counter()
    #if(t5-t4)>=0.004:
    #  l=l+1
    #print("Latency for inference: ", (t5-t4))
    #t4 = time.perf_counter()
    result = io_binding.copy_outputs_to_cpu()[0]
    
    #t5 = time.perf_counter()
    #print("Latency for Copy to CPU: ", (t5-t4))
    
    #cv2.imwrite("result-1.jpg",result)
    #print("result shape: ",result.shape[:2] )
    #cv2.resize(result,result.shape,result)
    
    #result1=cv2.resize(result,(1280,960))
    #cv2.namedWindow("super resolution-640x480-1280x960", cv2.WINDOW_KEEPRATIO)
    #cv2.resizeWindow("super resolution-640x480-1280x960", 1920,1080)
    #t4 = time.perf_counter()
    #cv2.resizeWindow(Screen_Name,1280,960)
    cv2.imshow(Screen_Name,result)
    fps = fps + 1
    f=f+1
    t1 = time.perf_counter()  
    
    if ((t1-t0)>=1):
        if(fps>120):
            fps=120
        print("FPS: ",fps)
 
        t0 = t1
        fps = 0
    
    if cv2.pollKey() & 0xFF == ord('q'):
        break
end_time=time.time()

elapsed = end_time-start_time
avg_frame_rate=f/(elapsed+0.008)
print("AVG Total FPS: ",avg_frame_rate)
cv2.destroyAllWindows()