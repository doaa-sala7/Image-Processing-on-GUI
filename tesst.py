from email.mime import image
from msilib.schema import ComboBox
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog,QComboBox 
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
import sys
import cv2 
# from skimage.color import rgb2gray
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets 
from PyQt5.QtGui import QImage
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray




class UI(QMainWindow):
    
	def __init__(self):


		super(UI, self).__init__()
		# Load the ui file
		uic.loadUi("cv3.ui", self)

		# Define our widgets
		self.button = self.findChild(QPushButton, "pushButton")
		self.button_2 = self.findChild(QPushButton, "pushButton_2")
		self.comboBox = self.findChild(QComboBox, "comboBox")


		self.label = self.findChild(QLabel, "label_2")
		self.label_2 = self.findChild(QLabel, "label")
		self.label_3 = self.findChild(QLabel, "label_3")
		self.label_4 = self.findChild(QLabel, "label_4")

		# self.comboBox =  self.findChild(QComboBox, "comboBox_4")

		# Click The Dropdown Box
		self.button.clicked.connect(self.generate_histogram)
		self.button_2.clicked.connect(self.apply_freq_filter)

		self.actionChoose_Photo.triggered.connect(self.clicker)

		
		# self.comboBox.activated.connect(self.filters)
						
		# Show The App
		self.show()
		self.fname = ""
        
	def clicker(self):
		self.fname = QFileDialog.getOpenFileName(self, "Open File", "C:\\pyqt5\\New folder\\cvtask", "All Files (*)")

		# Open The Image
		if self.fname:
			self.pixmap = QPixmap(self.fname[0])
			
			# Add Pic to label
			self.label.setPixmap(self.pixmap)

	def generate_histogram(self):
		img = Image.open(self.fname[0])
		img = np.asarray(img)
		flat = img.flatten()

		def get_histogram(image, bins):
    	# array with size of bins, set to zeros
			histogram = np.zeros(bins)
    
    	# loop through pixels and sum up counts of pixels
			for pixel in image:
				histogram[pixel] += 1
    
    	# return our final result
			return histogram

		hist = get_histogram(flat, 256)
		plt.cla()
		plt.plot(hist)
		plt.savefig('histo.png',bbox_inches='tight',transparent=True, pad_inches=0)
		pixmap = QPixmap("histo.png")
		self.label_4.setPixmap(pixmap)
		self.label_4.setScaledContents(True)

		def cumsum(a):
			a = iter(a)
			b = [next(a)]
			for i in a:
				b.append(b[-1] + i)
			return np.array(b)

		# execute the fn
		cs = cumsum(hist)

		nj = (cs - cs.min()) * 255
		N = cs.max() - cs.min()

		# re-normalize the cdf
		cs = nj / N

		cs = cs.astype('uint8')
		img_new = cs[flat]
		img_new = np.reshape(img_new, img.shape)
		cv2.imwrite("histo_filtered.jpg", img_new)
		pixmap = QPixmap("histo_filtered.jpg")
		self.label_3.setPixmap(pixmap)
		self.label_3.setScaledContents(True)
		# print(img)

	def apply_freq_filter(self):
		if self.comboBox.currentText() == "Low Pass(Spatial)":
			self.low_pass_spatial()
		elif self.comboBox.currentText() == "High Pass(Spatial)":
			self.high_pass_spatial()
		elif self.comboBox.currentText() == "Low Pass(Frequency)":
			self.low_pass_frequency()
		elif self.comboBox.currentText() == "High Pass(Frequency)":
			self.high_pass_frequency()
		elif self.comboBox.currentText() == "Median":
			self.median_spatial()
		elif self.comboBox.currentText() == "Laplacian":
			self.laplacian_spatial()
		
	def high_pass_spatial(self):
		img = cv2.imread(self.fname[0])
		kernel = np.array([[-1.0, -1.0], 
                   [2.0, 2.0],
                   [-1.0, -1.0]])

		kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

		#filter the source image
		high_pass_spatial = cv2.filter2D(img,-1,kernel)
		cv2.imwrite("Highpass_filtered.jpg", high_pass_spatial)
		pixmap = QPixmap("Highpass_filtered.jpg")
		self.label_2.setPixmap(pixmap)
		self.label_2.setScaledContents(True)





	def low_pass_spatial(self):
		img = cv2.imread(self.fname[0])
		kernal=np.ones((9,9),np.float32)/81
		low_pass_spatial=cv2.filter2D(img,-1,kernal)
		print(low_pass_spatial)
		cv2.imwrite("Lowpass_filtered.jpg", low_pass_spatial)
		pixmap = QPixmap("Lowpass_filtered.jpg")
		self.label_2.setPixmap(pixmap)
		self.label_2.setScaledContents(True)
		# print(low_pass_spatial)

	# /////////////////////////////////////////////
	# def freq_filters(self):
	# 	img_rows, img_cols = img.shape


	def filterapply(self,mask):
		img = cv2.imread(self.fname[0],0)
		img = rgb2gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  #gray scale
		dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
		dft_shift = np.fft.fftshift(dft)
		# magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
		fshift = dft_shift * mask
		# fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
		f_ishift = np.fft.ifftshift(fshift)
		img_back = cv2.idft(f_ishift)
		img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
		return img_back

	def low_pass_frequency(self):
		im = self.fname[0]
		img = cv2.imread(im,0)
		rows, cols = img.shape
		crow, ccol = int(rows / 2), int(cols / 2)
		mask = np.ones((rows, cols, 2), np.uint8)
		r = 70
		center = [crow, ccol]
		x, y = np.ogrid[:rows, :cols]
		mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
		mask[mask_area] = 0
		img_back= self.filterapply(mask)
		plt.cla()
		plt.imshow(img_back, cmap='gray')
		plt.savefig('Lowpass_freq.png',bbox_inches='tight',transparent=True, pad_inches=0)
		# print(img_back)

		# cv2.imwrite("Lowpass_freq.jpeg", img_back)
		pixmap = QPixmap("Lowpass_freq.png")
		self.label_2.setPixmap(pixmap)
		self.label_2.setScaledContents(True)



	def high_pass_frequency(self):
		img = cv2.imread(self.fname[0],0)
		rows, cols = img.shape
		crow, ccol = int(rows / 2), int(cols / 2)
		mask = np.zeros((rows, cols, 2), np.uint8)
		r = 70
		center = [crow, ccol]
		x, y = np.ogrid[:rows, :cols]
		mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
		mask[mask_area] = 1
		img_back= self.filterapply(mask)
		# print(img_back)
		plt.cla()
		plt.imshow(img_back, cmap='gray')
		plt.savefig('Highpass_freq.png',bbox_inches='tight',transparent=True, pad_inches=0)
		# cv2.imwrite("Highpass_freq.png", img_back)
		pixmap = QPixmap("Highpass_freq.png")
		self.label_2.setPixmap(pixmap)
		self.label_2.setScaledContents(True)
	
	
	def median_spatial(self):
		img = cv2.imread(self.fname[0])
		median_using_cv2 = cv2.medianBlur(img, 3)
		print(median_using_cv2)
		cv2.imwrite("median_filtered.jpg", median_using_cv2)
		pixmap = QPixmap("median_filtered.jpg")
		self.label_2.setPixmap(pixmap)
		self.label_2.setScaledContents(True)

	def laplacian_spatial(self):
		img = cv2.imread(self.fname[0])
		laplacian_using_cv2=cv2.Laplacian(img,cv2.CV_64F,ksize=5)
		cv2.imwrite("laplacian_filtered.jpg", laplacian_using_cv2)
		pixmap = QPixmap("laplacian_filtered.jpg")
		self.label_2.setPixmap(pixmap)
		self.label_2.setScaledContents(True)



# Initialize The App
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
