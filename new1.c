#include<stdio.h>
#include<stdlib.h>
#include<dlib.h>
#include<string.h>
#include<cv.h>

using namespace dlib;

vector<IplImage*> loadsavedFaces() {
  vector<IplImage*> savedFaces;

  for (const string& facePath : savedFacesPaths) {
    IplImage* faceImage = cvLoadImage(facePath.c_str());
    if (!faceImage) {
      printf("Error loading known face image: %s\n", facePath.c_str());
      exit(1);
    }
   savedFaces.push_back(faceImage);
  }
  return savedFaces;
}

vector<dlib::rectangle> detectFacesAndExtractEncodings(IplImage* image) {
 
  IplImage* grayImage = cvCreateImage(cvGetSize(image), image->depth, 1);
  cvCvtColor(image, grayImage, CV_BGR2GRAY);

  

  frontal_face_detector detector;
  std::vector<dlib::rectangle> faces = detector((cv_matrix)grayImage);

  

  std::vector<dlib::shape> shapes;
  shape_predictor predictor;
  deserialize(predictor, "shape_predictor_68_face_landmarks.dat");
  extract_face_features((cv_matrix)grayImage, faces, shapes);

  

  vector<Mat> encodings;
  for (const dlib::shape& shape : shapes) {
    Mat encoding = Mat(1, 128, CV_32F);
    for (int i = 0; i < 128; i++) {
      encoding.ptr<float>(0)[i] = shape.get_point(i).x;
    }
    encodings.push_back(encoding);
  }


  cvReleaseImage(&grayImage);

  return encodings;
}

void CFAMA(const vector<IplImage*>& savedFaces, const vector<Mat>& currentEncodings) {

  
  for (int i = 0; i < currentEncodings.size(); i++) {
    Mat currentEncoding = currentEncodings[i];
    for (int j = 0; j < savedFaces.size(); j++) {
      IplImage* savedFace = savedFaces[j];
      Mat knownEncoding = extractFaceEncoding(savedFace);

   
 

      double distance = compareFaceDescriptors(currentEncoding, knownEncoding, 0.8);

      
      if (distance < 0.8) {
        
        string name = savedFaceNames[j];

      
        markAttendance(name);
      }
    }
  }
}


// Function to mark attendance for a person
void markAttendance(const string& name) {
  // Update attendance records
  attendanceRecords[name].push_back(currentTimestamp());

  // Save attendance records to a file
  saveAttendanceRecords(attendanceRecords);
}

int main() {
  // Load known faces and their encodings
  vector<IplImage*> savedFaces = loadsavedFaces();

  // Initialize webcam capture
  CvCapture* videoCapture = cvCaptureFromCAM(0);

  while (true) {
    // Capture a frame from the webcam
    IplImage* frame = cvQueryFrame(videoCapture);
    if (!frame) {
      break;
    }

// Display the frame

  cvShowImage("Webcam Feed", frame);

  // Check ESC pressed or not so that the displat exists and code restarts

  if (cvWaitKey(1) & 0xFF == 27) {
    break;
  }




    // Detect the faces to extract the encodings

    vector<Mat> currentEncodings = detectFacesAndExtractEncodings(frame);

    // Compare face encodings and mark attendance
    CFAMA(savedFaces, currentEncodings);

    // Display the frame with marked faces
    cv