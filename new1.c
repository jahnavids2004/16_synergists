#include <opencv2/opencv.hpp>
#include <dlib/image_processing.hpp>
#include <dlib/facelandmark.hpp>

using namespace cv;
using namespace dlib;

// Function to load known faces and their encodings

vector<Mat> loadKnownFaces() {
  vector<Mat> knownFaces;
  
  // Load face images and convert them to matrices
  for (const string& facePath : knownFacePaths) {
    Mat faceImage = imread(facePath);
    knownFaces.push_back(faceImage);
  }
  return knownFaces;
}

// Function to detect faces and extract encodings from an image
vector<dlib::rectangle> detectFacesAndExtractEncodings(const Mat& image) {
  // Convert image to grayscale
  Mat grayImage;
  cvtColor(image, grayImage, COLOR_BGR2GRAY);

  // Detect faces using dlib's face detection algorithm
  frontal_face_detector detector;
  std::vector<dlib::rectangle> faces = detector(grayImage);

  // Extract encodings for each detected face
  std::vector<dlib::shape> shapes;
  shape_predictor predictor;
  deserialize(predictor, "shape_predictor_68_face_landmarks.dat");
  extract_face_features(grayImage, faces, shapes);

  // Convert dlib shapes to OpenCV Mat format
  vector<Mat> encodings;
  for (const dlib::shape& shape : shapes) {
    Mat encoding = Mat(1, 128, CV_32F);
    for (int i = 0; i < 128; i++) {
      encoding.ptr<float>(0)[i] = shape.get_point(i).x;
    }
    encodings.push_back(encoding);
  }

  return encodings;
}

// Function to compare face encodings and mark attendance
void compareFacesAndMarkAttendance(const vector<Mat>& knownEncodings, const vector<Mat>& currentEncodings) {
  // Compare each current encoding with the known encodings
  for (int i = 0; i < currentEncodings.size(); i++) {
    Mat currentEncoding = currentEncodings[i];
    for (int j = 0; j < knownEncodings.size(); j++) {
      Mat knownEncoding = knownEncodings[j];

      // Calculate the distance between the encodings
      double distance = compareFaceDescriptors(currentEncoding, knownEncoding, 0.5);

      // If the distance is small enough, mark attendance for the corresponding person
      if (distance < 0.5) {
        // Identify the corresponding person
        string name = knownFaceNames[j];

        // Mark attendance for the person
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
  vector<Mat> knownFaces = loadKnownFaces();

  // Capture an image or load an existing image
  Mat image = imread("test.jpg"); // or captureImageFromWebcam()

  // Detect faces and extract encodings
  vector<Mat> currentEncodings = detectFacesAndExtractEncodings(image);

  // Compare face encodings and mark attendance
  compareFacesAndMarkAttendance(knownFaces, currentEncodings);

  // Display the image with marked faces (if applicable)

  // Print attendance records

  return 0;
}