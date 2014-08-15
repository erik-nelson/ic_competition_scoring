#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <limits>

static enum { OFF, READY, SELECTED } mouse; // defaults to OFF

/* *********************************************************************************** */
void mouseCallback(int event, int x, int y, int flags, void *point)
{
  if (mouse != READY)
    return;

  if (event == CV_EVENT_LBUTTONDOWN)
  {
    // Cast back to cv::Point pointer
    cv::Point *p = static_cast<cv::Point*>(point);
    p->x = x;
    p->y = y;

    mouse = SELECTED;
  }
}

class Color
{
public:

  /* *********************************************************************************** */
  // Construct with default HSV search volume (5, 50, 50)
  explicit Color() : dh(5), ds(50), dv(50) {};

  /* *********************************************************************************** */
  ~Color() {};

  /* *********************************************************************************** */
  void setSearchRange(const unsigned int& dh_,
                      const unsigned int& ds_,
                      const unsigned int& dv_)
  {
    dh = dh_;
    ds = ds_;
    dv = dv_;
  }

  /* *********************************************************************************** */
  void set(const cv::Vec3b& source_color)
  {
    // Build the search volume. Bound each dimension to [0, 255]
    unsigned int min_h = source_color(0) - dh < 0   ? 0   : source_color(0) - dh;
    unsigned int min_s = source_color(1) - ds < 0   ? 0   : source_color(1) - ds;
    unsigned int min_v = source_color(2) - dv < 0   ? 0   : source_color(2) - dv;
    unsigned int max_h = source_color(0) + dh > 255 ? 255 : source_color(0) + dh;
    unsigned int max_s = source_color(1) + ds > 255 ? 255 : source_color(1) + ds;
    unsigned int max_v = source_color(2) + dv > 255 ? 255 : source_color(2) + dv;

    hsv_min = cv::Scalar(min_h, min_s, min_v);
    hsv_max = cv::Scalar(max_h, max_s, max_v);
  }

  /* *********************************************************************************** */
  cv::Mat search(const cv::Mat& frame_hsv)
  {
    // Get all pixels that lie within the predefined HSV bounding box
    cv::Mat out;
    cv::inRange(frame_hsv, hsv_min, hsv_max, out);
    return out;
  }

private:

  cv::Scalar hsv_min;
  cv::Scalar hsv_max;
  unsigned int dh, ds, dv;

};

class Webcam
{
public:

  /* *********************************************************************************** */
  explicit Webcam(const std::string& capture_window_name_,
                  const std::string& stream_window_name_) :
    capture_window_name(capture_window_name_),
    stream_window_name(stream_window_name_) {};

  /* *********************************************************************************** */
  ~Webcam() {};

  /* *********************************************************************************** */
  bool initialize()
  {
    stream = cv::VideoCapture(1);

    if (!stream.isOpened())
    {
      std::cout << "Cannot open webcam" << std::endl;
      return false;
    }
    w = stream.get(CV_CAP_PROP_FRAME_WIDTH);
    h = stream.get(CV_CAP_PROP_FRAME_HEIGHT);

    cv::namedWindow(stream_window_name, CV_WINDOW_AUTOSIZE);
    cv::moveWindow(stream_window_name, 0, 0);

    // Set the mouse callback function
    cv::setMouseCallback(stream_window_name, mouseCallback, static_cast<void*>(&mouse_click));

    std::cout << "\n\nStreaming from capture device. Press 'ESC' or 'Space' to capture frame" << std::endl;
    return true;
  }

  /* *********************************************************************************** */
  void pause(const cv::Mat& display, const std::string& name)
  {
    while(1)
    {
      cv::imshow(name, display);
      // Wait 30 ms for keypress
      unsigned int k = cv::waitKey(30);
      if (k == 27 || k == 32) // 'ESC' or 'SPACE'
        break;
    }
  }

  /* *********************************************************************************** */
  void spin(const unsigned int& spin_rate)
  {
    cv::Mat capture_frame;
    while(1)
    {
      cv::Mat frame;
      if (!stream.read(frame))
      {
        std::cerr << "Failed to read frame" << std::endl;
        return;
      }

      cv::imshow(stream_window_name, frame);

      // Wait 30 ms for keypress
      unsigned int k = cv::waitKey(spin_rate);
      if (k == 27 || k == 32) // 'ESC' or 'SPACE'
      {
        std::cout << "Capturing frame" << std::endl;
        capture_frame = frame;
        break;
      }
    }
    capture(capture_frame);
  }

  /* *********************************************************************************** */
  void capture(const cv::Mat& frame)
  {

    // Prompt the user for their color so we know what to search for
    mouse = READY;
    std::cout << "\nUsing the mouse, select your marker's color from the image." << std::endl;


    cv::Mat thresholded;
    while (1)
    {
      cv::waitKey(30);

      // Once the user clicks this will activate
      if (mouse == SELECTED)
      {
        mouse = OFF;

        // Convert the image to HSV color space. Easier to lookup specific colors in HSV than BGR
        cv::Mat frame_hsv = frame.clone();
        cv::cvtColor(frame, frame_hsv, CV_BGR2HSV);

        // What color was under the mouse?
        cv::Vec3b c = frame_hsv.at<cv::Vec3b>(mouse_click.y, mouse_click.x);
        color.set(c);

        // Search the HSV frame for all pixel instances of the specified color
        color.setSearchRange(5, 50, 50);
        cv::Mat mask = color.search(frame_hsv);

#if 0
        // Remove small contours from the image
        cv::Mat canny;
        std::vector< std::vector< cv::Point> > contours;
        std::vector< cv::Vec4i > hierarchy;
        cv::Canny(mask, canny, 100, 300, 3);
        cv::findContours(canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
#endif

        // Erode the mask so that we don't miss any pixels
        mask = cv::Scalar::all(255) - mask;
        cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
        mask = cv::Scalar::all(255) - mask;

        // "thresholded" is just a mask. Copy the source image into the mask to show color overlay
        cv::Mat thresh;
        frame.copyTo(thresh, mask);

        // Create a new window with their marker color
        cv::namedWindow(capture_window_name, CV_WINDOW_NORMAL);
        cv::resizeWindow(capture_window_name, w, h);
        cv::moveWindow(capture_window_name, 900, 0);
        cv::setMouseCallback(capture_window_name, mouseCallback, static_cast<void*>(&mouse_click));

        // Show the image to the user
        cv::imshow(capture_window_name, thresh);
        cv::waitKey(30);

        // Does the user like the line they made?
        std::cout << "Is this a good representation of your line? (y/n)" << std::endl;

        std::string choice("");
        std::getline(std::cin, choice);
        std::transform(choice.begin(), choice.end(), choice.begin(), ::tolower);

        if (!choice.compare("y") || !choice.compare("yes"))
        {
          thresholded = thresh;
          break;
        }
        else if (!choice.compare("n") || !choice.compare("no"))
        {
          std::cout << "\nUsing the mouse, select your marker's color from the image." << std::endl;
          mouse = READY;
        }
        else
        {
          std::cout << choice.c_str() << " was not an option.\n" << std::endl;
          std::cout << "\nUsing the mouse, select your marker's color from the image." << std::endl;
          mouse = READY;
        }
      }
    }


#if 0
    while (mouse != SELECTED)
    {
      cv::waitKey(30);
      // Still need to make sure they didn't pick black or white. Ask them if they want to repick?
    }
#endif

    // Find the pixels that lie outside of the boundary
    cv::Mat outside_mask = fillContours(frame);
    cv::Mat thresholded_outside;
    thresholded.copyTo(thresholded_outside, outside_mask(cv::Rect(1, 1, w-2, h-2)));

    pause(outside_mask, capture_window_name);

    finalDisplay(frame, thresholded, thresholded_outside);
  }

  /* *********************************************************************************** */
  cv::Mat fillContours(const cv::Mat& frame)
  {
    Color black;
    black.setSearchRange(127, 127, 40); // big search range. Makes sure we get all the black
    black.set(cv::Vec3b(127, 127, 15));
    cv::Mat stencil_mask = black.search(frame);

    cv::imshow(capture_window_name, stencil_mask);

    // Prompt the user to click a point on the inside of the object
    mouse = READY;
    std::cout << "\nUsing the mouse, click somewhere inside the boundary." << std::endl;

    while (mouse != SELECTED)
    {
      cv::waitKey(30);
      // Still need to make sure they didn't pick black or white. Ask them if they want to repick?
    }

    // The stencil_mask is now a binary image with 1 where black, and 0 elsewhere
    cv::Mat filled;

    // Make an output image padded with 1 extra pixel on all edges
    cv::copyMakeBorder(stencil_mask,
                       filled,
                       1, 1, 1, 1,
                       cv::BORDER_REPLICATE);

    // Fill everything on the inside of the black line with 128 intensity
    // Arguments are selected based on assumption that stencil_mask is binary
    cv::floodFill(stencil_mask, // input image
                  filled, // output
                  mouse_click, // seed point for filling
                  cv::Scalar(255), // ???
                  0,
                  cv::Scalar(), // maximum lower brightness difference for filling (all 0)
                  cv::Scalar(), // maximum upper brightness difference for filling (all 0)
                  4 | (128 << 8)); // color of filled output

    // Create a new binary mask for the inside of the course
    cv::Mat outside_mask, outside_dilated;
    cv::inRange(filled, cv::Scalar(128, 128, 128), cv::Scalar(128, 128, 128), outside_mask);
    outside_mask = cv::Scalar::all(255) - outside_mask;

    // Erode the mask; we need to grow it to the edges of the thick black lines
    cv::erode(outside_mask, outside_dilated, cv::Mat(), cv::Point(-1, -1), 10, 1, 1);

    return outside_dilated;
  }

  /* *********************************************************************************** */
  void finalDisplay(const cv::Mat& frame,
                    const cv::Mat& thresholded,
                    const cv::Mat& thresholded_outside)
  {
    cv::Mat final = frame;

    // Find all values of the user's color, paint them green.
    // Then find all values of the user's color that lie
    // outside of the mask. Paint those red.
    // The red will overwrite some of the green.
    // Display the user's screenshot with green and red painted on top

    // Make the masks binary
    cv::Mat thresholded_bin = thresholded > 0;
    cv::Mat thresholded_outside_bin = thresholded_outside > 0;

    cv::Mat green(final.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    green.copyTo(final, thresholded_bin);

    cv::Mat red(final.size(), CV_8UC3, cv::Scalar(0, 0, 255));
    red.copyTo(final, thresholded_outside_bin);

    pause(final, capture_window_name);

    // Prompt user to close window
  }


private:

  cv::VideoCapture stream;
  cv::Point mouse_click;

  unsigned int w, h;
  std::string capture_window_name;
  std::string stream_window_name;

  Color color;
};

int main(int argc, char** argv)
{
  Webcam wc("Captured Image", "IC Robotics Competition Scoring");

  if(!wc.initialize())
  {
    std::cerr << "Failed to initialize webcam" << std::endl;
    return -1;
  }

  // 30 ms per frame
  wc.spin(30);

  return 0;
}
