#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string.h>

namespace fs = boost::filesystem;


static enum { OFF, READY, SELECTED } mouse; // default OFF

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
  explicit Webcam(const std::string& window_name_) :
    window_name(window_name_) {};

  /* *********************************************************************************** */
  ~Webcam() { stream.release(); };

  /* *********************************************************************************** */
  bool initialize()
  {
    stream = cv::VideoCapture(0);

    if (!stream.isOpened())
    {
      std::cout << "Cannot open webcam" << std::endl;
      return false;
    }
    w = stream.get(CV_CAP_PROP_FRAME_WIDTH);
    h = stream.get(CV_CAP_PROP_FRAME_HEIGHT);

    cv::namedWindow(window_name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);//CV_WINDOW_AUTOSIZE);
    cv::resizeWindow(window_name, 1600, 1200);
    cv::moveWindow(window_name, 0, 0);

    // Set the mouse callback fn
    cv::setMouseCallback(window_name, mouseCallback, static_cast<void*>(&mouse_click));

    return true;
  }

  /* *********************************************************************************** */
  void pause(const cv::Mat& display)
  {

    while(1)
    {
      cv::imshow(window_name, display);
      if (cv::waitKey(0))
        break;
    }
  }

  /* *********************************************************************************** */
  bool spin(cv::Mat& out)
  {
    // Spin until user clicks the screen. Capture the image
    std::cout << "\n\nStreaming from capture device. Press any key to capture frame" << std::endl;
    while(1)
    {

      cv::Mat frame;
      if (!stream.read(frame))
      {
        std::cerr << "Failed to read frame" << std::endl;
        return false;
      }

      if (cv::waitKey(30) != -1)
      {
        std::cout << "\nCapturing frame" << std::endl;
        out = frame;
        break;
      }

      // Make a dummy to put text on and disply to the user
      cv::Mat display = frame;
      cv::putText(display, "Press any key to capture a frame", cv::Point(30, 15),
                  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 0, 0), 0, CV_AA);
      cv::imshow(window_name, display);

    }

    return true;
  }

  /* *********************************************************************************** */
  bool spinWithBlend(const cv::Mat& overlay, cv::Mat& out)
  {
    // Spin until user clicks the screen. Capture the image
    std::cout << "\n\nStreaming from capture device. Press 'ESC' or 'Space' to capture frame" << std::endl;
    while(1)
    {
      cv::Mat frame;
      if (!stream.read(frame))
      {
        std::cerr << "Failed to read frame" << std::endl;
        return false;
      }

      if (cv::waitKey(30) != -1)
      {
        std::cout << "\nCapturing frame" << std::endl;
        out = frame;
        break;
      }

      // Alpha blend the two frames together
      cv::Mat display;
      cv::addWeighted(overlay, 0.3, frame, 0.7, 0.0, display);

      // Add text
      cv::putText(display, "Align your image with the mask", cv::Point(30, 15),
                  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 0, 0), 0, CV_AA);

      cv::putText(display, "Press any key to capture a frame", cv::Point(30, 40),
                  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 0, 0), 0, CV_AA);

      cv::imshow(window_name, display);

    }

    return true;
  }

  /* *********************************************************************************** */
  void findBlack(const cv::Mat& in,
                 cv::Mat& out)
  {
    Color black;
    black.setSearchRange(127, 127, 35); // big search range. Makes sure we get all the black
    black.set(cv::Vec3b(127, 127, 35));
    out = black.search(in);
    cv::dilate(out, out, cv::Mat(), cv::Point(-1, -1), 5, 1, 1);
  }

  void floodFillCenter(const cv::Mat& in,
                       cv::Mat& out)
  {
    mouse = READY;
    std::cout << "Please select a pixel on the inside of the course boundary" << std::endl;

    // Wait for the user to pick a point.
    // The mouse callback will save the point as a private member of "this"
    while(1)
    {
      cv::waitKey(30);
      if (mouse == SELECTED)
      {
        mouse = OFF;
        break;
      }
    }

    // Make an output image padded with 1 extra pixel on all edges
    cv::copyMakeBorder(in,
                       out,
                       1, 1, 1, 1,
                       cv::BORDER_REPLICATE);

    // Fill everything on the inside of the black line with 128 intensity
    // Arguments are selected based on assumption that stencil_mask is binary
    cv::floodFill(in, // input image
                  out, // output
                  mouse_click, // seed point for filling
                  cv::Scalar(255), // ???
                  0,
                  cv::Scalar(), // maximum lower brightness difference for filling (all 0)
                  cv::Scalar(), // maximum upper brightness difference for filling (all 0)
                  4 | (100 << 8)); // color of filled output

    // Shave off a 1 pixel thick border from copying operations
    out = out(cv::Rect(1, 1, w-2, h-2));
  }

  /* *********************************************************************************** */
  void floodFillOutside(const cv::Mat& in,
                        cv::Mat& out)
  {
    // Everything that is not inside or border is outside

    // Find all regions that are not inside or on the border
    cv::Mat mask = in == 0;
    cv::copyMakeBorder(mask, mask, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    // Copy a new mat with 200s into the mask
    cv::Mat outside(in.size(), CV_8U, cv::Scalar(200));
    out = in;
    outside.copyTo(out, mask);
  }

  /* *********************************************************************************** */
  void makeOverlay(const cv::Mat& in,
                   cv::Mat& out)
  {
    // Find all stuff inside and on the border, make it blue with a 0.5 alpha value
    // for alignment when calling ./scoring.o use_template
    cv::Mat middle;
    cv::inRange(in, cv::Scalar(100), cv::Scalar(100), middle);

    cv::Mat border;
    cv::inRange(in, cv::Scalar(255), cv::Scalar(255), border);

    // Return the union of the two masks
    cv::Mat mask(h, w, CV_8UC3);
    cv::bitwise_or(middle, border, mask);
    cv::copyMakeBorder(mask,
                       mask,
                       1, 1, 1, 1,
                       cv::BORDER_REPLICATE);


    // Make a teal overlay with an alpha channel
    cv::Mat overlay(h, w, CV_8UC4);
    overlay.setTo(cv::Scalar(255, 255, 0, 0.5));

    out.create(h, w, CV_8UC4);
    overlay.copyTo(out, mask);
  }

  /* *********************************************************************************** */
  void showTemplate(const cv::Mat& display)
  {
    pause(display);
  }

  /* *********************************************************************************** */
  void getInside(const cv::Mat& capture,
                 const cv::Mat& mask,
                 cv::Mat& out)
  {
    // Get the user's black lines on the inside of the black border
    cv::Mat inside_mask;
    cv::inRange(mask, cv::Scalar(100), cv::Scalar(100), inside_mask);
    cv::copyMakeBorder(inside_mask, inside_mask, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    out.create(cv::Size(w, h), CV_8UC3);
    out.setTo(cv::Scalar(255, 255, 255));
    capture.copyTo(out, inside_mask);
  }

  /* *********************************************************************************** */
  void getOutside(const cv::Mat& capture,
                  const cv::Mat& mask,
                  cv::Mat& out)
  {
    // Get the user's black lines on the outside of the black border
    cv::Mat outside_mask;
    cv::inRange(mask, cv::Scalar(200), cv::Scalar(200), outside_mask);
    cv::copyMakeBorder(outside_mask, outside_mask, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    out.create(cv::Size(w, h), CV_8UC3);
    out.setTo(cv::Scalar(255, 255, 255));
    capture.copyTo(out, outside_mask);
  }

  /* *********************************************************************************** */
  unsigned int countLinePixels(const cv::Mat& capture,
                               cv::Mat& out)
  {
    Color black;
    black.setSearchRange(127, 127, 70); // big search range. Makes sure we get all the black
    black.set(cv::Vec3b(127, 127, 35));

    out = black.search(capture);
    return cv::countNonZero( out );
  }

  void showFinal(const cv::Mat& highlight_green,
                 const cv::Mat& highlight_red,
                 const unsigned int& inside_pix,
                 const unsigned int& outside_pix,
                 const unsigned int& area_inside,
                 cv::Mat& final_display,
                 bool bonus = false)
  {
    cv::Mat green(final_display.size(), CV_8UC3, cv::Scalar(0, 255, 0));
    green.copyTo(final_display, highlight_green);

    cv::Mat red(final_display.size(), CV_8UC3, cv::Scalar(0, 0, 255));
    red.copyTo(final_display, highlight_red);

    // Add the user's score to the final display
    std::ostringstream inside_ss, outside_ss, score_ss;

    inside_ss
    << "Number of pixels inside the border: "
    << inside_pix
    << " (" <<  std::setprecision(5)
    << static_cast<double>(inside_pix) / static_cast<double>(area_inside)
    << " %)";

    outside_ss
    << "Number of pixels outside the border: "
    << outside_pix;

    if (bonus)
      score_ss
      << "Final score: "
      << (1.5*double(inside_pix) - 2.*outside_pix) / area_inside;
    else
      score_ss
      << "Final score: "
      << (double(inside_pix) - 2.*outside_pix) / area_inside;

    cv::putText(final_display, inside_ss.str(), cv::Point(30, 15),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 0, 0), 0, CV_AA);

    cv::putText(final_display, outside_ss.str(), cv::Point(30, 40),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 0, 0), 0, CV_AA);

    cv::putText(final_display, score_ss.str(), cv::Point(30, 65),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(200, 0, 0), 0, CV_AA);

    pause(final_display);
  }

private:

  cv::VideoCapture stream;
  cv::Point mouse_click;

  unsigned int w, h;
  std::string window_name;
};





/* *********************************************************************************** */
void usage()
{
  std::cerr << "\nUsage:\n"
  << "./score.o set_template TEMPLATE_NAME\n"
  << "./score.o use_template TEMPLATE_NAME\n"
  << std::endl;
}

/* *********************************************************************************** */
std::string templateFilename(const fs::path& path,
                             const std::string& fname)
{
  // Get full path to this binary
  fs::path full_path(fs::initial_path<fs::path>());
  full_path = fs::system_complete( path );
  std::string out = full_path.parent_path().string();

  // Add the filename and extension and return it
  return out.append("/"+fname);
}

/* *********************************************************************************** */
void printScore(const unsigned int& inside_pix,
                const unsigned int& outside_pix,
                const unsigned int& area_inside,
                bool bonus = false)
{
  std::cout.unsetf( std::ios::floatfield );
  std::cout.precision(5);

  std::cout
  << "Number of pixels colored inside the border: "
  << inside_pix
  << " / "
  << area_inside
  << " ("
  << static_cast<double>(inside_pix) / static_cast<double>(area_inside)
  << " %)"
  << std::endl;

  std::cout << "Number of pixels colored outside the border: " << outside_pix << std::endl;

  if (bonus)
    std::cout << "\nTotal score: " << (inside_pix - 2.*outside_pix) / area_inside << std::endl;
  else
    std::cout << "\nTotal score: " << (1.5*inside_pix - 2.*outside_pix) / area_inside << std::endl;
}


/* *********************************************************************************** */
int main(int argc, char** argv)
{

  if (argc < 3 || argc > 4)
  {
    usage();
    return -1;
  }

  // If the user chose to set the template, create one
  if (std::string(argv[1]).compare("set_template") == 0 && argc == 3)
  {
    Webcam wc("IC Robotics Competition: Set Template");

    if(!wc.initialize())
    {
      std::cerr << "Failed to initialize webcam" << std::endl;
      return 0;
    }

    // Prompt the user to capture a frame for the template
    cv::Mat capture;
    if(!wc.spin(capture))
    {
      std::cerr << "Failed to capture frame" << std::endl;
      return 0;
    }

    // Find black regions
    cv::Mat border, mask1, mask2, overlay;
    wc.findBlack(capture, border);

    // Flood fill inside black border
    wc.floodFillCenter(border, mask1); // values inside are 100

    // Flood fill outside black border
    wc.floodFillOutside(mask1, mask2); // values outside are 200

    // Build the overlay for alignment in ./scoring.o use_template
    wc.makeOverlay(mask2, overlay);

    // Save the template and overlay
    std::string out_name = templateFilename( fs::path( argv[0] ), std::string( argv[2] ) );
    cv::imwrite(out_name + std::string(".bmp"), mask2 );
    cv::imwrite(out_name + std::string("_overlay.bmp"), overlay );

    wc.showTemplate(mask2);

    return -1;
  }

  // Otherwise, the user should have chosen to use a template
  if (std::string(argv[1]).compare("use_template") == 0 && (argc == 3 || argc == 4))
  {
    // Load the template
    std::string in_name = templateFilename( fs::path( argv[0] ), std::string( argv[2] ) );
    cv::Mat mask = cv::imread( in_name + std::string(".bmp"), CV_LOAD_IMAGE_GRAYSCALE );
    cv::Mat overlay = cv::imread( in_name + std::string("_overlay.bmp"),
                                CV_LOAD_IMAGE_COLOR);

    // Open the webcam
    Webcam wc("IC Robotics Competition: Score Drawing");

    if(!wc.initialize())
    {
     std::cerr << "Failed to initialize webcam" << std::endl;
     return 0;
    }

    // Prompt the user to capture a frame after aligning under the overlay
    cv::Mat capture;
    if(!wc.spinWithBlend(overlay, capture))
    {
      std::cerr << "Failed to capture frame" << std::endl;
      return 0;
    }

    // Get the user's black lines on the inside of the black border
    cv::Mat inside_masked;
    wc.getInside(capture, mask, inside_masked);

    // Get the user's black lines on the outside of the black border
    cv::Mat outside_masked;
    wc.getOutside(capture, mask, outside_masked);

    cv::Mat highlight_green, highlight_red;
    unsigned int inside_pix = wc.countLinePixels(inside_masked, highlight_green);
    unsigned int outside_pix = wc.countLinePixels(outside_masked, highlight_red);

    cv::Mat count_area_inside, garbage;
    cv::inRange(mask, cv::Scalar(100), cv::Scalar(100), count_area_inside);
    unsigned int area_inside = wc.countLinePixels(count_area_inside, garbage);

    // Print the user's score
    if (argv[3])
    {
      printScore(inside_pix, outside_pix, area_inside, true);
      wc.showFinal(highlight_green, highlight_red,
                   inside_pix, outside_pix,
                   area_inside, capture,
                   true);
    }
    else
    {
      printScore(inside_pix, outside_pix, area_inside);
      wc.showFinal(highlight_green, highlight_red,
                   inside_pix, outside_pix,
                   area_inside, capture);
    }




    return -1;
  }

  usage();
  return 0;

}
