#ifndef __CONTINUOUS_ROOMS_H__
#define __CONTINUOUS_ROOMS_H__

#include <rl_common/core.hh>
#include <opencv/cv.h>

struct ContinuousRooms : public Environment 
{
  ContinuousRooms(const std::string& map, double robotRadius, bool randomizeInitialPosition = false, double safetyMargin = 0, Random rng = Random());
   
  enum PRIMITIVE_ACTIONS { FORWARD, LEFT, RIGHT, NUM_ACTIONS };
  enum ROOM_COLORS { GREEN, BLUE, PURPLE, YELLOW, NUM_COLORS };

  static const double REWARD_FAILURE = -0.01;

  static const double REWARD_FAILURE_MINIMA = -1.0;

  static const double REWARD_SUCCESS = 1.0;

  static const double MAX_NUMBER_STEPS = 100; 

  static const double MIN_DISPLACEMENT = 1.0;

  /**
   * @Override
   */
  const std::vector<float> &sensation() const;

  /**
   * @Override
   */
  float apply(int action);

  /**
   * @Override
   */
  virtual bool terminal() const;

  /**
   * @Override
   */
  virtual void reset();

  /**
   * @Override
   */
  virtual int getNumActions(); 

  /**
   * @Override
   */
  virtual void getMinMaxFeatures(std::vector<float> *minFeat,
                                 std::vector<float> *maxFeat);

  /**
   * @Override
   */
  virtual void getMinMaxReward(float *minR, float *maxR);

    // RGB image corresponding to the layout of the world
    // FIXME make private
    cv::Mat map;
protected:
   /**
    * @param x 
    * @param y
    * @return true if there is a collision with this configuration
    */ 
   bool isCollisionFree(double x, double y); 

private:
    std::vector<int> circularROI;

    double robotRadius;

    /**
     * Return the boundaries of a circular region of interest
     * Used for collision detection
     * @param R radius
     * @param circularROI output vector
     */
    void getCircularROI(int R, std::vector<int>& circularROI);

    /**
     * Fill the internal state vector with the relevant state information
     */
    void updateStateVector();

    /**
     * @return true if the robot has been stuck in the
     * same location for a given number of steps 
     */
    bool detectMinima();

    // Current pose
    double x;
    double lastX;
    double y;
    double lastY;
    double psi;

    bool terminated;

    bool randomPosition;
    double safetyMargin;
    Random rng;

    std::vector<float> currentState;

    unsigned minimaSteps;
};

#endif
