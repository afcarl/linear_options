#ifndef __CONTINUOUS_ROOMS_H__
#define __CONTINUOUS_ROOMS_H__

#include <rl_common/core.hh>
#include <opencv/cv.h>

struct ContinuousRooms : public Environment 
{
  ContinuousRooms(const std::string& map, double robotRadius, double scaling = 1.0, Random rng = Random());
   
  enum PRIMITIVE_ACTIONS { FORWARD, LEFT, RIGHT, NUM_ACTIONS };

  static const double REWARD_FAILURE = -0.01;

  static const double REWARD_SUCCESS = 1.0;

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

private:
    // RGB image corresponding to the layout of the world
    cv::Mat map;

    double robotRadius;

    double scaling;

    /**
     * Return the boundaries of a circular region of interest
     * Used for collision detection
     * @param R radius
     * @param RxV output vector
     */
    void getCircularROI(int R, std::vector<int>& RxV);

    void updateStateVector();

    // Current pose
    double x;
    double y;
    double psi;

    bool terminated;

    Random rng;

    std::vector<float> currentState;
};

#endif
