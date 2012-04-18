/** \file
  * Implement Intra-Option learning for options. 
  */

#ifndef _INTRA_OPTION_H_
#define _INTRA_OPTION_H_

#include <rl_common/Random.h>
#include <rl_common/core.hh>

/**
 * Intra-Option learning with or without model learning. 
 */
class IntraOptionLearner: public Agent {
public:
  /** Standard constructor
      \param numactions The number of possible actions
      \param gamma The discount factor
      \param initialvalue The initial value of each Q(s,a)
      \param alpha The learning rate
      \param epsilon The probability of taking a random action
      \param rng Initial state of the random number generator to use */
  IntraOptionLearner(int numactions, float gamma,
	   float initialvalue, float alpha, float epsilon,
	   Random rng = Random());

  /** Unimplemented copy constructor: internal state cannot be simply
      copied. */
  IntraOptionLearner(const IntraOptionLearner &);

  virtual ~IntraOptionLearner();

  virtual void setDebug(bool d);

  virtual void seedExp(std::vector<experience>);

  virtual int first_action(const std::vector<float> &s);

  virtual int next_action(float r, const std::vector<float> &s);

  virtual void last_action(float r);

  float epsilon;

private:
  const int numactions;
  const float gamma;
  const float initialvalue;
  const float alpha;

  Random rng;

  bool ACTDEBUG;
};

#endif
