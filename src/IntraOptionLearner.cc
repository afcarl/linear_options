#include <linear_options/IntraOptionLearner.hh>
#include <Eigen/Core>

IntraOptionLearner::IntraOptionLearner(int numactions, float gamma,
                   float initialvalue, float alpha, float ep, rl::state_abstraction stateAbtraction,  
                   Random rng):
  numactions(numactions), gamma(gamma),
  initialvalue(initialvalue), alpha(alpha),
  stateAbstraction(stateAbstraction),
  rng(rng)
{

  epsilon = ep;
  ACTDEBUG = false;
}

IntraOptionLearner::~IntraOptionLearner() {}

int IntraOptionLearner::first_action(const std::vector<float> &s) {

  if (ACTDEBUG){
    std::cout << "First - in state: ";
    std::cout << endl;
  }

  return 0;
}

int IntraOptionLearner::next_action(float r, const std::vector<float> &s) {

  if (ACTDEBUG){
    std::cout << "Next: got reward " << r << " in state: ";
    std::cout << endl;
  }
  
   
  
  return 0;
}

void IntraOptionLearner::last_action(float r) {

  if (ACTDEBUG){
    std::cout << "Last: got reward " << r << endl;
  }

}

void IntraOptionLearner::setDebug(bool d){
  ACTDEBUG = d;
}

void IntraOptionLearner::seedExp(std::vector<experience> seeds){
}
