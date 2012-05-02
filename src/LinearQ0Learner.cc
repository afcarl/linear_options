#include <linear_options/LinearQ0Learner.hh>

int LinearQ0Learner::getBestAction(const Eigen::VectorXd& phi) 
{
    double maxValue;
    double maxAction = -1*std::numeric_limits<double>::max();
    for (unsigned i = 0; i < actionValueThetas.size(); i++) {
       if (actionValueThetas[i].dot(phi) > maxValue) {
           maxAction = i;
       }
    }

    return maxAction;
}

int LinearQ0Learner::first_action(const std::vector<float> &s)
{
    auto phi = project(s);

    lastAction  = (rng.uniform() < epsilon) ? rng.uniformDiscrete(0, numActions-1) : getBestAction(phi); 
    lastPhi = phi;

    return lastAction;
}

int LinearQ0Learner::next_action(float r, const std::vector<float> &s)
{
    auto phiPrime = project(s);

    actionValueThetas[lastAction] = actionValueThetas[lastAction].array() + lastPhi.array()*(reward + gamma*actionValueThetas[getBestAction(phiPrime)].dot(phiPrime) - actionValueThetas[lastAction].dot(lastPhi))*alpha;

    lastAction  = (rng.uniform() < epsilon) ? rng.uniformDiscrete(0, numActions-1) : getBestAction(phiPrime); 
    lastPhi = phiPrime;

    return lastAction;
}

void LinearQ0Learner::last_action(float r)
{
    // TODO Check if this is correct
    actionValueThetas[lastAction] = actionValueThetas[lastAction].array() + lastPhi.array()*(reward - actionValueThetas[lastAction].dot(lastPhi))*alpha;
}

void LinearQ0Learner::setDebug(bool d) 
{

}
