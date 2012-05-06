#include <linear_options/LinearQ0Learner.hh>
#include <linear_options/serialization.hh>

using namespace rl;

LinearQ0Learner::LinearQ0Learner(unsigned numActions, double alpha, double epsilon, double gamma, rl::state_abstraction& abstraction, Random rng) : 
        numActions(numActions), 
        alpha(alpha),
        epsilon(epsilon),
        gamma(gamma),
        stateAbstraction(&abstraction),
        rng(rng)
{ 
    actionValueThetas.resize(numActions); 
    for (auto it = actionValueThetas.begin(); it != actionValueThetas.end(); it++) {
        (*it) = Eigen::VectorXd::Zero(stateAbstraction->length());
    }
}

int LinearQ0Learner::getBestAction(const Eigen::VectorXd& phi) 
{
    double maxAction = 0;
    double maxValue = -1*std::numeric_limits<double>::max();

    for (unsigned i = 0; i < actionValueThetas.size(); i++) {
       double value = actionValueThetas[i].dot(phi);
       if (value > maxValue) {
           maxAction = i;
           maxValue = value;
       }
    }

    //std::cout << "Best action is " << maxAction << " with value " << maxValue << std::endl;
    return maxAction;
}

int LinearQ0Learner::epsilonGreedy(const Eigen::VectorXd& phi)
{
    lastAction  = (rng.uniform() < epsilon) ? rng.uniformDiscrete(0, numActions-1) : getBestAction(phi); 

    lastPhi = phi;
    return lastAction;
}

int LinearQ0Learner::first_action(const std::vector<float> &s)
{
    auto phi = project(s);
    return epsilonGreedy(phi);
}

int LinearQ0Learner::next_action(float reward, const std::vector<float> &s)
{
    auto phiPrime = project(s);

    actionValueThetas[lastAction] = actionValueThetas[lastAction].array() + lastPhi.array()*(reward + gamma*actionValueThetas[getBestAction(phiPrime)].dot(phiPrime) - actionValueThetas[lastAction].dot(lastPhi))*alpha;

    //std::cout << "Error " << actionValueThetas[getBestAction(phiPrime)].dot(phiPrime) - actionValueThetas[lastAction].dot(lastPhi) << std::endl;

    return epsilonGreedy(phiPrime);
}

void LinearQ0Learner::last_action(float reward)
{
    std::cerr << "**************************************************** EXECUTING LAST ACTION" << std::endl;
    actionValueThetas[lastAction] = actionValueThetas[lastAction].array() + lastPhi.array()*(reward - actionValueThetas[lastAction].dot(lastPhi))*alpha;
}

void LinearQ0Learner::setDebug(bool d) 
{

}
void LinearQ0Learner::savePolicy(const std::string& filename)
{
    std::ofstream file(filename); 
    boost::archive::text_oarchive oa(file);
    oa << actionValueThetas;
}

void LinearQ0Learner::loadPolicy(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    boost::archive::text_iarchive ia(ifs);
    ia >> actionValueThetas;
}
