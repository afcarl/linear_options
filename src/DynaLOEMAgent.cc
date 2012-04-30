#include <linear_options/DynaLOEMAgent.hh>
#include <algorithm>

using namespace rl;

int DynaLOEMAgent::epsilonGreedy(const Eigen::VectorXd& phi)
{
    // Epsilon-greedy action selection
    // FIXME 
    int nextAction = (rng.uniform() < epsilon) ? rng.uniformDiscrete(0, numActions-1) : 1;
    //lastAction = nextAction;
    lastState = phi;

    return nextAction;
}

LinearOption& DynaLOEMAgent::getBestOption(const Eigen::VectorXd& phi)
{
    // Follow option's policy if currently executing
    if (!currentOption.beta(phi)) {
        // TODO FIXME
        //return currentOption.policy(phi);
    }

    ValueComparator comp(phi);
    currentOption = *std::max_element(options.begin(), options.end(), comp); 

    return currentOption;
}

int DynaLOEMAgent::first_action(const std::vector<float> &s)
{
    auto phi = project(s);
    return epsilonGreedy(phi);
}

int DynaLOEMAgent::next_action(float r, const std::vector<float> &s)
{
    auto phi = project(s); 

    // Model parameters of the last executed action 
    Eigen::VectorXd& theta = lastOption.theta; 
    Eigen::VectorXd& b = lastOption.b;
    Eigen::MatrixXd& F = lastOption.F;

    // Intra-Option Learning update
    // Update every consistent option for which u(phi) = a
    for (auto it = options.begin(); it != options.end(); it++) {
        if (it->policy(phi) == lastAction) {
            Eigen::VectorXd& thetaOption = it->theta;
            double U = (1 - it->beta(phi))*thetaOption.dot(phi) + it->beta(phi)*getBestOption(phi).theta.dot(phi);
            thetaOption = thetaOption + alpha*(r + gamma*U - thetaOption.transpose()*phi)*phi;
        }
    }

    // Execute one planning update 
    NextStateValueComparator comp(phi);
    LinearOption& maxOption = *std::max_element(options.begin(), options.end(), comp);
    double maxOptionValue = maxOption.theta.dot(maxOption.F*phi);
    theta = theta.array() + alpha*b.dot(phi) + maxOptionValue - theta.dot(phi); 

    return epsilonGreedy(phi);
}

void DynaLOEMAgent::last_action(float r)
{
}

void DynaLOEMAgent::setDebug(bool d)
{
}
