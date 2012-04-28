#include <linear_options/DynaLOEMAgent.hh>

int DynaLOEMAgent::epsilonGreedy(const Eigen::VectorXd& phi)
{
    // Epsilon-greedy action selection
    int nextAction = (rng.uniform() < epsilon)? rng.uniformDiscrete(0, numActions-1) : getBestActions(phi);
    lastAction = nextAction;
    lastState = phi;

    return nextAction;
}

LinearOption& DynaLOEMAgent::getBestAction(const Eigen::VectorXd s)
{
    // Follow option's policy if currently executing
    if (!currentOption.terminate(phi)) {
        return currentOption.policy(phi);
    }

    // Choose new option according to maini behavior policy 
    struct ValueComparator {
        bool operator() (const LinearOption& a, const LinearOption& b) { 
            return a.theta.tranpose()*s < b.theta.tranpose()*s;     
        }
    } comp;

    currentOption = std::max_element(options.begin(), options.end(), comp); 

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
    Eigen::VectorXd& theta = options[lastAction].model; 
    Eigen::VectorXd& b = options[lastAction].b;
    Eigen::MatrixXd& F = options[lastAction].F;

    // Intra-Option Learning update
    // Update every consistent option for which u(phi) = a
    for (auto it = options.begin(); it != options.end(); it++) {
        if (it->policy(phi) === lastAction) {
            Eigen::VectorXd& thetaOption = (*it);

            double U = (1 - it->beta(phi))*thetaOption.transpose()*phi + it->beta(phi)*getBestAction(phi).theta.tranpose()*phi; 
            thetaOption = thetaOption + alpha*(r + gamma*U - thetaOption.transpose()*phi)*phi;
        }
    }

    // Execute one planning update 
    // Maximum value for next state from s
    struct NextStateValueComparator {
        bool operator() (const LinearOption& a, const LinearOption& b) { 
            return a.theta.tranpose()*a.F*phi < b.theta.tranpose()*b.F*phi;     
        }
    } comp;

    LinearOption& maxOption = (std::max_element(options.begin(), options.end(), comp)->second);
    double maxOptionValue = maxOption.theta.tranpose()*maxOption.F*phi;
    theta = theta + alpha*(b.transpose()*phi + maxOptionValue - theta.tranpose()*phi); 

    return epsilonGreedy(phi);
}

void DynaLOEMAgent::last_action(float r)
{
}

void DynaLOEMAgent::setDebug(bool d)
{
}
