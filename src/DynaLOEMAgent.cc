#include <linear_options/DynaLOEMAgent.hh>
#include <algorithm>

using namespace rl;

LinearOption* DynaLOEMAgent::getBestOption(const Eigen::VectorXd& phi)
{
    ValueComparator comp(phi);

    LinearOption* nextOption = (rng.uniform() < epsilon) ? 
        options.at(rng.uniformDiscrete(0, options.size()-1)) : 
        // FIXME This assumes every option is available everywhere
        *std::max_element(options.begin(), options.end(), comp); 

    return nextOption;
}

int DynaLOEMAgent::first_action(const std::vector<float> &s)
{
    auto phi = project(s);
    lastPhi = phi;

    currentOption = getBestOption(phi);

    return currentOption->greedyPolicy(phi);
}

int DynaLOEMAgent::next_action(float r, const std::vector<float> &s)
{
    auto phi = project(s); 

    // Find the option with highest expected discounted reward from the current state
    NextStateValueComparator comp(this, phi);
    LinearOption* maxOption = *std::max_element(options.begin(), options.end(), comp);
    double maxOptionValue = maxOption->theta.dot(optionModels[maxOption]->F*phi);

    for (auto it = options.begin(); it != options.end(); it++) {
        // Update every consistent option for which u(phi) = a
        if ((*it)->greedyPolicy(phi) == lastAction) {
            
            // Intra-Option value learning 
            Eigen::VectorXd& thetaOption = (*it)->theta;
            double U = (1 - (*it)->beta(phi))*thetaOption.dot(phi) + (*it)->beta(phi)*getBestOption(phi)->theta.dot(phi);
            thetaOption = thetaOption + alpha*(r + gamma*U - thetaOption.transpose()*phi)*phi;

            // Intra-Option model learning for transition kernel F 
            Eigen::VectorXd eta = lastPhi - gamma*(1 - (*it)->beta(phi))*phi;
            optionModels[(*it)]->F = optionModels[(*it)]->F + alpha*(gamma*(*it)->beta(phi)*phi - optionModels[(*it)]->F*eta);
        }

        // Execute one planning update for every option
        (*it)->theta = (*it)->theta.array() + alpha*optionModels[*it]->b.dot(phi) + maxOptionValue - (*it)->theta.dot(phi); 
    }

    // Pick a new option if the current one must terminate
    if (currentOption->terminate(phi)) {
        currentOption = getBestOption(phi);
    }

    lastPhi = phi;

    return currentOption->greedyPolicy(phi);
}

void DynaLOEMAgent::last_action(float r)
{
}

void DynaLOEMAgent::setDebug(bool d)
{
}
