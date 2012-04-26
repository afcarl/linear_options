#include <linear_options/LOEMModel.hh>

bool updateWithExperience(Eigen::VectorXd s, int action, Eigen::VectorXd sprime, double reward)
{
    // Update transition model 
    Eigen::VectorXd k;
    k = (1/(1 + s.transpose()*P*sprime))*P*s;  

    P = P - k*s.tranpose()*P;
    F = F + k*(sprime - F*s).transpose();
    
    // Update reward model    
}

Eigen::VectorXd predictNextState(Eigen::VectorXd& s)
{
    return F*s;
}

double predictReward(Eigen::VectorXd& s)
{
    return b.transpose()*s;
}
