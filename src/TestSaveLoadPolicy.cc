#include <linear_options/LinearQ0Learner.hh>
#include <linear_options/StateAbstraction.hh>
#include <linear_options/ContinuousRooms.hh>

struct room_abstraction : public rl::state_abstraction
{
    /**
     * @param U The mean of the RBF
     * @param C
     * @param b
     */
    room_abstraction(Eigen::MatrixXd U, Eigen::Vector3d C, double b) :
       b(b), U(U), C(C.asDiagonal()) {};

    /**
     * @param s Project the input vector in the n-d space
     */
    Eigen::VectorXd operator()(const Eigen::VectorXd& s) {
        Eigen::VectorXd phi(length());
        // The first 4 elements are binary indicator variables for floor color
        phi(0) = s[0];
        phi(1) = s[1];
        phi(2) = s[2];
        phi(3) = s[3];

        // The next 3 elements: x, y, psi
        for (int i = 0; i < U.rows(); i++) {
            phi(i + 4) = b*exp(-0.5*(s.tail(U.cols()) - U.row(i).transpose()).dot(C*(s.tail(U.cols()) - U.row(i).transpose()))); 
            if (phi(i + 4) < 0.1) { 
                phi(i + 4) = 0;
            }
        }

        return phi;
    }

    int length() { return U.rows() + 4; }

private:
    double b;
    Eigen::MatrixXd U;
    Eigen::DiagonalMatrix<double, 3, 3> C;
};

int main(void)
{
// Radial-basis functions are placed every 10 units in 
// in the x and y dimensions and every 30 degrees
Eigen::MatrixXd U(5200, 3);
int i = 0;
for (double x = 10.2/2.0; x < 200; x += 10) {
    for (double y = 10.2/2.0; y < 200; y += 10) {
        for (double psi = 0; psi <= 360; psi += 30) {
            U(i, 0) = x;
            U(i, 1) = y;
            U(i, 2) = psi;
            i += 1;
        }
    }
}
Eigen::Vector3d C(1.0/10.2, 1.0/10.2, 1/30);
room_abstraction stateAbstraction(U, C, 20);
rl::LinearQ0Learner agent(ContinuousRooms::NUM_ACTIONS, 5e-4, 0.1, 0.9, stateAbstraction);

agent.loadPolicy("agent1.rl");
agent.savePolicy("agent11.rl");

return 0;
}
