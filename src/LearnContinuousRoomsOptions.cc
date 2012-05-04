#include <linear_options/StateAbstraction.hh>
#include <linear_options/ContinuousRooms.hh>
#include <linear_options/LinearQ0Learner.hh>

/**
 * We build the feature vector from a set of radial basis functions
 * spread over the space in the x, y and psi dimensions. 
 */
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
        phi(0) = s[0];
        phi(1) = s[1];
        phi(2) = s[2];
        phi(3) = s[3];
        for (int i = 4; i < U.cols(); i++) {
            phi(i) = -0.5*(s - U.col(i)).transpose()*C*(s - U.col(i));
        }

        return phi;
    }

    int length() { return U.cols() + 4; }

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
room_abstraction stateAbstraction(U.transpose(), C, 200);

ContinuousRooms env("map.png", 5);
rl::LinearQ0Learner linearQ0Learner(env.getNumActions(), 5e-4, 0.1, 0.6, stateAbstraction);

return 0;
};
