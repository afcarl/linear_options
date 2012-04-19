#include <linear_options/IntraOptionLearner.hh>
#include <linear_options/StateAbstraction.hh>

#include <Eigen/Core>
#include <functional>

struct room_abstraction : public rl::state_abstraction
{
    room_abstraction(Eigen::MatrixXd U, Eigen::Vector3d C = Eigen::Vector3d(1.0/1.2, 1.0/1.2, 1/30), double b = 20) :
       b(b), U(U), C(C.asDiagonal()) {};

    Eigen::VectorXd operator()(const Eigen::VectorXd& s) {
        Eigen::VectorXd phi(U.cols());

        for (int i = 0; i < U.cols(); i++) {
            phi(i) = -0.5*(s - U.col(i)).transpose()*C*(s - U.col(i));
        }

        return phi;
    }

private:
    double b;
    Eigen::MatrixXd U;
    Eigen::DiagonalMatrix<double, 3, 3> C;
};

int main(void)
{
    Eigen::MatrixXd U(4, 3);
    U << 0, 0, 0,
         1, 1, 30, 
         2, 2, 60, 
         3, 3, 90;

    auto abstraction = room_abstraction(U.transpose());

    Eigen::VectorXd s(3);
    s << 1.5, 1.2, 15;

    auto phi = abstraction(s);
    std::cout << phi;

    return 0;
}
