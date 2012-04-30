#include <linear_options/DynaLOEMAgent.hh>
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

    std::vector<float> test(3);
    test[0] = 1.5;
    test[1] = 1.2;
    test[2] = 15;
    std::vector<double> test2(test.begin(), test.end());

    Eigen::Map<Eigen::VectorXd>(&test2[0], test2.size()); 

    auto phi = abstraction(s);
    std::cout << phi;

    return 0;
}
