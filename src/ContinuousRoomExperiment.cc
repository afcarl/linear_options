#include <linear_options/DynaLOEMAgent.hh>
#include <linear_options/StateAbstraction.hh>
#include <linear_options/ContinuousRooms.hh>

#include <Eigen/Core>
#include <opencv/cv.h>
#include <opencv/highgui.h>

/**
 * Use radial basis functions as features
 */
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
    ContinuousRooms env("map.png", 5); 
    cv::Mat img = cv::imread("map.png");

    env.apply(ContinuousRooms::LEFT);
    env.apply(ContinuousRooms::LEFT);
    env.apply(ContinuousRooms::LEFT);
    env.apply(ContinuousRooms::LEFT);
    env.apply(ContinuousRooms::RIGHT);

    for (int i = 0; i < 125; i++) {
        double reward = env.apply(ContinuousRooms::FORWARD);
        auto s = env.sensation();
        std::cout << "Sensation x, y reward " << s[4] << " " << s[5] << " " << reward << std::endl;

        cv::circle(img, cv::Point(s[4], s[5]), 5, cv::Scalar(0, 0, 0), 1); 
        cv::imshow("world", img);

        if(cv::waitKey(30) >= 0) break;
        sleep(0.5);
    }

    // Basis location
    Eigen::MatrixXd U(4, 3);
    U << 0, 0, 0,
         1, 1, 30, 
         2, 2, 60, 
         3, 3, 90;
    auto abstraction = room_abstraction(U.transpose());

    return 0;
}
