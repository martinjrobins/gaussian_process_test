#include "Aboria.h"
#include <Python.h>
#include <boost/python.hpp>

using namespace Aboria;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

const size_t D = 3;
typedef Particles<std::tuple<>, D> Particles;
typedef Particles::position position;
typedef Particles::const_reference const_reference;

class GaussianProcessTest {
public:
  GaussianProcessTest(boost::python::numeric::array points) {
    const size_t N = boost::python::extract<int>(points.get_shape()[0]);
    std::cout << "reading " << N << " particles from numpy array" << std::endl;
    x.resize(N);
    vdouble3 low = vdouble3::Constant(std::numeric_limits<double>::max());
    vdouble3 high = vdouble3::Constant(std::numeric_limits<double>::min());
    for (size_t i = 0; i < N; ++i) {
      for (size_t d = 0; d < D; ++d) {
        const double val =
            boost::python::extract<double>(points[std::make_tuple(i, d)]);
        get<position>(x)[i][d] = val;
        if (val < low[d])
          low[d] = val;
        if (val > high[d])
          high[d] = val;
      }
    }
    x.init_neighbour_search(low, high + std::numeric_limits<double>::epsilon(),
                            vbool3::Constant(false));
    std::cout << "done reading particles from numpy array" << std::endl;
  }
  void set_theta(boost::python::numeric::array theta) {
    const double tol = 1e-6;
    vdouble3 length_scale;
    for (size_t i = 0; i < 3; ++i) {
      length_scale[i] = boost::python::extract<double>(theta[i]);
    }
    const double amplitude2 =
        std::pow(boost::python::extract<double>(theta[3]), 2);
    vdouble3 inv_length_scale2 = 0.5 / (length_scale * length_scale);

    auto kernel_function = [&](const vdouble3 &xi, const vdouble3 &xj) {
      const vdouble3 dx = xi - xj;
      return amplitude * std::exp(-(dx * inv_length_scale2).dot(dx));
    };
    auto kernel = [&](const_reference xi, const_reference xj) {
      return kernel_function(get<position>(xi), get<position>(xj));
    };
    auto expansions = make_h2lib_black_box_expansion(5, kernel_function);
    std::cout << "creating kernel h2 matrix" << std::endl;
    K = std::make_unique<H2LibMatrix>(x, x, expansions, kernel, 1.0);
    K.compress(tol);
    std::cout << "creating cholesky decomp of h2 matrix" << std::endl;
    solver = std::make_unique<H2LibCholeskyDecomposition>(K.chol(tol));
    std::cout << "finished" << std::endl;
  }
  void calculate_mu_at(boost::python::numeric::array points) {
    const size_t N = boost::python::extract<int>(points.get_shape()[0]);
    Particles y(N);
    vdouble3 low = vdouble3::Constant(std::numeric_limits<double>::max());
    vdouble3 high = vdouble3::Constant(std::numeric_limits<double>::min());
    for (size_t i = 0; i < N; ++i) {
      for (size_t d = 0; d < D; ++d) {
        const double val =
            boost::python::extract<double>(points[std::make_tuple(i, d)]);
        get<position>(x)[i][d] = val;
        if (val < low[d])
          low[d] = val;
        if (val > high[d])
          high[d] = val;
      }
    }
  }
  void calculate_s_at(boost::python::numeric::array points);

private:
  Particles x;
  std::unique_ptr<H2LibMatrix> K;
  std::unique_ptr<H2LibCholeskyDecomposition> solver;
};