#include "Aboria.h"
#include <Python.h>
#include <boost/python.hpp>

using namespace Aboria;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

const size_t D = 3;
ABORIA_VARIABLE(data, double, "data");
ABORIA_VARIABLE(weights, double, "weights");
typedef Particles<std::tuple<data, weights>, D> Particles_t;
typedef Particles_t::position position;
typedef Particles_t::const_reference const_reference;

class GaussianProcessTest {
public:
  GaussianProcessTest(boost::python::numeric::array points,
                      boost::python::numeric::array data_array) {
    const size_t N = boost::python::extract<int>(points.getshape()[0]);
    std::cout << "reading " << N << " particles from numpy array" << std::endl;
    x.resize(N);
    vdouble3 low = vdouble3::Constant(std::numeric_limits<double>::max());
    vdouble3 high = vdouble3::Constant(std::numeric_limits<double>::min());
    for (size_t i = 0; i < N; ++i) {
      get<data>(x)[i] = boost::python::extract<double>(data_array[i]);
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
    amplitude2 = std::pow(boost::python::extract<double>(theta[3]), 2);
    inv_length_scale2 = 0.5 / (length_scale * length_scale);
    order = 5;

    auto kernel = [&](const_reference xi, const_reference xj) {
      return kernel_function(get<position>(xi), get<position>(xj));
    };
    auto expansions = make_h2lib_black_box_expansion<D>(order, kernel_function);
    std::cout << "creating kernel h2 matrix" << std::endl;
    K = std::make_unique<H2LibMatrix>(x, x, expansions, kernel, 1.0);
    K.compress(tol);
    std::cout << "creating cholesky decomp of h2 matrix" << std::endl;
    solver = std::make_unique<H2LibCholeskyDecomposition>(K.chol(tol));
    std::cout << "finished" << std::endl;
  }

  double calculate_neg_mll() {
    std::cout << "solving for weights" << std::endl;
    solver.solve(get<data>(x), get<weights>(x));
    const double term1 =
        -0.5 * std::inner_product(get<data>(x).begin(), get<data>(x).end(),
                                  get<weights>(x).begin(), 0);

    std::cout << "calculating log det" << std::endl;
    const double term2 = -0.5 * solver.log_determinant();

    const double term3 = -0.5 * x.size() * std::log(2 * pi);

    return -(term1 + term2 + term3);
  }

  boost::python::numeric::array
  predict_at(boost::python::numeric::array points) {
    const size_t N = boost::python::extract<int>(points.get_shape()[0]);
    y.resize(N);
    vdouble3 low = vdouble3::Constant(std::numeric_limits<double>::max());
    vdouble3 high = vdouble3::Constant(std::numeric_limits<double>::min());
    for (size_t i = 0; i < N; ++i) {
      for (size_t d = 0; d < D; ++d) {
        const double val =
            boost::python::extract<double>(points[std::make_tuple(i, d)]);
        get<position>(y)[i][d] = val;
        if (val < low[d])
          low[d] = val;
        if (val > high[d])
          high[d] = val;
      }
    }
    y.init_neighbour_search(low, high + std::numeric_limits<double>::epsilon(),
                            vbool2::Constant(false));
    auto kernel = [&](const_reference xi, const_reference xj) {
      return kernel_function(get<position>(xi), get<position>(xj));
    };
    auto expansions = make_h2lib_black_box_expansion(order, kernel_function);
    std::cout << "creating Ks and Kss h2 matrix for sample points" << std::endl;
    Ks = std::make_unique<H2LibMatrix>(y, x, expansions, kernel, 1.0);
    Ks.compress(tol);

    std::cout << "solving for weights" << std::endl;
    solver.solve(get<data>(x), get<weights>(x));

    std::cout << "multiplying by Ks" << std::endl;
    std::fill(get<data>(y).begin(), get<data>(y).end(), 0.0);
    Ks.matrix_vector_multiply(get<data>(y), get<weights>(x));

    std::cout << "finished, outputting array" << std::endl;

    npy_intp dims = y.size();
    double *p = reinterpret_cast<double *>(get<data>(y).data());
    auto object = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, p);
    boost::python::handle<> handle(object);
    boost::python::numeric::array arr(handle) return arr;
  }

private:
  static double kernel_function(const vdouble3 &xi, const vdouble3 &xj) {
    const vdouble3 dx = xi - xj;
    const double result =
        amplitude2 * std::exp(-(dx * inv_length_scale2).dot(dx));
    return result;
  };
  Particles_t x;
  Particles_t y;
  double amplitude2;
  vdouble3 inv_length_scale2;
  unsigned int order;
  std::unique_ptr<H2LibMatrix> K;
  std::unique_ptr<H2LibCholeskyDecomposition> solver;
};
