#include "Aboria.h"
#include <Python.h>
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

using namespace Aboria;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

const double pi = boost::math::constants::pi<double>();

const size_t D = 3;
ABORIA_VARIABLE(data, double, "data");
ABORIA_VARIABLE(weights, double, "weights");
typedef Particles<std::tuple<data, weights>, D, std::vector, Kdtree>
    Particles_t;
typedef Particles_t::position position;
typedef Particles_t::const_reference const_reference;

struct Kernel {
  double amplitude2;
  vdouble3 inv_length_scale2;
  double operator()(const vdouble3 &xi, const vdouble3 &xj) const {
    const vdouble3 dx = xi - xj;
    const double result =
        amplitude2 * std::exp(-(dx * inv_length_scale2).dot(dx));
    return result;
  }
};

class GaussianProcessTest {
public:
  GaussianProcessTest(boost::python::numeric::array points,
                      boost::python::numeric::array data_array) {

    tol = 1e-6;

    const size_t N = boost::python::extract<int>(points.attr("shape")[0]);
    std::cout << "reading " << N << " particles from numpy array" << std::endl;
    x.resize(N);

    /*
        PyArrayObject *points_obj = (PyArrayObject
       *)PyArray_FROM_O(points.ptr()); PyArrayObject *data_obj = (PyArrayObject
       *)PyArray_FROM_O(data_array.ptr()); double *points_data =
       reinterpret_cast<double *>(points_obj->data); double *data_data =
       reinterpret_cast<double *>(data_obj->data);
        */

    vdouble3 low = vdouble3::Constant(std::numeric_limits<double>::max());
    vdouble3 high = vdouble3::Constant(std::numeric_limits<double>::min());
    for (int i = 0; i < N; ++i) {
      get<data>(x)[i] = boost::python::extract<double>(
          data_array[boost::python::make_tuple(i, 0)]);
      for (int d = 0; d < D; ++d) {
        const double val = boost::python::extract<double>(
            points[boost::python::make_tuple(i, d)]);
        get<position>(x)[i][d] = val;
        if (val < low[d])
          low[d] = val;
        if (val > high[d])
          high[d] = val;
      }
      std::cout << "reading particle at " << get<position>(x)[i] << std::endl;
    }
    std::cout << "low = " << low << std::endl;
    std::cout << "hight = " << high << std::endl;
    x.init_neighbour_search(low,
                            high + 1e5 * std::numeric_limits<double>::epsilon(),
                            vbool3::Constant(false), 36);
    assert(x.size() == N);
    std::cout << "done reading particles from numpy array" << std::endl;
  }
  void set_theta(boost::python::numeric::array theta) {
    vdouble3 length_scale;
    for (size_t i = 0; i < 3; ++i) {
      length_scale[i] = boost::python::extract<double>(theta[i]);
    }
    kernel_function.amplitude2 =
        std::pow(boost::python::extract<double>(theta[3]), 2);
    kernel_function.inv_length_scale2 = 0.5 / (length_scale * length_scale);
    order = 6;

    std::cout << "got amplitude = " << std::sqrt(kernel_function.amplitude2)
              << " lengthscale = " << length_scale << std::endl;

    auto kernel = [&](const_reference xi, const_reference xj) {
      return kernel_function(get<position>(xi), get<position>(xj));
    };

    auto expansions = make_h2lib_black_box_expansion<D>(order, kernel_function);
    K = std::make_unique<H2LibMatrix>(x, x, expansions, kernel, 1.0);
    K->compress(tol);
    solver = std::make_unique<H2LibCholeskyDecomposition>(K->chol(tol));
  }

  double calculate_neg_mll() {
    solver->solve(get<data>(x), get<weights>(x));
    const double term1 =
        -0.5 * std::inner_product(get<data>(x).begin(), get<data>(x).end(),
                                  get<weights>(x).begin(), 0);

    const double term2 = -0.5 * solver->log_determinant();

    const double term3 = -0.5 * x.size() * std::log(2 * pi);
    std::cout << "term1 = " << term1 << " term2 = " << term2
              << " term3 = " << term3 << std::endl;

    return std::isnan(term2) ? 1e-3 * std::numeric_limits<double>::max()
                             : -(term1 + term2 + term3);
  }

  boost::python::numeric::array
  predict_at(boost::python::numeric::array points) {
    const size_t N = boost::python::extract<int>(points.attr("shape")[0]);
    y.resize(N);
    vdouble3 low = vdouble3::Constant(std::numeric_limits<double>::max());
    vdouble3 high = vdouble3::Constant(std::numeric_limits<double>::min());
    for (size_t i = 0; i < N; ++i) {
      for (size_t d = 0; d < D; ++d) {
        const double val = boost::python::extract<double>(
            points[boost::python::make_tuple(i, d)]);
        get<position>(y)[i][d] = val;
        if (val < low[d])
          low[d] = val;
        if (val > high[d])
          high[d] = val;
      }
    }
    y.init_neighbour_search(low, high + std::numeric_limits<double>::epsilon(),
                            vbool3::Constant(false));
    auto kernel = [&](const_reference xi, const_reference xj) {
      return kernel_function(get<position>(xi), get<position>(xj));
    };
    auto expansions = make_h2lib_black_box_expansion<D>(order, kernel_function);
    std::cout << "creating Ks and Kss h2 matrix for sample points" << std::endl;
    auto Ks = std::make_unique<H2LibMatrix>(y, x, expansions, kernel, 1.0);
    Ks->compress(tol);

    std::cout << "solving for weights" << std::endl;
    solver->solve(get<data>(x), get<weights>(x));

    std::cout << "multiplying by Ks" << std::endl;
    std::fill(get<data>(y).begin(), get<data>(y).end(), 0.0);
    Ks->matrix_vector_multiply(get<data>(y), 1.0, false, get<weights>(x));

    std::cout << "finished, outputting array" << std::endl;

    npy_intp dims = y.size();
    double *p = reinterpret_cast<double *>(get<data>(y).data());
    auto object = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, p);
    boost::python::handle<> handle(object);
    boost::python::numeric::array arr(handle);
    return arr;
  }

private:
  Kernel kernel_function;
  Particles_t x;
  Particles_t y;
  double amplitude2;
  vdouble3 inv_length_scale2;
  unsigned int order;
  double tol;
  std::unique_ptr<H2LibMatrix> K;
  std::unique_ptr<H2LibCholeskyDecomposition> solver;
};
