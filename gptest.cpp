
#include "gptest.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(gptest) {
  import_array();
  numeric::array::set_module_and_type("numpy", "ndarray");

  class_<GaussianProcessTest, boost::noncopyable>(
      "GaussianProcessTest",
      init<boost::python::numeric::array, boost::python::numeric::array>())
      .def("set_theta", &GaussianProcessTest::set_theta)
      .def("calculate_neg_mll", &GaussianProcessTest::calculate_neg_mll)
      .def("predict_at", &GaussianProcessTest::predict_at);
}