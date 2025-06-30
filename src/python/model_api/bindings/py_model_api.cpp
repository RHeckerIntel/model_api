#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_classification(nb::module_& m);

NB_MODULE(_model_api, m) {
    m.doc() = "Nanobind binding for OpenVINO Model API library";
    init_classification(m);
}
