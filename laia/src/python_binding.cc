#include "laia_scheduler.h"
#include "binding.h"

using namespace laia_cache;


PYBIND11_MODULE(laia_cache, m) {
    m.doc() = "laia test"; // optional module docstring

    py::class_<LaiaScheduler>(m, "LaiaScheduler")
        .def(py::init<>())
        .def("start", &LaiaScheduler::start)
        .def("pop", &LaiaScheduler::pop)
        .def("length", &LaiaScheduler::queue_length);
} // PYBIND11_MODULE
