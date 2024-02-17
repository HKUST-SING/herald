#include "laia_scheduler.h"
#include "binding.h"
#include "topk_scheduler.h"

using namespace laia_cache;


PYBIND11_MODULE(laia_cache, m) {
    m.doc() = "laia test"; // optional module docstring

    py::class_<LaiaScheduler>(m, "LaiaScheduler")
        .def(py::init<>())
        .def("start", &LaiaScheduler::start)
        .def("pop", &LaiaScheduler::pop)
        .def("length", &LaiaScheduler::queue_length);

    py::class_<TopkScheduler>(m, "TopkScheduler")
        .def(py::init<>())
        .def("start", &TopkScheduler::start)
        .def("pop", &TopkScheduler::pop)
        .def("pop_from_local_worker", &TopkScheduler::pop_from_local_worker)
        .def("length", &TopkScheduler::queue_length);
} // PYBIND11_MODULE
