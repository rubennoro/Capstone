// SPDX-License-Identifier: MIT
// rigpy: pybind11 bindings of rig::camera_model and rig::imageprep (T4). Used by the offline `images` stage and by
// deployment so that both run the identical C++ image preparation (PIPELINE 2 "Image views").
//
// numpy conventions:
// - Images in: uint8, C-contiguous, (H, W) or (H, W, C) with C in 1..4. Wrapped as a cv::Mat without a copy
//   (non-contiguous or non-uint8 inputs raise; pass np.ascontiguousarray(img)).
// - Images out: numpy arrays that own a heap cv::Mat through a capsule (no copy).
// - RemapTable.map1/map2/valid_mask: read-only views into the table (they keep the table alive).
// - Errors: kInvalidArgument -> ValueError, kNotFound -> KeyError, everything else -> RuntimeError.
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rig/camera_model/camera_model.hpp"
#include "rig/camera_model/coverage.hpp"
#include "rig/imageprep/imageprep.hpp"

namespace py = pybind11;

namespace {

[[noreturn]] void raise(const rig::Status& s) {
  switch (s.code()) {
    case rig::StatusCode::kInvalidArgument: throw py::value_error(s.to_string());
    case rig::StatusCode::kNotFound: throw py::key_error(s.to_string());
    default: throw std::runtime_error(s.to_string());
  }
}

template <class T>
T unwrap(rig::Result<T>&& r) {
  if (!r.ok()) raise(r.status());
  return std::move(r).value();
}

// numpy uint8 image -> cv::Mat header over the same memory (no copy). `arr` must outlive the Mat.
cv::Mat as_mat(const py::array& arr) {
  if (!py::isinstance<py::array_t<uint8_t>>(arr) || arr.dtype().kind() != 'u' || arr.itemsize() != 1) {
    throw py::value_error("expected a uint8 image");
  }
  if (!(arr.flags() & py::array::c_style)) throw py::value_error("expected a C-contiguous image (np.ascontiguousarray)");
  const int nd = static_cast<int>(arr.ndim());
  if (nd != 2 && nd != 3) throw py::value_error("expected an (H, W) or (H, W, C) image");
  const int h = static_cast<int>(arr.shape(0)), w = static_cast<int>(arr.shape(1));
  const int c = nd == 3 ? static_cast<int>(arr.shape(2)) : 1;
  if (c < 1 || c > 4) throw py::value_error("expected 1..4 channels");
  if (h == 0 || w == 0) throw py::value_error("empty image");
  return cv::Mat(h, w, CV_MAKETYPE(CV_8U, c), const_cast<void*>(arr.data()), static_cast<size_t>(arr.strides(0)));
}

// cv::Mat -> numpy array owning the Mat (no copy).
py::array to_array(cv::Mat m) {
  if (m.empty()) return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, 0});
  auto* heap = new cv::Mat(std::move(m));
  py::capsule owner(heap, [](void* p) { delete static_cast<cv::Mat*>(p); });
  const py::ssize_t es1 = static_cast<py::ssize_t>(heap->elemSize1());
  std::vector<py::ssize_t> shape{heap->rows, heap->cols};
  std::vector<py::ssize_t> strides{static_cast<py::ssize_t>(heap->step[0]), static_cast<py::ssize_t>(heap->elemSize())};
  if (heap->channels() > 1) {
    shape.push_back(heap->channels());
    strides.push_back(es1);
  }
  py::dtype dt;
  switch (heap->depth()) {
    case CV_8U: dt = py::dtype::of<uint8_t>(); break;
    case CV_16S: dt = py::dtype::of<int16_t>(); break;
    case CV_16U: dt = py::dtype::of<uint16_t>(); break;
    case CV_32F: dt = py::dtype::of<float>(); break;
    default: delete heap; throw std::runtime_error("unsupported cv::Mat depth");
  }
  return py::array(dt, shape, strides, heap->data, owner);
}

// Read-only numpy view of a table's Mat: the capsule holds a refcounted header copy, so the pixels stay alive even
// if the table is destroyed first.
py::array view(const cv::Mat& m) {
  py::array a = to_array(m);
  a.attr("flags").attr("writeable") = false;
  return a;
}

py::array_t<double> points_in(const py::array& a, int dim, const char* what) {
  auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
  if (!arr) throw py::value_error(std::string(what) + ": expected a float array");
  if (!((arr.ndim() == 2 && arr.shape(1) == dim) || (arr.ndim() == 1 && arr.shape(0) == dim))) {
    throw py::value_error(std::string(what) + ": expected shape (N, " + std::to_string(dim) + ") or (" +
                          std::to_string(dim) + ",)");
  }
  return arr;
}

py::tuple project(const rig::CameraModel& m, const py::array& pts, bool jacobian) {
  auto P = points_in(pts, 3, "project");
  const py::ssize_t n = P.ndim() == 1 ? 1 : P.shape(0);
  py::array_t<double> uv({n, py::ssize_t{2}});
  py::array_t<bool> ok(n);
  py::array_t<double> J({jacobian ? n : py::ssize_t{0}, py::ssize_t{2}, py::ssize_t{3}});
  const double* p = P.data();
  double* u = uv.mutable_data();
  bool* o = ok.mutable_data();
  double* j = J.mutable_data();
  {
    py::gil_scoped_release nogil;
    for (py::ssize_t i = 0; i < n; ++i) {
      Eigen::Vector2d r;
      Eigen::Matrix<double, 2, 3> Ji;
      o[i] = m.project(Eigen::Vector3d(p[3 * i], p[3 * i + 1], p[3 * i + 2]), &r, jacobian ? &Ji : nullptr);
      u[2 * i] = r.x();
      u[2 * i + 1] = r.y();
      if (jacobian) {
        for (int a = 0; a < 2; ++a)
          for (int b = 0; b < 3; ++b) j[6 * i + 3 * a + b] = o[i] ? Ji(a, b) : rig::kNaN;
      }
    }
  }
  if (jacobian) return py::make_tuple(uv, ok, J);
  return py::make_tuple(uv, ok);
}

py::tuple unproject(const rig::CameraModel& m, const py::array& pix) {
  auto U = points_in(pix, 2, "unproject");
  const py::ssize_t n = U.ndim() == 1 ? 1 : U.shape(0);
  py::array_t<double> rays({n, py::ssize_t{3}});
  py::array_t<bool> ok(n);
  const double* u = U.data();
  double* r = rays.mutable_data();
  bool* o = ok.mutable_data();
  {
    py::gil_scoped_release nogil;
    for (py::ssize_t i = 0; i < n; ++i) {
      Eigen::Vector3d ray;
      o[i] = m.unproject(Eigen::Vector2d(u[2 * i], u[2 * i + 1]), &ray);
      r[3 * i] = ray.x();
      r[3 * i + 1] = ray.y();
      r[3 * i + 2] = ray.z();
    }
  }
  return py::make_tuple(rays, ok);
}

}  // namespace

PYBIND11_MODULE(rigpy, mod) {
  mod.doc() = "physicalSetup rig bindings (T4): camera models, remap tables, policy image prep";

  // ------------------------------------------------------------------------------------------- camera_model
  auto cm = mod.def_submodule("camera_model", "rig::CameraModel and remap tables (rig/camera_model/*.hpp)");
  py::enum_<rig::CameraModelKind>(cm, "CameraModelKind")
      .value("PINHOLE_RADTAN", rig::CameraModelKind::kPinholeRadtan)
      .value("KANNALA_BRANDT4", rig::CameraModelKind::kKannalaBrandt4)
      .value("DOUBLE_SPHERE", rig::CameraModelKind::kDoubleSphere)
      .value("EUCM", rig::CameraModelKind::kEucm)
      .value("PINHOLE", rig::CameraModelKind::kVirtualPinhole);

  py::class_<rig::CameraModelParams>(cm, "CameraModelParams")
      .def(py::init<>())
      .def(py::init([](rig::CameraModelKind kind, int width, int height, double fx, double fy, double cx, double cy,
                       std::vector<double> distortion, double fov_max_deg) {
             rig::CameraModelParams p;
             p.kind = kind;
             p.width = width;
             p.height = height;
             p.fx = fx;
             p.fy = fy;
             p.cx = cx;
             p.cy = cy;
             p.distortion = std::move(distortion);
             p.fov_max_deg = fov_max_deg;
             return p;
           }),
           py::arg("kind"), py::arg("width"), py::arg("height"), py::arg("fx"), py::arg("fy"), py::arg("cx"),
           py::arg("cy"), py::arg("distortion") = std::vector<double>{}, py::arg("fov_max_deg") = 0.0)
      .def_readwrite("kind", &rig::CameraModelParams::kind)
      .def_readwrite("width", &rig::CameraModelParams::width)
      .def_readwrite("height", &rig::CameraModelParams::height)
      .def_readwrite("fx", &rig::CameraModelParams::fx)
      .def_readwrite("fy", &rig::CameraModelParams::fy)
      .def_readwrite("cx", &rig::CameraModelParams::cx)
      .def_readwrite("cy", &rig::CameraModelParams::cy)
      .def_readwrite("distortion", &rig::CameraModelParams::distortion)
      .def_readwrite("fov_max_deg", &rig::CameraModelParams::fov_max_deg)
      .def_readwrite("note", &rig::CameraModelParams::note);

  py::class_<rig::CameraModel, std::shared_ptr<rig::CameraModel>>(cm, "CameraModel")
      .def_property_readonly("kind", &rig::CameraModel::kind)
      .def_property_readonly("width", &rig::CameraModel::width)
      .def_property_readonly("height", &rig::CameraModel::height)
      .def_property_readonly("params", &rig::CameraModel::params)
      .def("project", &project, py::arg("points"), py::arg("jacobian") = false,
           "points (N,3) -> (uv (N,2), valid (N,)[, J (N,2,3)]); invalid rows are NaN")
      .def("unproject", &unproject, py::arg("uv"), "uv (N,2) -> (unit rays (N,3), valid (N,))")
      .def(
          "scaled",
          [](const rig::CameraModel& m, int w, int h) -> std::shared_ptr<rig::CameraModel> {
            auto s = m.scaled(w, h);
            if (!s) throw py::value_error("scaled: target size is not a uniform scale of the model size");
            return std::shared_ptr<rig::CameraModel>(std::move(s));
          },
          py::arg("width"), py::arg("height"))
      .def("in_image", [](const rig::CameraModel& m, double u, double v) { return m.in_image({u, v}); });

  py::class_<rig::VirtualPinhole, rig::CameraModel, std::shared_ptr<rig::VirtualPinhole>>(cm, "VirtualPinhole")
      .def_static(
          "create",
          [](const rig::CameraModelParams& p) { return std::shared_ptr<rig::VirtualPinhole>(unwrap(rig::VirtualPinhole::create(p))); },
          py::arg("params"))
      .def("K", &rig::VirtualPinhole::K);

  cm.def(
      "make_camera_model",
      [](const rig::CameraModelParams& p) -> std::shared_ptr<rig::CameraModel> {
        if (p.kind == rig::CameraModelKind::kVirtualPinhole) {
          return std::shared_ptr<rig::VirtualPinhole>(unwrap(rig::VirtualPinhole::create(p)));
        }
        return std::shared_ptr<rig::CameraModel>(unwrap(rig::make_camera_model(p)));
      },
      py::arg("params"), "Model for params.kind (a VirtualPinhole for kind PINHOLE)");

  py::class_<rig::RemapTable, std::shared_ptr<rig::RemapTable>>(cm, "RemapTable")
      .def_property_readonly("map1", [](const rig::RemapTable& t) { return view(t.map1); })
      .def_property_readonly("map2", [](const rig::RemapTable& t) { return view(t.map2); })
      .def_property_readonly("valid_mask",
                             [](const rig::RemapTable& t) { return view(t.valid_mask); })
      .def_property_readonly("src_size",
                             [](const rig::RemapTable& t) { return py::make_tuple(t.src_size.width, t.src_size.height); })
      .def_property_readonly("dst_size",
                             [](const rig::RemapTable& t) { return py::make_tuple(t.dst_size.width, t.dst_size.height); })
      .def_readonly("coverage", &rig::RemapTable::coverage)
      .def_readonly("fixed_point", &rig::RemapTable::fixed_point)
      .def(
          "apply",
          [](const rig::RemapTable& t, const py::array& src, int interpolation) {
            const cv::Mat s = as_mat(src);
            cv::Mat dst;
            rig::Status st;
            {
              py::gil_scoped_release nogil;
              st = t.apply(s, dst, interpolation);
            }
            if (!st.ok()) raise(st);
            return to_array(std::move(dst));
          },
          py::arg("src"), py::arg("interpolation") = static_cast<int>(cv::INTER_LINEAR),
          "physical image (src_size) -> virtual image (dst_size); uncovered pixels are 0");

  cm.def(
      "build_remap",
      [](const rig::CameraModel& phys, const rig::VirtualPinhole& virt, const Eigen::Matrix3d& R, bool fixed_point) {
        rig::Result<rig::RemapTable> r = rig::invalid_argument("unset");
        {
          py::gil_scoped_release nogil;
          r = rig::build_remap(phys, virt, R, fixed_point);
        }
        return std::make_shared<rig::RemapTable>(unwrap(std::move(r)));
      },
      py::arg("phys"), py::arg("virt"), py::arg("R_phys_virt") = Eigen::Matrix3d::Identity(),
      py::arg("fixed_point") = true);

  py::class_<rig::CoverageReport>(cm, "CoverageReport")
      .def_readonly("dst_width", &rig::CoverageReport::dst_width)
      .def_readonly("dst_height", &rig::CoverageReport::dst_height)
      .def_readonly("n_total", &rig::CoverageReport::n_total)
      .def_readonly("n_valid", &rig::CoverageReport::n_valid)
      .def_readonly("n_model_invalid", &rig::CoverageReport::n_model_invalid)
      .def_readonly("n_outside", &rig::CoverageReport::n_outside)
      .def_readonly("coverage", &rig::CoverageReport::coverage)
      .def_readonly("min_margin_px", &rig::CoverageReport::min_margin_px)
      .def_readonly("max_ray_angle_deg", &rig::CoverageReport::max_ray_angle_deg)
      .def_readonly("src_u_min", &rig::CoverageReport::src_u_min)
      .def_readonly("src_u_max", &rig::CoverageReport::src_u_max)
      .def_readonly("src_v_min", &rig::CoverageReport::src_v_min)
      .def_readonly("src_v_max", &rig::CoverageReport::src_v_max);

  cm.def(
      "coverage_report",
      [](const rig::CameraModel& phys, const rig::VirtualPinhole& virt, const Eigen::Matrix3d& R) {
        return unwrap(rig::coverage_report(phys, virt, R));
      },
      py::arg("phys"), py::arg("virt"), py::arg("R_phys_virt") = Eigen::Matrix3d::Identity());

  // ----------------------------------------------------------------------------------------------- imageprep
  auto ip = mod.def_submodule("imageprep", "Policy image prep (rig/imageprep/imageprep.hpp; iPhUMI cv_util.py:5-34)");
  ip.def(
      "letterbox_square_resize",
      [](const py::array& img, int out_size, bool swap_rb) {
        const cv::Mat src = as_mat(img);
        rig::Result<cv::Mat> r = rig::invalid_argument("unset");
        {
          py::gil_scoped_release nogil;
          r = rig::letterbox_square_resize(src, out_size, swap_rb);
        }
        return to_array(unwrap(std::move(r)));
      },
      py::arg("img"), py::arg("out_size") = 224, py::arg("swap_rb") = false,
      "Pad to a centred black square and resize with INTER_AREA; swap_rb = cv_util bgr_to_rgb");
  ip.def(
      "letterbox_geometry",
      [](int iw, int ih) {
        const auto g = rig::letterbox_geometry(iw, ih);
        py::dict d;
        d["size"] = g.size;
        d["top"] = g.top;
        d["bottom"] = g.bottom;
        d["left"] = g.left;
        d["right"] = g.right;
        return d;
      },
      py::arg("iw"), py::arg("ih"));
  ip.def(
      "remap_letterbox",
      [](const rig::RemapTable& t, const py::array& img, int out_size, bool swap_rb) {
        const cv::Mat src = as_mat(img);
        rig::Result<cv::Mat> r = rig::invalid_argument("unset");
        {
          py::gil_scoped_release nogil;
          cv::Mat virt;
          const rig::Status st = t.apply(src, virt);
          r = st.ok() ? rig::letterbox_square_resize(virt, out_size, swap_rb) : rig::Result<cv::Mat>(st);
        }
        return to_array(unwrap(std::move(r)));
      },
      py::arg("table"), py::arg("img"), py::arg("out_size") = 224, py::arg("swap_rb") = false,
      "ImagePrep PrepMode.kRemapLetterbox: table.apply then letterbox_square_resize");
}
