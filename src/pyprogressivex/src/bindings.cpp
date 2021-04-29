#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "progressivex_python.h"

namespace py = pybind11;

py::tuple find6DPoses(
	py::array_t<double>  x1y1_,
	py::array_t<double>  x2y2z2_,
	py::array_t<double>  K_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number) 
{
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=3");
	}
	if (NUM_TENTS < 3) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=3");
	}
	
	py::buffer_info buf1a = x2y2z2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 3) {
		throw std::invalid_argument("x2y2z2 should be an array with dims [n,3], n>=3");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2z2 should be the same size");
	}
	
	py::buffer_info buf1K = K_.request();
	size_t DIMK1 = buf1K.shape[0];
	size_t DIMK2 = buf1K.shape[1];

	if (DIMK1 != 3 || DIMK2 != 3) {
		throw std::invalid_argument("K should be an array with dims [3,3]");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2z2;
	x2y2z2.assign(ptr1a, ptr1a + buf1a.size);

	double *ptr1K = (double *)buf1K.ptr;
	std::vector<double> K;
	K.assign(ptr1K, ptr1K + buf1K.size);
	
	std::vector<double> poses;
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = find6DPoses_(
		x1y1,
		x2y2z2,
		K,
		labeling,
		poses,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number);

	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> poses_ = py::array_t<double>({ static_cast<size_t>(num_models) * 3, 4 });
	py::buffer_info buf2 = poses_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 12 * num_models; i++)
		ptr2[i] = poses[i];
	return py::make_tuple(poses_, labeling_);
}

py::tuple findHomographies(
	py::array_t<double>  x1y1_,
	py::array_t<double>  x2y2_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number) {
		
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=4");
	}
	if (NUM_TENTS < 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=4");
	}
	py::buffer_info buf1a = x2y2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 2) {
		throw std::invalid_argument("x2y2 should be an array with dims [n,2], n>=4");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2 should be the same size");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2;
	x2y2.assign(ptr1a, ptr1a + buf1a.size);
	std::vector<double> homographies;
	
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = findHomographies_(
		x1y1,
		x2y2,
		labeling,
		homographies,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number);
		
	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> homographies_ = py::array_t<double>({ static_cast<size_t>(num_models) * 3, 3 });
	py::buffer_info buf2 = homographies_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 9 * num_models; i++)
		ptr2[i] = homographies[i];
	return py::make_tuple(homographies_, labeling_);
}



py::tuple findTwoViewMotions(
	py::array_t<double>  x1y1_,
	py::array_t<double>  x2y2_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double maximum_tanimoto_similarity,
	int max_iters,
	int minimum_point_number,
	int maximum_model_number) 
{		
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=4");
	}
	if (NUM_TENTS < 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=4");
	}
	py::buffer_info buf1a = x2y2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 2) {
		throw std::invalid_argument("x2y2 should be an array with dims [n,2], n>=4");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2 should be the same size");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2;
	x2y2.assign(ptr1a, ptr1a + buf1a.size);
	std::vector<double> motions;
	
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = findTwoViewMotions_(
		x1y1,
		x2y2,
		labeling,
		motions,
		spatial_coherence_weight,
		threshold,
		conf,
		neighborhood_ball_radius,
		maximum_tanimoto_similarity,
		max_iters,
		minimum_point_number,
		maximum_model_number);
		
	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> motions_ = py::array_t<double>({ static_cast<size_t>(num_models) * 3, 3 });
	py::buffer_info buf2 = motions_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 9 * num_models; i++)
		ptr2[i] = motions[i];
	return py::make_tuple(motions_, labeling_);
}

template<int _SkipPoints>
py::bool_ MutltiPlaneFitting(	
	std::string input_path_, // The path of the detected correspondences
	std::string output_path_, // The path of the detected correspondences
	bool oriented_points_,
	int ransac_iteration_number_,
	double confidence_, // The RANSAC confidence value
	double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	double cell_number_, // The radius of the neighborhood ball for determining the neighborhoods.
	double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	int minimum_point_number

){
	
	MultiPlaneFitting_( input_path_, output_path_, oriented_points_,
						ransac_iteration_number_, confidence_,
						inlier_outlier_threshold_, spatial_coherence_weight_,
						cell_number_, maximum_tanimoto_similarity_, 
						minimum_point_number)

	return true;
}

PYBIND11_PLUGIN(pyprogressivex) {
                                                                             
    py::module m("pyprogressivex", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pyprogressivex
        .. autosummary::
           :toctree: _generate
           
           find6DPoses,
           findHomographies,
           findTwoViewMotions,
		   findMultiplePlanes,

    )doc");

	m.def("findMultiplePlanes", &MutltiPlaneFitting, R"doc(some doc)doc",
		py::arg("input_path"),
		py::arg("output_path"),
		py::arg("oriented_points") = true,
		py::arg("ransac_iteration_number") = 250,
		py::arg("confidence") = 0.99,
		py::arg("inlier_outlier_threshold") = 10.0,
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("cell_number") = 25,
		py::arg("maximum_tanimoto_similarity") = 0.9,
		py::arg("minimum_point_number") = 100);

	m.def("findHomographies", &findHomographies, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2"),
		py::arg("threshold") = 3.0,
		py::arg("conf") = 0.90,
		py::arg("spatial_coherence_weight") = 0.1,
		py::arg("neighborhood_ball_radius") = 20.0,
		py::arg("maximum_tanimoto_similarity") = 0.9,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 2 * 4,
		py::arg("maximum_model_number") = -1);

	m.def("findTwoViewMotions", &findTwoViewMotions, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2"),
		py::arg("threshold") = 0.75,
		py::arg("conf") = 0.90,
		py::arg("spatial_coherence_weight") = 0.1,
		py::arg("neighborhood_ball_radius") = 20.0,
		py::arg("maximum_tanimoto_similarity") = 0.9,
		py::arg("max_iters") = 1000,
		py::arg("minimum_point_number") = 2 * 7,
		py::arg("maximum_model_number") = -1);
		
	m.def("find6DPoses", &find6DPoses, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2z2"),
		py::arg("K"),
		py::arg("threshold") = 4.0,
		py::arg("conf") = 0.90,
		py::arg("spatial_coherence_weight") = 0.1,
		py::arg("neighborhood_ball_radius") = 20.0,
		py::arg("maximum_tanimoto_similarity") = 0.9,
		py::arg("max_iters") = 400,
		py::arg("minimum_point_number") = 2 * 3,
		py::arg("maximum_model_number") = -1);

  return m.ptr();
}
