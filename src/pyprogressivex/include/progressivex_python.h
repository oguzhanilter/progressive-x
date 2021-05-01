#include <vector>
#include <string>

int find6DPoses_(
	const std::vector<double>& imagePoints,
	const std::vector<double>& worldPoints,
	const std::vector<double>& intrinsicParams,
	std::vector<size_t>& labeling,
	std::vector<double>& homographies,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number);

int findHomographies_(
		const std::vector<double>& sourcePoints,
		const std::vector<double>& destinationPoints,
		std::vector<size_t>& labeling,
		std::vector<double>& homographies,
		const double &spatial_coherence_weight,
		const double &threshold,
		const double &confidence,
		const double &neighborhood_ball_radius,
		const double &maximum_tanimoto_similarity,
		const size_t &max_iters,
		const size_t &minimum_point_number,
		const int &maximum_model_number);
		
int findTwoViewMotions_(
		const std::vector<double>& sourcePoints,
		const std::vector<double>& destinationPoints,
		std::vector<size_t>& labeling,
		std::vector<double>& homographies,
		const double &spatial_coherence_weight,
		const double &threshold,
		const double &confidence,
		const double &neighborhood_ball_radius,
		const double &maximum_tanimoto_similarity,
		const size_t &max_iters,
		const size_t &minimum_point_number,
		const int &maximum_model_number);

void MultiPlaneFitting_(
	const std::string& input_path_, // The path of the detected correspondences
	const std::string& output_path_, // The path of the detected correspondences
	const bool oriented_points_,
	const size_t ransac_iteration_number_,
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double cell_number_, // The radius of the neighborhood ball for determining the neighborhoods.
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const size_t minimum_point_number_,// The minimum number of inlier for a model to be kept.
	const size_t _SkipPoints); 