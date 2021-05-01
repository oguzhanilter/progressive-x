#include "progressivex_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include "progx_utils.h"
#include "utils.h"
#include "GCoptimization.h"
#include "grid_neighborhood_graph.h"
#include "flann_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "essential_estimator.h"

#include "progressive_x.h"

#include <ctime>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <direct.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>
#include <glog/logging.h>

int find6DPoses_(
	const std::vector<double>& imagePoints,
	const std::vector<double>& worldPoints,
	const std::vector<double>& intrinsicParams,
	std::vector<size_t>& labeling,
	std::vector<double>& poses,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	// Calculate the inverse of the intrinsic camera parameters
	Eigen::Matrix3d K;
	K << intrinsicParams[0], intrinsicParams[1], intrinsicParams[2],
		intrinsicParams[3], intrinsicParams[4], intrinsicParams[5],
		intrinsicParams[6], intrinsicParams[7], intrinsicParams[8];
	const Eigen::Matrix3d Kinv =
		K.inverse();
	
	Eigen::Vector3d vec;
	vec(2) = 1;
	size_t num_tents = imagePoints.size() / 2;
	cv::Mat points(num_tents, 5, CV_64F);
	cv::Mat normalized_points(num_tents, 5, CV_64F);
	size_t iterations = 0;
	for (size_t i = 0; i < num_tents; ++i) {
		vec(0) = imagePoints[2 * i];
		vec(1) = imagePoints[2 * i + 1];
		
		points.at<double>(i, 0) = imagePoints[2 * i];
		points.at<double>(i, 1) = imagePoints[2 * i + 1];
		points.at<double>(i, 2) = worldPoints[3 * i];
		points.at<double>(i, 3) = worldPoints[3 * i + 1];
		points.at<double>(i, 4) = worldPoints[3 * i + 2];
		
		normalized_points.at<double>(i, 0) = Kinv.row(0) * vec;
		normalized_points.at<double>(i, 1) = Kinv.row(1) * vec;
		normalized_points.at<double>(i, 2) = worldPoints[3 * i];
		normalized_points.at<double>(i, 3) = worldPoints[3 * i + 1];
		normalized_points.at<double>(i, 4) = worldPoints[3 * i + 2];
	}
	
	// Normalize the threshold
	const double f = 0.5 * (K(0,0) + K(1,1));
	const double normalized_threshold =
		threshold / f;
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::UniformSampler main_sampler(&points);

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultPnPEstimator, // The type of the used model estimator
		gcransac::sampler::UniformSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = normalized_threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(normalized_points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	poses.reserve(12 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		poses.emplace_back(model.descriptor(0, 0));
		poses.emplace_back(model.descriptor(0, 1));
		poses.emplace_back(model.descriptor(0, 2));
		poses.emplace_back(model.descriptor(0, 3));
		poses.emplace_back(model.descriptor(1, 0));
		poses.emplace_back(model.descriptor(1, 1));
		poses.emplace_back(model.descriptor(1, 2));
		poses.emplace_back(model.descriptor(1, 3));
		poses.emplace_back(model.descriptor(2, 0));
		poses.emplace_back(model.descriptor(2, 1));
		poses.emplace_back(model.descriptor(2, 2));
		poses.emplace_back(model.descriptor(2, 3));
	}
	
	return progressive_x.getModelNumber();
}

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
	const int &maximum_model_number)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	const size_t num_tents = sourcePoints.size() / 2;
	
	double max_x = std::numeric_limits<double>::min(),
		min_x =  std::numeric_limits<double>::max(),
		max_y = std::numeric_limits<double>::min(),
		min_y =  std::numeric_limits<double>::max();
		
	cv::Mat points(num_tents, 4, CV_64F);
	for (size_t i = 0; i < num_tents; ++i) {
		
		const double 
			&x1 = sourcePoints[2 * i],
			&y1 = sourcePoints[2 * i + 1],
			&x2 = destinationPoints[2 * i],
			&y2 = destinationPoints[2 * i + 1];
		
		max_x = MAX(max_x, x1);
		min_x = MIN(min_x, x1);
		max_x = MAX(max_x, x2);
		min_x = MIN(min_x, x2);
		
		max_y = MAX(max_y, y1);
		min_y = MIN(min_y, y1);
		max_y = MAX(max_y, y2);
		min_y = MIN(min_y, y2);
		
		points.at<double>(i, 0) = x1;
		points.at<double>(i, 1) = y1;
		points.at<double>(i, 2) = x2;
		points.at<double>(i, 3) = y2;
	}
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::ProgressiveNapsacSampler<4> main_sampler(&points, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		gcransac::utils::DefaultHomographyEstimator::sampleSize(), // The size of a minimal sample
		{ max_x + std::numeric_limits<double>::epsilon(), // The width of the source image
			max_y + std::numeric_limits<double>::epsilon(), // The height of the source image
			max_x + std::numeric_limits<double>::epsilon(), // The width of the destination image
			max_y + std::numeric_limits<double>::epsilon() }); // The height of the destination image

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultHomographyEstimator, // The type of the used model estimator
		gcransac::sampler::ProgressiveNapsacSampler<4>, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	homographies.reserve(9 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		homographies.emplace_back(model.descriptor(0, 0));
		homographies.emplace_back(model.descriptor(0, 1));
		homographies.emplace_back(model.descriptor(0, 2));
		homographies.emplace_back(model.descriptor(1, 0));
		homographies.emplace_back(model.descriptor(1, 1));
		homographies.emplace_back(model.descriptor(1, 2));
		homographies.emplace_back(model.descriptor(2, 0));
		homographies.emplace_back(model.descriptor(2, 1));
		homographies.emplace_back(model.descriptor(2, 2));
	}
	
	return progressive_x.getModelNumber();
}

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
	const int &maximum_model_number)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	const size_t num_tents = sourcePoints.size() / 2;
	
	double max_x = std::numeric_limits<double>::min(),
		min_x =  std::numeric_limits<double>::max(),
		max_y = std::numeric_limits<double>::min(),
		min_y =  std::numeric_limits<double>::max();
		
	cv::Mat points(num_tents, 4, CV_64F);
	for (size_t i = 0; i < num_tents; ++i) {
		
		const double 
			&x1 = sourcePoints[2 * i],
			&y1 = sourcePoints[2 * i + 1],
			&x2 = destinationPoints[2 * i],
			&y2 = destinationPoints[2 * i + 1];
		
		max_x = MAX(max_x, x1);
		min_x = MIN(min_x, x1);
		max_x = MAX(max_x, x2);
		min_x = MIN(min_x, x2);
		
		max_y = MAX(max_y, y1);
		min_y = MIN(min_y, y1);
		max_y = MAX(max_y, y2);
		min_y = MIN(min_y, y2);
		
		points.at<double>(i, 0) = x1;
		points.at<double>(i, 1) = y1;
		points.at<double>(i, 2) = x2;
		points.at<double>(i, 3) = y2;
	}
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::ProgressiveNapsacSampler<4> main_sampler(&points, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		gcransac::utils::DefaultFundamentalMatrixEstimator::sampleSize(), // The size of a minimal sample
		{max_x + std::numeric_limits<double>::epsilon(), // The width of the source image
			max_y + std::numeric_limits<double>::epsilon(), // The height of the source image
			max_x + std::numeric_limits<double>::epsilon(), // The width of the destination image
			max_y + std::numeric_limits<double>::epsilon() }); // The height of the destination image

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultFundamentalMatrixEstimator, // The type of the used model estimator
		gcransac::sampler::ProgressiveNapsacSampler<4>, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	homographies.reserve(9 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		homographies.emplace_back(model.descriptor(0, 0));
		homographies.emplace_back(model.descriptor(0, 1));
		homographies.emplace_back(model.descriptor(0, 2));
		homographies.emplace_back(model.descriptor(1, 0));
		homographies.emplace_back(model.descriptor(1, 1));
		homographies.emplace_back(model.descriptor(1, 2));
		homographies.emplace_back(model.descriptor(2, 0));
		homographies.emplace_back(model.descriptor(2, 1));
		homographies.emplace_back(model.descriptor(2, 2));
	}
	
	return progressive_x.getModelNumber();
}


template<size_t _SkipPoints>
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
	const size_t minimum_point_number_) // The minimum number of inlier for a model to be kept.
{

	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}


	// Loading oriented or non-oriented 3D points from file
	cv::Mat points;
	if (oriented_points_) // Load an oriented point cloud. In this case, each row in the file contains 6 values
		gcransac::utils::loadPointsFromFile<6, _SkipPoints, false>(points, input_path_.c_str());
	else // Load an unoriented point cloud. In this case, each row in the file contains 4 values
		gcransac::utils::loadPointsFromFile<3, _SkipPoints, false>(points, input_path_.c_str());
	
	// The number of loaded points
	size_t pointNumber = points.rows;
	if (pointNumber == 0) // If no points are loaded, return.
	{
		LOG(ERROR) << "No points are loaded.";
		return;
	}

	// Filter points if they are [0, 0, 0]
	cv::Mat filteredPoints; // The kept points
	std::vector<size_t> keptPointIndices; // The indices of the kept points
	keptPointIndices.reserve(pointNumber); // Occupy the maximum memory 
	// Store the bounding box of the kept points. This will be important for the neighborhood structure
	std::vector<double> boundingBox = { std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
		std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), 
		std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest() };
	// Iterate through all points and check if their coordinates are [0,0,0]
	for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
	{
		const double& x = points.at<double>(pointIdx, 0), // The x coordinate
			& y = points.at<double>(pointIdx, 1), // The y coordinate
			& z = points.at<double>(pointIdx, 2); // The z coordinate
		// Check if the sum of the coordinates is zero.
		// If not, keep the point.
		if (std::abs(x) + std::abs(y) + std::abs(z) > std::numeric_limits<double>::epsilon())
		{
			// Store the index of the points
			keptPointIndices.emplace_back(pointIdx);
			
			// Updating the bounding box
			boundingBox[0] = MIN(boundingBox[0], x);
			boundingBox[1] = MAX(boundingBox[1], x);
			boundingBox[2] = MIN(boundingBox[2], y);
			boundingBox[3] = MAX(boundingBox[3], y);
			boundingBox[4] = MIN(boundingBox[4], z);
			boundingBox[5] = MAX(boundingBox[5], z);
		}
	}

	// Calculating the dimensions of the bounding box along each axis
	const double kSizeX = boundingBox[1] - boundingBox[0], // The dimension along axis X
		kSizeY = boundingBox[3] - boundingBox[2], // The dimension along axis Y
		kSizeZ = boundingBox[5] - boundingBox[4]; // The dimension along axis Z

	// The number of kept point
	pointNumber = keptPointIndices.size();
	// Create a matrix where the kept points will be stored
	filteredPoints.create(pointNumber, points.cols, CV_64F);
	// Copying the coordinates of the kept points
	for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
	{
		// The index of the current point
		const size_t& idx = 
			keptPointIndices[pointIdx];
		// Copy the coordinates of the kept points
		points.row(idx).copyTo(filteredPoints.row(pointIdx));
	}

	cv::Mat onlyCoordinates(pointNumber, 3, CV_64F);
	filteredPoints(cv::Rect(0, 0, 3, filteredPoints.rows)).copyTo(onlyCoordinates);

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::GridNeighborhoodGraph<3> neighborhood(&onlyCoordinates, // All data points
		{ kSizeX / cell_number_,
			kSizeY / cell_number_,
			kSizeZ / cell_number_ },
		cell_number_); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::UniformSampler main_sampler(&points);

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::GridNeighborhoodGraph<3>, // The type of the used neighborhood-graph
		gcransac::utils::Default3DPlaneEstimator, // The type of the used model estimator
		gcransac::sampler::UniformSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings& settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number_;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = inlier_outlier_threshold_;
	// The required confidence in the results
	settings.setConfidence(confidence_);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity_;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight_;
	// The maximum iteration number of GC-RANSAC
	settings.proposal_engine_settings.max_iteration_number = ransac_iteration_number_;

	progressive_x.run(filteredPoints, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC

	printf("Processing time = %f secs.\n", progressive_x.getStatistics().processing_time);
	printf("Number of found model instances = %d.\n", progressive_x.getModelNumber());

	// Store the inliers of the current model to the statistics object
	const auto& labeling = progressive_x.getStatistics().labeling; // The labeling
	std::vector<std::vector<size_t>> modelsInliers(progressive_x.getModelNumber()); // The inliers of each model
	for (auto& inlierContainer : modelsInliers) // Occupying the memory for the points
		inlierContainer.reserve(pointNumber);

	// Finding the indices of the points assigned to each particular model
	for (size_t pointIdx = 0; pointIdx < labeling.size(); ++pointIdx)
	{
		// The label of the points
		const auto& label = labeling[pointIdx];
		// If the point is not assigned to the outlier class, store its index
		if (label < progressive_x.getModelNumber())
			modelsInliers[label].emplace_back(pointIdx);
	}

	// Calculating the total number of inliers assigned to some model
	size_t totalInlierNumber = 0;
	for (const auto inlierContainer : modelsInliers)
		totalInlierNumber += inlierContainer.size();

	// Save the point cloud so it can be visualized, e.g., in Meshlab 
	cv::Mat coloredPoints(totalInlierNumber, filteredPoints.cols + 4, filteredPoints.type());
	size_t rowIdx = 0;

	// Copying the coordinates of the points
	const auto& models = progressive_x.getModels();
	for (size_t modelIdx = 0; modelIdx < models.size(); ++modelIdx)
	{
		// The inliers of the current model
		const auto& inlierContainer = modelsInliers[modelIdx];
		// The parameters of the current model
		const auto& currentModel = models[modelIdx];
		// The normal of the current model
		const Eigen::Vector3d &normal = 
			currentModel.descriptor.block<3, 1>(0, 0);

		// Generate a random color that can be visualized
		const double r = 255.0 * static_cast<double>(rand()) / static_cast<double>(RAND_MAX), 
			g = 255.0 * static_cast<double>(rand()) / static_cast<double>(RAND_MAX),
			b = 255.0 * static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

		// Save each inlier
		for (const auto& pointIdx : inlierContainer)
		{
			size_t col;
			// Save the coordinates
			for (col = 0; col < 3; ++col)
				coloredPoints.at<double>(rowIdx, col) = filteredPoints.at<double>(pointIdx, col);
			// Save the normal
			for (col = 3; col < 6; ++col)
				coloredPoints.at<double>(rowIdx, col) = normal(col - 3);
			// Save the color
			coloredPoints.at<double>(rowIdx, col++) = r;
			coloredPoints.at<double>(rowIdx, col++) = g;
			coloredPoints.at<double>(rowIdx, col++) = b;
			// Add a reflectance parameter so it can be easily loaded into Meshlab
			coloredPoints.at<double>(rowIdx, col) = 1.0;
			// Increase the row index
			++rowIdx;
			// nice
		}
	}

	// Save the points to file
	gcransac::utils::savePointsToFile(coloredPoints,
		output_path_.c_str());
}