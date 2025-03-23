import matplotlib.pyplot as plt
import numpy as np
import yaml
from utils import stereo_depth, decomposition, feature_extractor, feature_matching, motion_estimation
from mpl_toolkits.mplot3d import Axes3D
import gtsam
#import gtsam_unstable
#from gtsam_unstable import BatchFixedLagSmoother#,FixedLagSmootherParams
from gtsam import (
    symbol, Pose3, Point3, Cal3_S2, GenericProjectionFactorCal3_S2,
    PriorFactorPose3, LevenbergMarquardtOptimizer,Cal3_S2,BetweenFactorPose3, 
    Values, NonlinearFactorGraph, LevenbergMarquardtParams
)

#from gtsam.noiseModel import Diagonal

# Load Config File
with open("../config/initial_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        print(error)

# Declare Necessary Variables
detector_name = config['parameters']['detector']
subset = config['parameters']['subset']
threshold = config['parameters']['distance_threshold']

sigmas_meas = np.array([1.0,1.0])
sigmas_pose = np.array([0.1,0.1,0.1,0.1,0.1,0.1])

def run_bundle_adjustment(window_poses, window_landmarks, window_observations, K):
    """    
    Perform sliding window bundle adjustment using GTSAM with robust loss.
    
    Parameters:
      window_poses: list of 4x4 numpy arrays (poses for frames in the sliding window)
      window_landmarks: dict mapping landmark id -> 3D position (np.array shape (3,))
      window_observations: dict mapping landmark id -> list of observations,
                           each observation is a tuple (frame_index, [u, v])
      K: Camera intrinsic matrix
      
    Returns:
      optimized_poses: list of 4x4 numpy arrays (optimized poses)
      optimized_landmarks: dict mapping landmark id -> optimized 3D point (np.array)
    """
    graph = NonlinearFactorGraph()
    initial_estimate = Values()
    
    # Camera calibration and base noise models.
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    calibration = Cal3_S2(fx, fy, 0, cx, cy)
    
    # Base noise models
    base_meas_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas_meas)  # e.g. 1 pixel std dev.
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas_pose)         # Pose prior noise
    
    # Wrap the measurement noise in a robust loss (Huber) to mitigate outliers.
    robust_meas_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber(1.345), base_meas_noise)
    
    num_frames = len(window_poses)
    # Add pose variables to the graph.
    for i, pose in enumerate(window_poses):
        R = gtsam.Rot3(pose[:3, :3])
        t = pose[:3, 3].reshape((3, 1))
        pose3 = Pose3(R, t)
        key = symbol('x', i)
        initial_estimate.insert(key, pose3)
        # Adding a prior on each pose (even a weak one) can help anchor the optimization.
        graph.add(PriorFactorPose3(key, pose3, pose_noise))
    
    # Filter out landmarks that do not have any observation in the sliding window.
    valid_landmarks = {}
    for lm_id, pt in window_landmarks.items():
        if lm_id in window_observations and len(window_observations[lm_id]) > 0:
            valid_landmarks[lm_id] = pt

    # Add only valid landmark variables to the graph.
    for lm_id, pt in valid_landmarks.items():
        lm_key = symbol('l', lm_id)
        initial_estimate.insert(lm_key, Point3(pt))
    
    # Add projection factors for each observation (using robust noise).
    for lm_id, obs_list in window_observations.items():
        # Only proceed if this landmark is valid.
        if lm_id not in valid_landmarks:
            continue
        for (frame_idx, measurement) in obs_list:
            pose_key = symbol('x', frame_idx)
            if not initial_estimate.exists(pose_key):
                continue  # Skip if the pose is not in the sliding window.
            factor = GenericProjectionFactorCal3_S2(
                gtsam.Point2(measurement[0], measurement[1]),
                robust_meas_noise,
                pose_key,
                symbol('l', lm_id),
                calibration
            )
            graph.add(factor)
    
    # Optimize using Levenberg-Marquardt.
    params = LevenbergMarquardtParams()
    # Optionally, you can set a maximum number of iterations or tweak other parameters:
    params.setMaxIterations(100)
    optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    # Extract optimized poses.
    optimized_poses = []
    for i in range(num_frames):
        pose_opt = result.atPose3(symbol('x', i))
        T = np.eye(4)
        T[:3, :3] = pose_opt.rotation().matrix()
        T[:3, 3] = pose_opt.translation()
        optimized_poses.append(T)
    
    # Extract optimized landmarks.
    optimized_landmarks = {}
    for lm_id in valid_landmarks.keys():
        pt_opt = result.atPoint3(symbol('l', lm_id))
        optimized_landmarks[lm_id] = np.asarray(pt_opt)
    
    return optimized_poses, optimized_landmarks



# ------------------------------------------------------------------------------
# Visual Odometry with Sliding Window Bundle Adjustment
# ------------------------------------------------------------------------------
def visual_odometry(data_handler, detector=detector_name, mask=None, subset=None, plot=True):
    """    
    Compute the visual odometry using all components and incorporate sliding
    window bundle adjustment (BA) with GTSAM.
    
    Assumes the existence of functions:
      - stereo_depth(image_left, image_right, P0, P1)
      - feature_extractor(image, detector, mask)
      - feature_matching(desc1, desc2, detector, distance_threshold)
      - motion_estimation(matches, keypoints1, keypoints2, intrinsic_matrix, depth)
      - decomposition(P) -> (intrinsic, R, t)
      
    And that a global variable 'threshold' is defined.
    """
    if subset is not None:
        num_frames = subset
    else:
        num_frames = data_handler.frames
    plt.ion()
    
    if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = data_handler.ground_truth[:, 0, 3]
        ys = data_handler.ground_truth[:, 1, 3]
        zs = data_handler.ground_truth[:, 2, 3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='dimgray')
        ax.set_title("Ground Truth vs Estimated Trajectory")
    
    homo_matrix = np.eye(4)
    #homo_matrix = data_handler.ground_truth[0]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = homo_matrix[:3, :]
    
    # Get left camera's intrinsic matrix from the projection matrix
    left_intrinsic_matrix, _, _ = decomposition(data_handler.P0)
    
    if data_handler.low_memory:
        data_handler.reset_frames()
        next_image = next(data_handler.left_images)
    
    # --- Sliding Window Initialization ---
    window_size = 5  # Adjust as needed
    sliding_window_poses = []           # List of poses in the sliding window
    sliding_window_landmarks = {}       # {landmark_id: 3D point (world coords)}
    sliding_window_observations = {}    # {landmark_id: list of (frame_idx_in_window, [u,v])}
    landmark_counter = 0                # Unique landmark id counter
    
    # Data association for the previous frame:
    # Maps keypoint index in the previous frame to landmark id.
    prev_frame_landmark_ids = {}
    
    # Process the first frame to initialize landmarks
    if data_handler.low_memory:
        image_left = next_image
        image_right = next(data_handler.right_images)
        next_image = next(data_handler.left_images)
    else:
        image_left = data_handler.left_images[0]
        image_right = data_handler.right_images[0]
    
    depth = stereo_depth(image_left, image_right, P0=data_handler.P0, P1=data_handler.P1)
    keypoints_left, descriptors_left = feature_extractor(image_left, detector, mask)
    
    fx, fy = left_intrinsic_matrix[0, 0], left_intrinsic_matrix[1, 1]
    cx, cy = left_intrinsic_matrix[0, 2], left_intrinsic_matrix[1, 2]
    
    for idx, kp in enumerate(keypoints_left):
        u, v = kp.pt
        # Get depth value at the keypoint location (ensure valid indices)
        d = depth[int(round(v)), int(round(u))]
        if d <= 0:
            continue
        # Backproject to 3D (in camera frame)
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d
        point_cam = np.array([X, Y, Z])
        # Transform to world coordinates using current pose
        point_world = (homo_matrix[:3, :3] @ point_cam) + homo_matrix[:3, 3] #basically doing R*pointcam + t 
        sliding_window_landmarks[landmark_counter] = point_world
        sliding_window_observations.setdefault(landmark_counter, []).append((0, np.array([u, v])))
        prev_frame_landmark_ids[idx] = landmark_counter
        landmark_counter += 1
    
    sliding_window_poses.append(homo_matrix.copy())
    
    # --- Main Visual Odometry Loop ---
    for i in range(num_frames - 1):
        if data_handler.low_memory:
            image_left = next_image
            image_right = next(data_handler.right_images)
            next_image = next(data_handler.left_images)
        else:
            image_left = data_handler.left_images[i]
            image_right = data_handler.right_images[i]
            next_image = data_handler.left_images[i+1]
        
        depth = stereo_depth(image_left, image_right, P0=data_handler.P0, P1=data_handler.P1)
        keypoints_left_curr, descriptors_left_curr = feature_extractor(image_left, detector, mask)
        keypoints_left_next, descriptors_left_next = feature_extractor(next_image, detector, mask)
        
        matches = feature_matching(descriptors_left_curr, descriptors_left_next,
                                   detector=detector, distance_threshold=threshold)
        
        # Estimate relative motion between frames
        rotation_matrix, translation_vector, _, _, status = motion_estimation(
            matches, keypoints_left_curr, keypoints_left_next, left_intrinsic_matrix, depth)
        
        Transformation_matrix = np.eye(4)
        if status > 0:
            Transformation_matrix[:3, :3] = rotation_matrix
            Transformation_matrix[:3, 3] = translation_vector.T
            homo_matrix = homo_matrix.dot(np.linalg.inv(Transformation_matrix))
            trajectory[i+1, :, :] = homo_matrix[:3, :]
            
            # Update sliding window poses
            sliding_window_poses.append(homo_matrix.copy())
            if len(sliding_window_poses) > window_size:
                sliding_window_poses.pop(0)
                # Adjust observation frame indices for the sliding window
                for lm_id in sliding_window_observations:
                    updated_obs = [(frame_idx - 1, meas) 
                                   for (frame_idx, meas) in sliding_window_observations[lm_id]
                                   if frame_idx - 1 >= 0]
                    sliding_window_observations[lm_id] = updated_obs
            
            
            for lm_id, obs_list in sliding_window_observations.items():
                for (frame_idx, meas) in obs_list:
                    if frame_idx >= len(sliding_window_poses):
                        raise ValueError(f"Observation for landmark {lm_id} has invalid frame index {frame_idx} (window size {len(sliding_window_poses)})")


            # --- Data Association ---
            new_frame_landmark_ids = {}
            for match in matches:
                query_idx = match.queryIdx
                train_idx = match.trainIdx
                kp_next = keypoints_left_next[train_idx]
                u, v = kp_next.pt
                if query_idx in prev_frame_landmark_ids:
                    # Existing landmark: add new observation
                    lm_id = prev_frame_landmark_ids[query_idx]
                    sliding_window_observations.setdefault(lm_id, []).append(
                        (len(sliding_window_poses)-1, np.array([u, v]))
                    )
                    new_frame_landmark_ids[train_idx] = lm_id
                else:
                    # New landmark: initialize using depth from current frame
                    d = depth[int(round(v)), int(round(u))]
                    if d <= 0:
                        continue
                    X = (u - cx) * d / fx
                    Y = (v - cy) * d / fy
                    Z = d
                    point_cam = np.array([X, Y, Z])
                    point_world = (homo_matrix[:3, :3] @ point_cam) + homo_matrix[:3, 3]
                    sliding_window_landmarks[landmark_counter] = point_world
                    sliding_window_observations.setdefault(landmark_counter, []).append(
                        (len(sliding_window_poses)-1, np.array([u, v]))
                    )
                    new_frame_landmark_ids[train_idx] = landmark_counter
                    landmark_counter += 1
            
            prev_frame_landmark_ids = new_frame_landmark_ids.copy()
            
            # --- Run Bundle Adjustment ---
            if len(sliding_window_poses) == window_size:

                optimized_poses, optimized_landmarks = run_bundle_adjustment(
                    sliding_window_poses, sliding_window_landmarks,
                    sliding_window_observations, left_intrinsic_matrix)
                sliding_window_poses = optimized_poses
                sliding_window_landmarks = optimized_landmarks
                # Update the global trajectory with the latest optimized pose
                trajectory[i+1, :, :] = sliding_window_poses[-1][:3, :]
            
            if i % 10 == 0:
                print(f'{i} frames have been computed')
            if i == num_frames - 2:
                print('All frames have been computed')
            
            if plot:
                xs = trajectory[:i+2, 0, 3]
                ys = trajectory[:i+2, 1, 3]
                zs = trajectory[:i+2, 2, 3]
                plt.plot(xs, ys, zs, c='darkorange')
                plt.pause(1e-32)
    
    if plot:
        plt.show(block=False)
    return trajectory
