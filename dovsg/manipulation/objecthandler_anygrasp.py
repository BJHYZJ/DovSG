import numpy as np
import open3d as o3d
from PIL import Image
from omegaconf import OmegaConf
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from pathlib import Path
from IPython import embed
import cv2
from dovsg.utils.utils import anygrasp_checkpoint_path
from dovsg.perception.models.mygroundingdinosam2 import MyGroundingDINOSAM2
import copy
from PIL import ImageDraw, Image
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict

config_path = "dovsg/manipulation/configs/objecthandler.yml"
cfgs = OmegaConf.load(config_path)
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
cfgs.checkpoint_path = anygrasp_checkpoint_path

class ObjectHandler():
    def __init__(
        self,
        box_threshold: float,
        text_threshold: float,
        device: str="cuda",
        is_visualize: bool=False,
        robot_min_height = -0.32,
        robot_max_height = 0.7,
        rot_cost_rate = 0.70,
        gripper_length=0.172
    ):
        self.is_visualize = is_visualize
        self.robot_min_height = robot_min_height
        self.robot_max_height = robot_max_height
        self.rot_cost_rate = rot_cost_rate
        self.gripper_length = gripper_length

        self.grasping_model = AnyGrasp(cfgs)
        self.grasping_model.load_net()
        # ### Initialize the GroundingDINO SAM2 model ###
        self.mygroundingdino_sam2 = MyGroundingDINOSAM2(
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )

        self.gripper_initial_pose = np.array([
            [1,  0,  0, 0.25287],
            [0, -1,  0, 0.005279],
            [0,  0, -1, 0.31426],
            [0,  0,  0, 1],

        ])

    def process_observations(self, observations, query):
        if isinstance(query, str):
            classes = [f"{query}"]
        elif isinstance(query, list):
            classes = query
        else:
            raise TypeError
        
        observations_new = {}
        for name, obs in observations.items():
            image = (obs["rgb"] * 255).astype(np.uint8)
            # Image.fromarray(image).show()
            detections = self.mygroundingdino_sam2.run(
                image=image,
                classes=classes
            )
            if len(detections.class_id) > 0:
                if isinstance(query, list) and 1 not in detections.class_id:
                    print("Just has noise object been catch!")
                    continue

                # print(len(detections))
                if self.is_visualize and True:
                    annotated_image, labels = self.mygroundingdino_sam2.vis_result(image, detections, classes)
                    Image.fromarray(annotated_image).show()

                observations_new[name] = copy.deepcopy(obs)
                observations_new[name]["detections"] = detections
            else:
                print(f"obs {name} has not found query: {query}!")

        if len(observations_new) == 0:
            print("No target object been catch!")
            return
        
        observations_new = OrderedDict(list(sorted(observations_new.items(), key=lambda item: item[1]['detections'].confidence.item(), reverse=True))[:3])
        return observations_new

    def get_lims(self, points):
        xmin = points[:, 0].min()
        xmax = points[:, 0].max()
        ymin = points[:, 1].min()
        ymax = points[:, 1].max()
        zmin = points[:, 2].min()
        zmax = points[:, 2].max()
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]
        return lims

    # The function is to calculate the qpos of the end effector given the front and up vector
    # front is defined as the direction of the gripper
    # Up is defined as the reverse direction with special shape
    def get_pose_from_front_up_end_effector(self, front, up):
        # Convert to numpy array
        if type(front) is list:
            front = np.array(front)
        if type(up) is list:
            up = np.array(up)
        front = front / np.linalg.norm(front)
        up = -up / np.linalg.norm(up)
        left = np.cross(up, front)
        left = left / np.linalg.norm(left)
        new_up = np.cross(front, left)

        rotation_mat = np.eye(3)
        rotation_mat[:3, :3] = np.stack([left, new_up, front], axis=1)
        # quat = mat2quat(rotation_mat)
        # return quat
        return rotation_mat

    def heuristics_pick_up_point(self, obj_points, obj_colors):
        front_direction = np.array([0, 0, -1])
        up_direction = np.array([0, 1, 0])
        # up_direction = np.array([1, 0, 0])
        print("use heuristics pick method.")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.colors = o3d.utility.Vector3dVector(obj_colors)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

        points = np.asarray(pcd.points)

        pick_point = self.get_object_center_top_point(point=points)
        
        # width in x coord
        object_Wx = points[:, 0].max() - points[:, 0].min()
        # width in y coord
        object_Wy = points[:, 1].max() - points[:, 1].min()
        # height
        object_H = points[:, 2].max() - points[:, 2].min()

        if object_H > 2 * max(object_Wx, object_Wy):
            pick_point[2] = points[:, 2].min() + 2 * object_H / 3
            # front_direction = np.array([1, 0, 0])
            front_direction = np.array([1, 0, -1])
        else:
            pick_point[2] -= min(0.05, object_H * 1 / 3)
            if object_Wy > object_Wx:
                up_direction = np.array([1, 0, 0])

        pick_rotation = self.get_pose_from_front_up_end_effector(
            front=front_direction, up=up_direction
        )
        pick_up_pose = np.eye(4)
        pick_up_pose[:3, 3] = pick_point
        pick_up_pose[:3, :3] = pick_rotation

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=0.02)
        cylinder_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        cylinder.rotate(cylinder_rot)
        cylinder.translate(pick_point)
        cylinder.paint_uniform_color([1, 192 / 255, 203 / 255])

        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, cylinder, coordinate])
        o3d.visualization.draw_geometries([pcd, cylinder])
        # o3d.visualization.draw_geometries([pcd])

        return pick_up_pose


    def pickup(self, observations, query):
        observations = self.process_observations(observations=observations, query=query)

        if len(observations) == 0:
            print(f"Error: no {query} in observations")
            exit(0)

        _points = []
        _colors = []
        _obj_points = []
        _obj_colors = []
        padding = 150
        for name, obs in observations.items():
            rgb = obs["rgb"]
            point = obs["point"]
            mask = obs["mask"]
            detections = obs["detections"]
            seg_mask = detections.mask[0]
            
            x_min, y_min, x_max, y_max = detections.xyxy[0]
            image_height, image_width = rgb.shape[:2]
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Apply the adjusted padding
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            mask_obj_range = np.zeros_like(mask, dtype=bool)
            mask_obj_range[y_min:y_max, x_min:x_max] = True

            mask_append = point[..., 2] < 1.5  # just for 1.2 meter range
            # mask = np.logical_and(mask, mask_append)
            mask_all = np.logical_and(mask, mask_obj_range, mask_append)

            pose_b = obs["c2b"]  # pose in coordinate system
            point_b = point @ pose_b[:3, :3].T + pose_b[:3, 3]
            _points.append(point_b[mask_all])
            _colors.append(rgb[mask_all])

            _obj_points.append(point_b[np.logical_and(mask, seg_mask)])
            _obj_colors.append(rgb[np.logical_and(mask, seg_mask)])


        _obj_points = np.vstack(_obj_points).astype(np.float32)
        _obj_colors = np.vstack(_obj_colors).astype(np.float32)


        _points = np.vstack(_points).astype(np.float32)
        _colors = np.vstack(_colors).astype(np.float32)

        transform = np.array([
            [ 0, -1,  0],
            [-1,  0,  0],
            [ 0,  0, -1],
        ])

        grasp_pose = np.array([
            [ 0,  0,  1],
            [ 0, -1,  0],
            [ 1,  0,  0],
        ])

        # _points = np.dot(_points, R_z[:3, :3]).astype(np.float32)
        _points = np.dot(_points, transform.T).astype(np.float32)

        if True:
            # dnoise point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(_points)
            pcd.colors = o3d.utility.Vector3dVector(_colors)
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
            _points = np.asarray(pcd.points).astype(np.float32)
            _colors = np.asarray(pcd.colors).astype(np.float32)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, coordinate])
        o3d.visualization.draw_geometries([pcd])

        lims = self.get_lims(points=_points)
        gg, cloud = self.grasping_model.get_grasp(_points, _colors, lims=lims)
        # lims = self.get_lims(points=points)
        # gg, cloud = self.grasping_model.get_grasp(points, colors, lims=lims)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            pick_up_pose = self.heuristics_pick_up_point(
                obj_points=_obj_points, 
                obj_colors=_obj_colors
            )
            return pick_up_pose

        gg = gg.nms().sort_by_score()
        # print(gg[0].score)

        transform_inv = np.eye(4)
        transform_inv[:3, :3] = transform
        transform_inv = np.linalg.inv(transform_inv)
        cloud.transform(transform_inv)
        gg_trans = copy.deepcopy(gg).transform(transform_inv) 
        # grippers = gg.to_open3d_geometry_list()
        # for gripper in grippers:
        #     gripper.transform(transform_inv) 
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([*grippers, cloud, coordinate])

        filter_gg = GraspGroup()
        filter_gg_idx_set = set()
        for name, obs in observations.items():
            height, width = obs["rgb"].shape[:2]
            pose_c = np.linalg.inv(obs["c2b"])  # pose in cam
            intrinsic = obs["intrinsic"]
            dist_coef = obs["dist_coef"]
            detections = obs["detections"]
            image = (obs["rgb"] * 255).astype(np.uint8)
            image_pil = Image.fromarray(image)
            image_draw = ImageDraw.Draw(image_pil)
            seg_mask = detections.mask[0]
            for gidx, grasp in enumerate(gg_trans):
                grasp_center = grasp.translation[np.newaxis]  # grasp center
                grasp_center_c = grasp_center @ pose_c[:3, :3].T + pose_c[:3, 3]
                if grasp_center_c[0, 2] <= 0:
                    continue
                # grasp in image position
                gix, giy = cv2.projectPoints(
                    grasp_center_c,
                    np.zeros(3),
                    np.zeros(3),
                    intrinsic,
                    dist_coef,
                )[0][:, 0, :][0].astype(np.int32)

                if not (0 <= giy <= height - 1 and 0 <= gix <= width - 1):
                    continue
                if 0 <= giy <= height - 1 and 0 <= gix <= width - 1 and seg_mask[giy, gix]:
                    if gidx not in filter_gg_idx_set:
                        filter_gg.add(gg[gidx])
                        filter_gg_idx_set.add(gidx)
                    image_draw.ellipse([(gix - 5, giy - 5), (gix + 5, giy + 5)], fill="green")
                else:
                    image_draw.ellipse([(gix - 5, giy - 5), (gix + 5, giy + 5)], fill="red")

            if self.is_visualize and False:
                image_pil.show()


        if len(filter_gg) == 0:
            print('No Grasp detected after filter!')
            pick_up_pose = self.heuristics_pick_up_point(
                obj_points=_obj_points, 
                obj_colors=_obj_colors,
            )
            return pick_up_pose

        best_pose = None
        best_gg = None
        min_cost = np.inf
        # second filter gripper by init pose to target consumption

        g_initial_rot = self.gripper_initial_pose[:3, :3]
        g_initial_trans = self.gripper_initial_pose[:3, 3]

        for idx, fg in enumerate(filter_gg):
            # print(idx)
            grasp_rot = fg.rotation_matrix
            grasp_tran = fg.translation
            depth = fg.depth
            grasp_matrix = np.eye(4)
            grasp_matrix[:3, :3] = grasp_rot @ grasp_pose
            grasp_matrix[:3, 3] = grasp_tran
            grasp_matrix = transform_inv @ grasp_matrix

            # rotation cost
            relative_rot = np.linalg.inv(g_initial_rot) @ grasp_matrix[:3, :3]
            rot_vector = R.from_matrix(relative_rot).as_rotvec()
            rot_cost = np.linalg.norm(rot_vector)

            # translation cost
            relative_trans = grasp_matrix[:3, 3] - g_initial_trans
            trans_cost = np.linalg.norm(relative_trans)

            cost = rot_cost * self.rot_cost_rate + trans_cost * (1 - self.rot_cost_rate)

            if cost < min_cost:
                min_cost = cost
                best_gg = fg
                best_pose = grasp_matrix

            # print(cost)
            # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([fg.transform(transform_inv).to_open3d_geometry(), cloud, coordinate])
        print(f"min cost is {min_cost}")

        # visualization
        if self.is_visualize:
            # grippers = grippers_base.to_open3d_geometry_list()
            # fgrippers = filter_gg.to_open3d_geometry_list()
            
            # filter_grippers= filter_gg.to_open3d_geometry_list()
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries(filter_gg.transform(transform_inv).to_open3d_geometry_list() + [cloud, coordinate])
            # o3d.visualization.draw_geometries([filter_grippers[0], cloud_base, coordinate])
            # o3d.visualization.draw_geometries([best_gg.transform(transform_inv).to_open3d_geometry(), cloud, coordinate])
            o3d.visualization.draw_geometries([best_gg.transform(transform_inv).to_open3d_geometry(), cloud])

        return best_pose
        

    def pick_back_object(self, back_observation: dict, query: str):
        back_observation = self.process_observations(back_observation, query=query)
        assert len(back_observation) == 1
        obs = back_observation[0]
        rgb = obs["rgb"]
        point = obs["point"]
        mask = obs["mask"]
        detections = obs["detections"]
        pose_b = obs["c2b"] 
        seg_mask = detections.mask[0]
        mask_append = point[..., 2] < 1.2
        mask_all = np.logical_and(mask, seg_mask, mask_append)
        
        point_b = point @ pose_b[:3, :3].T + pose_b[:3, 3]

        obj_points = point_b[mask_all]
        obj_colors = rgb[mask_all]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.colors = o3d.utility.Vector3dVector(obj_colors)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, coordinate])
        o3d.visualization.draw_geometries([pcd])

        points = np.asarray(pcd.points)
        pick_point = self.get_object_center_top_point(points)
        # pick_point = points[points[:, 2].argmax()]
        object_height = points[:, 2].max() - points[:, 2].min()
        pick_point[2] -= min(0.05, object_height * 1 / 3)
        pick_point[2] = max(pick_point[2], 0.33)

        front_direction = np.array([0, 0, -1])
        up_direction = np.array([0, -1, 0])

        # width in x coord
        object_Wx = points[:, 0].max() - points[:, 0].min()
        # width in y coord
        object_Wy = points[:, 1].max() - points[:, 1].min()

        if object_Wy > object_Wx:
            up_direction = np.array([-1, 0, 0])

        rotation_mat = self.get_pose_from_front_up_end_effector(
            front=front_direction, up=up_direction)

        pick_back_objct_pose = np.eye(4)
        pick_back_objct_pose[:3, 3] = pick_point
        pick_back_objct_pose[:3, :3] = rotation_mat

        return pick_back_objct_pose


    def get_object_center_top_point(self, point):
        points_xy = point[:, :2]
        points_xy_unique = np.unique(points_xy, axis=0)
        p_x = np.median(points_xy_unique[:, 0])
        p_y = np.median(points_xy_unique[:, 1])
        x_margin, y_margin = 0.1, 0.1

        x_mask = np.logical_and(point[:, 0] > (p_x - x_margin), point[:, 0] < (p_x + x_margin))
        y_mask = np.logical_and(point[:, 1] > (p_y - y_margin), point[:, 1] < (p_y + y_margin))
        z_mask = np.logical_and(point[:, 2] > self.robot_min_height, point[:, 2] < self.robot_max_height)
        place_mask = np.logical_and(x_mask, y_mask, z_mask)
        p_z = np.max(point[place_mask][:, 2])

        center_top_point = np.array(
            [p_x, p_y, p_z]
        )  # Final placing point in base coordinate system

        return center_top_point


    def place(self, observations, object1, object2):
        observations = self.process_observations(observations=observations, query=[object1, object2])

        place_point_list = []
        place_color_list = []
        for name, obs in observations.items():
            rgb = obs["rgb"]
            point = obs["point"]
            mask = obs["mask"]
            pose_b = obs["c2b"]  # pose in coordinate system
            point_b = point @ pose_b[:3, :3].T + pose_b[:3, 3]
            detections = obs["detections"]
            # process noise mask which belong to object1 (which is been pickuped)
            class_id = detections.class_id

            if 1 in class_id:
                range_mask = np.logical_and(point[..., 2] > 0.35, point[..., 2] < 1.2)
                target_detections = detections[np.where(class_id == 1)[0]]
                target_mask = target_detections.mask[0]
                mask = np.logical_and(range_mask, target_mask, mask)
                place_point_list.append(point_b[mask])
                place_color_list.append(rgb[mask])

                # Image.fromarray((rgb * 255).astype(np.uint8)).show()
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(point_b[mask])
                # pcd.colors = o3d.utility.Vector3dVector(rgb[mask])
                # o3d.visualization.draw_geometries([pcd])

        place_points = np.vstack(place_point_list)
        place_colors = np.vstack(place_color_list)

        place_point = self.get_object_center_top_point(place_points)

        place_point[2] += 0.05

        if self.is_visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(place_points)
            pcd.colors = o3d.utility.Vector3dVector(place_colors)

            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=0.02)
            cylinder_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            cylinder.rotate(cylinder_rot)
            cylinder.translate(place_point)
            cylinder.paint_uniform_color([1, 192 / 255, 203 / 255])
            # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, cylinder])
            # o3d.visualization.draw_geometries([pcd])

        front_direction = np.array([1/np.sqrt(2), 0, -1/np.sqrt(2)])
        up_direction = np.array([0, 1, 0])
        place_rotation = self.get_pose_from_front_up_end_effector(
           front=front_direction, up=up_direction
        )
        place_pose = np.eye(4)
        place_pose[:3, 3] = place_point
        place_pose[:3, :3] = place_rotation

        return place_pose

