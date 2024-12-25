from dovsg.controller import Controller
from dovsg.utils.utils import vis_depth
import argparse

controller = Controller(
    step=0, 
    tags="test", 
    interval=3, 
    resolution=0.01,
    occ_avoid_radius=0.2,
    save_memory=True,
    debug=True
)

observations, correct_success = controller.get_align_observations(
    just_wrist=True,
    show_align=True,
    use_inlier_mask=True,
    self_align=False,
    align_to_world=False,
    save_name="test11",
)

if not correct_success:
    assert 1 == 0, "Init Pose Error!"

controller.pick_up(object1="cocacola", observations=observations, is_visualize=True)