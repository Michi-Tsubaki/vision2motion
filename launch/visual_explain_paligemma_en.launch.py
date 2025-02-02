from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare arguments
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='image_raw',
        description='Input topic for camera image'
    )
    
    token_arg = DeclareLaunchArgument(
        'hf_token',
        default_value='default_token',
        description='HuggingFace API token'
    )

    # Create node
    explain_node = Node(
        package='vision2motion',  # package name
        executable='visual_explain_paligemma_en',  # name of the excutable file
        name='visual_explain_paligemma_en',  # name of the node
        parameters=[{
            'input_topic': LaunchConfiguration('input_topic'),
            'hf_token': LaunchConfiguration('hf_token')
        }],
        #remappings=[
        #    (LaunchConfiguration('input_topic'), '/usb_cam/image_raw'),
        #]
    )

    return LaunchDescription([
        input_topic_arg,
        token_arg,
        explain_node
    ])
