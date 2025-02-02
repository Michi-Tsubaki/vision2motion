#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Michitoshi Tsubaki <michi.tsubaki.tech@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""
Description: This node responds with an explanation for the input image in Japanese.
- input_topic_name :'input_topic' (default: 'image_raw')
- output_topic_name :'picture_explanation' (Explanation of captured image, String type)
- debug_output_topic_name : 'picture_explanation/debug_image' (Captured image with frame_id that is same as explanation, Image type)
- access_token :'hf_token' (default: 'default_token' <- [caution] this is unavailable!)
"""

# Import libraries and msg types for ROS2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Import libraries for image processing
from PIL import Image as PILImage
import io
import numpy as np

# Import libraries to use PaliGemma-3B (on Hugging Face) LLM model
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
import gc

from typing import Optional

class PaliGemmaExplainNode(Node):
    # Use Gogle/PaliGemma-3B Model
    MODEL_ID = "google/paligemma-3b-mix-224"
    
    def __init__(self):
        super().__init__("paligemma_explain_node")
        
        # Parameters
        self.declare_parameter('input_topic', 'image_raw')
        self.declare_parameter('hf_token', 'default_token')
        
        self._input_topic = self.get_parameter('input_topic').value
        self._token = self.get_parameter('hf_token').value
        
        # Initialize components
        self._bridge = CvBridge()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cuda":
            self.get_logger().info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            self.get_logger().warn("CUDA not available, using CPU")
        self._processor: Optional[AutoProcessor] = None
        self._model: Optional[PaliGemmaForConditionalGeneration] = None
        self._sequence_num = 1
        
        # Publishers and Subscribers
        self._pub = self.create_publisher(String, "picture_explanation", 5)
        self._processed_img_pub = self.create_publisher(Image, "picture_explanation/debug_image", 5)
        self._sub = self.create_subscription(
            Image,
            self._input_topic,
            self._image_callback,
            10)
        
        # Initialize model and processor
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            self.get_logger().info("Initializing processor...")
            self._processor = AutoProcessor.from_pretrained(
                self.MODEL_ID, 
                token=self._token
            )
            
            self.get_logger().info("Loading model...")
            # Optimize GPU use
            model_kwargs = {
                "torch_dtype": torch.float16 if self._device.type == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": "auto",
                "token": self._token,
            }
            
            if self._device.type == "cuda":
                # Optimize GPU use (additional)
                model_kwargs.update({
                    "max_memory": {0: "3GiB"},  # GPU Memory Restriction
                    "offload_folder": "offload", 
                })
            
            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                **model_kwargs
            ).eval()
            
            # Check memory use
            if self._device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                self.get_logger().info(f"GPU Memory: Allocated={memory_allocated:.1f}MB, Reserved={memory_reserved:.1f}MB")
            
            self.get_logger().info(f"Model initialization completed on {self._device}")
        
        except torch.cuda.OutOfMemoryError as e:
            self.get_logger().error(f"GPU out of memory: {str(e)}")
            rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize model: {str(e)}")
            rclpy.shutdown()

    def _image_callback(self, msg: Image) -> None:
        if not self._processor or not self._model:
            self.get_logger().error("Model not initialized")
            return

        try:
            # Convert ROS Image to PIL Image
            cv_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Generate explanation
            model_inputs = self._processor(
                text="<image>画像を説明して下さい",
                images=pil_image,
                return_tensors="pt",
                max_length=100
            ).to(self._device)
            
            with torch.inference_mode():
                response = self._model.generate(
                    **model_inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=1
                )
                
                response = response[0][model_inputs["input_ids"].shape[1]:]
                decoded_response = self._processor.decode(response, skip_special_tokens=True)
            
            # Publish result
            msg = String()
            msg.data = f'[{self._sequence_num}] {decoded_response}'
            self._pub.publish(msg)
            self.get_logger().info(f'Published explanation #{self._sequence_num}')

            try:
                # Create new image message
                img_msg = Image()
                img_msg.header = Header()
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = str(self._sequence_num)
                
                # Convert OpenCV image to ROS image
                cv_msg = self._bridge.cv2_to_imgmsg(cv_image, "bgr8")
                img_msg.height = cv_msg.height
                img_msg.width = cv_msg.width
                img_msg.encoding = cv_msg.encoding
                img_msg.is_bigendian = cv_msg.is_bigendian
                img_msg.step = cv_msg.step
                img_msg.data = cv_msg.data
                
                self._processed_img_pub.publish(img_msg)
                self.get_logger().info(f'Published processed image #{self._sequence_num}')
            except Exception as e:
                self.get_logger().error(f"Failed to publish image: {str(e)}")
            
            self._sequence_num += 1
            
        except cv2.error as e:
            self.get_logger().error(f"OpenCV error: {str(e)}")
        except torch.cuda.OutOfMemoryError:
            self.get_logger().error("CUDA out of memory")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PaliGemmaExplainNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
