import rclpy
from rclpy.node import Node


class FlowerSubscriber(Node):
    def __init__(self, flower_train_fn):
        super().__init__("flower_subscriber")
        self.subscriptions = self.create_subscription(
            msg_type="uint8[]",  # Subject to changes
            topic="training_data",
            callback=self.listener_callback,
            raw=True,  # Store incoming data as raw bytes, could be useful if we are actuall sending
            # floats
        )
        self.subscription
        self.external_callback = flower_train_fn

    def listener_callback(self, msg):
        # TODO: link this to be handled in the flower client
        raise NotImplementedError
