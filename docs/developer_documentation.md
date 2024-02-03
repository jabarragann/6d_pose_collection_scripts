
## Data generation pipeline

Data generation pipeline depends on a `SimulatorDataProcessor` that transforms the raw poses obtained from the ROS topics to the left camera coordinate frame. See UML diagram below in case additional data sources want to be added. `SimulatorDataProcessor` has also additional methods to visualized the objects poses overlaid on the image plane. An example of this methods are shown at the bottom of `SimulatorDataProcesssor.py`.

<img src="./UML/data_generation.png" width="800">
