
## Data generation pipeline

Data generation pipeline depends on a `SimulatorDataProcessor` that transforms the raw poses obtained from the ROS topics to the left camera coordinate frame. See UML diagram below to see the classes involved in this process. `DatasetSample` class has methods to visualized the objects' poses overlaid on the image plane. An example of these methods are shown at the bottom of `SimulatorDataProcesssor.py`.

<img src="./UML/data_generation.png" width="800">

## Adding additional data sources

If needed poses of new objects or cameras, this should be added in two different locations: in configuration block of `Rostopics.py` and as member attributes of the `RawSimulationData` class. To test the script after adding the new data sources use the main function in `SimulatorDataProcessor.py`. Additionally, `scripts/testing_scripts/` have scripts to test if the rostopics are sending data with a syncronized and a unsyncronized client. 

### Adding meshes of new objects

Meshes to new objects need to be reformatted to `.ply` format and expressed in mm. To change the scale of a mesh open it in meshlab and select `Filters > Normals, Curvatures and Orientation > Transform: Scale, normalize` and set the scale to 1000 with the uniform scaling option. After applying the object, you can get it back to screen by using the keyboard shortcut `Ctrl+h`. Meshes should only contain triangle faces.

Then mesh can be exported to `.ply` format by selecting `File > Export Mesh As...` and selecting the `.ply` format.

**Additional resources**
* https://gist.github.com/SeungBack/e71eac0faa52088e3038395fef684494 
* https://www.youtube.com/watch?v=6psAppbOOXM&ab_channel=MisterP.MeshLabTutorials