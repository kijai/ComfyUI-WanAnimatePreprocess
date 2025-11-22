## ComfyUI helper nodes for [Wan video 2.2 Animate preprocessing](https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess)


Nodes to run the ViTPose model, get face crops and keypoint list for SAM2 segmentation.

Models:

to `ComfyUI/models/detection` (subject to change in the future)

YOLO:

https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/blob/main/process_checkpoint/det/yolov10m.onnx

ViTPose ONNX:

Use either the Large model from here:

https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx/wholebody

Or the Huge model like in the original code, it's split into two files due to ONNX file size limit:

Both files need to be in same directory, and the onnx file selected in the model loader:

`vitpose_h_wholebody_data.bin` and `vitpose_h_wholebody_model.onnx`

https://huggingface.co/Kijai/vitpose_comfy/tree/main/onnx


![example](example.png)


"PoseDataAdaptiveUpperBodyOffsetHelper": "Pose Data Adaptive Upper Body Offset Helper",
Anleitung für den Workflow
Damit das genau so funktioniert, wie du es willst ("Füße bleiben genau da, wo sie waren"):

Helper Node:

Stelle person_height_m (z.B. 1.70) und canvas_height_m (z.B. 2.20) ein.

Der Helper berechnet jetzt das head_padding relativ zu den aktuellen Füßen.

Editor Node (PoseDataEditorAutomaticOnlyTorsoHeadOffsetV2 o.ä.):

Verbinde head_padding vom Helper mit head_padding vom Editor.

Wichtig: Setze im Editor auto_feet_to_padding auf False.

Setze lock_feet (oder dont_offset_feet) auf True (dies ist meist Standard in den Offset-Modi, aber sicherstellen!).

offset_auto_duration_seconds kannst du mit duration_seconds vom Helper verbinden.

Ergebnis: Der Editor zieht den Oberkörper hoch/runter, bis der Kopf genau den berechneten Abstand zum oberen Rand hat. Da die Füße gelockt sind und das Padding basierend auf der aktuellen Fußposition berechnet wurde, wird die Person exakt auf die Proportion 1.70m / 2.20m gestreckt/gestaucht, ohne dass die Füße verrutschen.
