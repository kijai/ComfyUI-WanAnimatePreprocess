# Agent Instructions & Project Context

## Project State: Splitting `nodes.py`

This repository originally contained a monolithic `nodes.py` file with over 30,000 lines of code. It was successfully refactored into a more maintainable, modular structure to fit within context limits and improve developer experience.

### What was done:
1. **Extraction of Classes:** All Custom Node classes inside `nodes.py` were programmatically extracted based on their `CATEGORY` attributes.
2. **Directory Structure:** They have been split into the `node_categories/` folder, replicating the category hierarchy:
   - For example, `CATEGORY = "WanAnimatePreprocess/Retargeting"` is now located at `node_categories/WanAnimatePreprocess/Retargeting.py`.
   - Single-level categories like `WanAnimatePreprocess` are located at `node_categories/WanAnimatePreprocess/nodes.py`.
3. **Re-Integration:** `nodes.py` still acts as the central hub. It retains all the original top-level utility functions, initial imports (up to line 89), and dynamically imports all the extracted classes from `node_categories/`.
4. **Mappings Maintained:** The dictionaries `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` were intentionally kept at the bottom of `nodes.py` as requested. 
5. **Location Reference Comments:** A comprehensive mapping table is located right above the `NODE_CLASS_MAPPINGS` dictionary in `nodes.py`. It shows exactly which file contains which class (e.g., `# PoseDataEditor -> node_categories/WanAnimatePreprocess/Editor.py`).

### How to deal with it going forward:
- **Adding new nodes:** If you add a new node, you can create it directly inside the appropriate module in `node_categories/` and then add the import and mapping into `nodes.py`.
- **Modifying existing nodes:** You must look up the class location using the reference comment in `nodes.py`, then edit the corresponding file inside `node_categories/`.
- **Imports:** Inside the extracted files in `node_categories/`, absolute-relative imports are used (e.g., `from ...models.onnx_models import ViTPose`). When adding new files inside these folders, maintain the `...` syntax for importing from the root of the custom node directory.
- **Top-level execution:** `__init__.py` continues to load from `nodes.py` without any required changes, ensuring complete compatibility with ComfyUI.
