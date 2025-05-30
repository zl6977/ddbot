import json
import logging
import os

import utils.configs.globals_config as glb
import utils.configs.log_config as log_config
import utils.task_manager as tm

# import utils.utils as ut
from flask import (  # redirect,; url_for,
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from flask_cors import CORS  # Import CORS

PORT = 5666  # random.randint(5000, 9999)
server_dir = glb.server_dir
ui_dir = glb.ui_dir
tasks_dir = glb.tasks_dir
default_log_path = glb.default_log_path
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=ui_dir)
CORS(app)  # Enable CORS for all routes


@app.route("/api/preprocess_user_input", methods=["POST"])
def process_data():
    if request.is_json:
        data = request.get_json()
        combined_text = data.get("combined_text", "")
        logger.info(f"Received combined_text for preprocessing: {combined_text}")
        # Here you would add your server-side processing logic
        preprocessed_metadata = tm.preprocess_free_user_input(combined_text)
        return jsonify({"data_before": data, "data_preprocessed": preprocessed_metadata}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


@app.route("/ui/task_initializer", methods=["GET"])
@app.route("/ui", methods=["GET"])
@app.route("/", methods=["GET"])
def task_initializer():
    # Construct the base URL for the task monitor, without the project_id
    # task_monitor_base_url = f"http://127.0.0.1:{PORT}/ui/task_monitor/"
    return render_template("task_initializer.html", serverEndpointPort=PORT)
    # return send_from_directory(ui_dir, "task_initializer.html")


@app.route("/api/task_initializer", methods=["POST"])
def create_task_initializer():
    if request.is_json:
        data = request.get_json()
        projectInfo = data["projectInfo"]
        preprocessedMetadataBatch = data["preprocessedMetadataBatch"]
        project_id = projectInfo["project_id"]  # Assuming project_id is part of projectInfo

        task_batch = tm.initialize_task_batch(preprocessedMetadataBatch)
        task_dir = os.path.join(tasks_dir, project_id)
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "task_batch.json"), "w") as file:
            json.dump(task_batch, file, indent=4)
        with open(os.path.join(task_dir, "projectInfo.json"), "w") as file:
            json.dump(projectInfo, file, indent=4)
        with open(os.path.join(task_dir, "preprocessedMetadataBatch.json"), "w") as file:
            json.dump(preprocessedMetadataBatch, file, indent=4)
        return jsonify({"message": "Data received successfully!", "data": data}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


@app.route("/api/task_batch/<string:project_id>", methods=["PUT"])
def update_task_batch(project_id):
    if request.is_json:
        data = request.get_json()
        task_batch = data

        task_dir = os.path.join(tasks_dir, project_id)
        task_batch_path = os.path.join(task_dir, "task_batch.json")

        if os.path.exists(task_batch_path):
            with open(os.path.join(task_dir, "task_batch.json"), "w") as file:
                json.dump(task_batch, file, indent=4)
            return jsonify({"message": "task_batch.json updated successfully!"}), 200
        else:
            return jsonify({"error": "task_batch.json does not exist. Wrong project_id?"}), 400
    else:
        return jsonify({"error": "Request must be JSON"}), 400


@app.route("/api/projectInfo/<string:project_id>", methods=["PUT"])
def update_projectInfo(project_id):
    if request.is_json:
        data = request.get_json()
        projectInfo = data["projectInfo"]

        task_dir = os.path.join(tasks_dir, project_id)
        projectInfo_path = os.path.join(task_dir, "projectInfo.json")

        if os.path.exists(projectInfo_path):
            with open(os.path.join(task_dir, "projectInfo.json"), "w") as file:
                json.dump(projectInfo, file, indent=4)
            return jsonify({"message": "projectInfo.json updated successfully!"}), 200
        else:
            return jsonify({"error": "projectInfo.json does not exist. Wrong project_id?"}), 400
    else:
        return jsonify({"error": "Request must be JSON"}), 400


@app.route("/api/task_batch", methods=["GET"])
def get_task_batch():
    project_id = request.args.get("project_id")  # 获取查询参数 ?projectId=xyz
    if not project_id:
        return jsonify({"error": "Missing project_id"}), 404
    task_dir = os.path.join(tasks_dir, project_id)
    task_batch_path = os.path.join(task_dir, "task_batch.json")

    if os.path.exists(task_batch_path):
        with open(task_batch_path, "r") as file:
            task_batch = json.load(file)
        return task_batch, 200
    else:
        print("File does not exist.")
        return jsonify({"error": f"No task_batch related to {project_id}"}), 400


@app.route("/ui/task_monitor", methods=["GET"])
def get_task_monitor():
    global default_log_path
    project_id = request.args.get("project_id")
    if not project_id:
        return send_from_directory(ui_dir, "task_monitor.html")  # Or handle error/redirect

    task_dir = os.path.join(tasks_dir, project_id)
    task_batch_path = os.path.join(task_dir, "task_batch.json")
    project_info_path = os.path.join(task_dir, "projectInfo.json")

    project_log_path = os.path.join(task_dir, "sessions.log")
    log_config.configure_logger(project_log_path, "a", True)

    task_batch = {}
    project_info = {}

    if os.path.exists(task_batch_path):
        with open(task_batch_path, "r") as file:
            task_batch = json.load(file)
    else:
        log_config.configure_logger(default_log_path, "a", True)
        return jsonify({"error": "Task batch not found"}), 404

    if os.path.exists(project_info_path):
        with open(project_info_path, "r") as file:
            project_info = json.load(file)
    # It's okay if projectInfo.json doesn't exist, we'll just pass an empty dict

    log_config.configure_logger(default_log_path, "a", True)
    return (
        render_template("task_monitor.html", jsonData=json.dumps(task_batch), projectInfoData=json.dumps(project_info)),
        200,
    )


@app.route("/api/interpretation", methods=["POST"])
def interpret():
    if request.is_json:
        global default_log_path

        data = request.get_json()
        project_id = data.get("project_id")
        # task_key = data.get("task_key")
        task_data = data.get("task_data")
        raw_content = task_data.get("Raw_content", "")
        Interpretation_user = task_data.get("Interpretation_user", "")
        task_control = task_data.get("TaskControl", "")
        CoT_flag = task_control.get("Chain_of_Thought", False)

        task_dir = os.path.join(tasks_dir, project_id)
        project_log_path = os.path.join(task_dir, "sessions.log")
        log_config.configure_logger(project_log_path, "a", True)

        logger.info(f"Received data for interpretation: {data}")

        llm_response, _ = tm.interpret_mnemonic(raw_content + " " + Interpretation_user, CoT_flag)
        Interpretation_user = Interpretation_user + llm_response

        # In a real scenario, you would process the task_key and user_input with an LLM
        # For now, simulate the LLM response by appending "LLM: " to the user_input
        interpretation_user = f"{Interpretation_user}\n"
        result = {"Interpretation_user": interpretation_user}

        log_config.configure_logger(default_log_path, "a", True)
        return jsonify(result), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400



@app.route("/api/run_task", methods=["POST"])
def run_task():
    if request.is_json:
        global default_log_path
        data = request.get_json()
        logger.info(f"Received data for running task: {data}")
        project_id = data.get("project_id")
        task_key = data.get("task_key")
        task_data = data.get("task_data")

        task_dir = os.path.join(tasks_dir, project_id)
        project_log_path = os.path.join(task_dir, "sessions.log")
        log_config.configure_logger(project_log_path, "a", True)

        if not project_id or not task_data:
            log_config.configure_logger(default_log_path, "a", True)
            return jsonify({"error": "Missing project_id or task_data"}), 400

        # Pass the entire task_data object to run_single_task
        result = tm.run_single_task(task_key, task_data)
        log_config.configure_logger(default_log_path, "a", True)
        return jsonify(result), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


@app.route("/api/confirm_recognition", methods=["POST"])
def confirm_recognition():
    if request.is_json:
        global default_log_path

        data = request.get_json()
        project_id = data.get("project_id")
        # task_key = data.get("task_key")
        task_data = data.get("task_data", {})

        task_dir = os.path.join(tasks_dir, project_id)
        project_log_path = os.path.join(task_dir, "sessions.log")
        log_config.configure_logger(project_log_path, "a", True)

        logger.info(f"Received data for confirm_recognition: {data}")

        if not project_id or not task_data:
            log_config.configure_logger(default_log_path, "a", True)
            return jsonify({"error": "Missing project_id or task_data"}), 400

        try:
            payload = tm.handle_user_interaction(project_id, task_data)
            # Backward-compatible message if not provided
            payload.setdefault("message", "Interpretation confirmed and post-confirm tasks executed.")
            status = 200
        except Exception as e:
            logger.exception("Error during confirm_interaction")
            payload = {"error": str(e)}
            status = 500

        log_config.configure_logger(default_log_path, "a", True)
        return jsonify(payload), status
    else:
        return jsonify({"error": "Request must be JSON"}), 400


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(ui_dir, "css"), filename)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(ui_dir, "js"), filename)


@app.route("/ui/<path:filename>")
def serve_ui_static(filename):
    return send_from_directory(ui_dir, filename)


if __name__ == "__main__":
    # tm.load_files()
    print(f"Starting Flask app on port {PORT}")
    app.run(debug=False, port=PORT)
