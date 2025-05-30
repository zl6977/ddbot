document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const freeTextInput = document.getElementById("freeTextInput");
  const preprocessButton = document.getElementById("preprocessButton");
  const projectIdInput = document.getElementById("projectId");
  const userInteractionCheckbox = document.getElementById("userInteraction");
  const modelsHighLowInput = document.getElementById("modelsHighLow");
  const preprocessedMetadataBatchPreview = document.getElementById(
    "preprocessedMetadataBatchPreview"
  );
  const projectInfoPreview = document.getElementById("projectInfoPreview");
  const downloadButton = document.getElementById("downloadButton");
  const sendToServerButton = document.getElementById("sendToServerButton");

  let preprocessedMetadataBatch = {};
  let projectInfo = {}; // Initialize projectInfo
  let currentFileContent = null; // Store content of the last loaded file
  let currentFileType = null; // Store type of the last loaded file

  console.log("Initial projectInfo:", projectInfo);

  // Helper function to generate default Project ID
  const generateDefaultProjectId = () => {
    const date = new Date();
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, "0");
    return `ddbot_demo_${year}${month}`;
  };

  // Set initial Project ID on page load
  projectIdInput.value = generateDefaultProjectId();

  // Helper function to clear file-related state
  const clearFileState = () => {
    preprocessedMetadataBatch = {};
    preprocessedMetadataBatchPreview.textContent = "{}";
    currentFileContent = null;
    currentFileType = null;
  };

  // Function to process and display file content
  const processAndDisplayFile = (content, fileType, projectId) => {
    try {
      if (fileType === "xml") {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(content, "application/xml");
        preprocessedMetadataBatch = transformXMLToPreprocessedMetadata(
          xmlDoc,
          projectId
        );
        console.log("Transformed XML metadata:", preprocessedMetadataBatch);
      } else if (fileType === "yaml" || fileType === "yml") {
        const yamlData = jsyaml.load(content);
        console.log("Parsed YAML data:", yamlData);
        preprocessedMetadataBatch = transformYamlToPreprocessedMetadata(
          yamlData,
          projectId
        );
        console.log("Transformed YAML metadata:", preprocessedMetadataBatch);
      } else {
        alert("Unsupported file type. Please upload an XML or YAML file.");
        clearFileState();
        return;
      }
      preprocessedMetadataBatchPreview.textContent = JSON.stringify(
        preprocessedMetadataBatch,
        null,
        4
      );
    } catch (error) {
      alert("Error parsing file: " + error.message);
      console.error("Error processing file:", error);
      clearFileState();
    }
  };

  // Function to update projectInfo and its preview
  const updateProjectInfo = () => {
    let projectId = projectIdInput.value; // projectId always comes from the input field
    const userInteraction = userInteractionCheckbox.checked;
    const modelsHighLow = modelsHighLowInput.value
      .split(",")
      .map((s) => s.trim());

    projectInfo = {
      project_id: projectId,
      user_interaction_required: userInteraction,
      models: {
        high: modelsHighLow[0] || "",
        low: modelsHighLow[1] || "",
      },
    };
    projectInfoPreview.textContent = JSON.stringify(projectInfo, null, 4);
    console.log("Updated projectInfo:", projectInfo);

    // If a file has been loaded, re-process preprocessedMetadataBatch with the new projectId
    if (currentFileContent && projectId) {
      processAndDisplayFile(currentFileContent, currentFileType, projectId);
    }
  };

  // Initial update of projectInfo
  updateProjectInfo();

  // Event listeners for dynamic projectInfo updates
  projectIdInput.addEventListener("input", updateProjectInfo);
  userInteractionCheckbox.addEventListener("change", updateProjectInfo);
  modelsHighLowInput.addEventListener("input", updateProjectInfo);

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) {
      clearFileState();
      updateProjectInfo(); // Update project info if file is cleared
      return;
    }

    // Auto-generate projectId
    const fileNameWithoutExt = file.name.split(".").slice(0, -1).join(".");
    const date = new Date();
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, "0");
    projectIdInput.value = `${fileNameWithoutExt}_${year}${month}`;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      const fileType = file.name.split(".").pop().toLowerCase();

      currentFileContent = content; // Store content
      currentFileType = fileType; // Store type

      console.log("File type:", fileType);
      console.log("File content:", content);

      // Ensure projectInfo is updated with the file-derived projectId before transformations
      updateProjectInfo(); // This will trigger processAndDisplayFile if currentFileContent is set
    };
    reader.readAsText(file);
  });

  // XML to JSON conversion function
  function transformXMLToPreprocessedMetadata(xmlDoc, projectId) {
    const result = {};

    const logCurveInfos = xmlDoc.getElementsByTagName("logCurveInfo");
    for (const item of logCurveInfos) {
      const uid = item.getAttribute("uid");
      if (!uid) continue;

      const getText = (tag) => {
        const el = item.getElementsByTagName(tag)[0];
        return el ? el.textContent.trim() : "";
      };

      result[uid] = {
        Mnemonic: getText("mnemonic"),
        Description: getText("curveDescription"),
        DataType: getText("typeLogData"),
        Unit: getText("unit"),
        dataSource: getText("dataSource"),
        Namespace: "http://ddhub.demo/" + projectId.toString(),
      };
    }
    return result;
  }

  function transformYamlToPreprocessedMetadata(yamlData, projectId) {
    const transformed = {};
    if (Array.isArray(yamlData)) {
      yamlData.forEach((item) => {
        const mnemonic = item.mnemonic;
        if (mnemonic) {
          const knownFields = ["mnemonic", "description", "unit"];
          const entry = {
            Mnemonic: mnemonic,
            Description: item.description || "",
            Unit: item.unit || "",
            Namespace: "http://ddhub.demo/" + projectId.toString(),
          };

          // Add any other fields not explicitly listed
          for (const key in item) {
            if (item.hasOwnProperty(key) && !knownFields.includes(key)) {
              entry[key] = item[key];
            }
          }
          transformed[mnemonic] = entry;
        }
      });
    } else {
      // Handle single object YAML if necessary, or alert user about unsupported format
      console.warn("Unsupported YAML structure. Expected an array of objects.");
    }
    return transformed;
  }

  // Function to download JSON data as a file
  const downloadJsonFile = (data, filename) => {
    const blob = new Blob([JSON.stringify(data, null, 4)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Event listener for the Preprocess button
  preprocessButton.addEventListener("click", () => {
    const freeText = freeTextInput.value;
    const projectId = projectIdInput.value;

    if (!freeText.trim() && !projectId.trim()) {
      alert("Please enter some text or a Project ID to preprocess.");
      return;
    }

    const formattedProjectId = projectId.trim()
      ? `Namespace: ${projectId.trim()}`
      : "";
    const combinedText = `${freeText.trim()}\n${formattedProjectId.trim()}`.trim();

    if (!combinedText) {
      alert("No input provided for preprocessing.");
      return;
    }

    const serverEndpointBaseURL = `http://127.0.0.1:${serverEndpointPort}`;

    fetch(serverEndpointBaseURL + "/api/preprocess_user_input", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ combined_text: combinedText }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Server responded with status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        console.log("Preprocess response:", data);
        if (data && data.data_preprocessed) {
          preprocessedMetadataBatch = JSON.parse(data.data_preprocessed);
          preprocessedMetadataBatchPreview.textContent = JSON.stringify(
            preprocessedMetadataBatch,
            null,
            4
          );
        } else {
          preprocessedMetadataBatch = {};
          preprocessedMetadataBatchPreview.textContent = JSON.stringify(
            preprocessedMetadataBatch,
            null,
            4
          );
          console.warn("No 'data_preprocessed' field in server response.");
        }
      })
      .catch((error) => {
        console.error("Error sending text for preprocessing:", error);
        alert(
          "There was an error sending text for preprocessing: " + error.message
        );
      });
  });

  downloadButton.addEventListener("click", () => {
    // Download preprocessed_metadata_batch.json
    downloadJsonFile(
      preprocessedMetadataBatch,
      "preprocessed_metadata_batch.json"
    );
    // Download task_batch.json
    downloadJsonFile(projectInfo, "task_batch.json");
    alert("Files downloaded successfully.");
  });

  sendToServerButton.addEventListener("click", () => {
    // Combine projectInfo and preprocessedMetadataBatch into a single object
    const dataToSend = {
      projectInfo: projectInfo,
      preprocessedMetadataBatch: preprocessedMetadataBatch,
    };

    // Send combined data to server
    const serverEndpointBaseURL = `http://127.0.0.1:${serverEndpointPort}`;

    fetch(serverEndpointBaseURL + "/api/task_initializer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(dataToSend),
    })
      .then((response) => {
        if (response.ok) {
          // If the response is OK or redirected, navigate to the task monitor page
          // The server will redirect to /ui/task_monitor?project_id=...

          task_monitor_URL =
            serverEndpointBaseURL +
            "/ui/task_monitor" +
            `?project_id=${projectInfo.project_id}`;
          window.location.href = task_monitor_URL;
        } else {
          throw new Error(`Server responded with status: ${response.status}`);
        }
      })
      .catch((error) => {
        console.error("Error sending data to server:", error);
        alert(
          "There was an error sending data to the server: " + error.message
        );
      });
  });
});
