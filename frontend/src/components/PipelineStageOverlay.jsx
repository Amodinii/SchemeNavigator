import React from "react";

const stageColors = {
  IDLE: "bg-gray-400",
  METADATA_EXTRACTION: "bg-yellow-400",
  PLANNING: "bg-blue-400",
  EXECUTION: "bg-indigo-400",
  COMPLETED: "bg-green-500",
  ERROR: "bg-red-500",
};

const PipelineStageOverlay = ({ stage }) => {
  if (!stage) return null;

  return (
    <div className="fixed top-4 right-4 z-50 shadow-lg">
      <div
        className={`px-4 py-2 rounded-xl text-white font-medium text-sm animate-slide-in ${stageColors[stage] || "bg-gray-500"}`}
      >
        🛠️ Current Stage: {stage.replace("_", " ")}
      </div>
    </div>
  );
};

export default PipelineStageOverlay;