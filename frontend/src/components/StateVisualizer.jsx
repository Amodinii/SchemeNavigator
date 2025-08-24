function StateVisualizer({ stage }) {
  if (!stage || stage === 'COMPLETED') return null;

  const stageColorMap = {
    METADATA: 'bg-yellow-500',
    PLANNER: 'bg-blue-500',
    TOOL_EXECUTION: 'bg-purple-500',
  };

  const stageLabel = {
    METADATA: 'Understanding your Query',
    PLANNER: 'Planning the next steps',
    TOOL_EXECUTION: 'Running the tools',
  };

  return (
    <div className="fixed top-6 right-6 z-50">
      <div
        className={`px-4 py-2 rounded-xl text-white shadow-lg animate-pulse transition-all duration-500 ease-in-out
        ${stageColorMap[stage] || 'bg-gray-600'}`}
      >
        {stageLabel[stage] || `Current stage: ${stage}`}
      </div>
    </div>
  );
}

export default StateVisualizer;