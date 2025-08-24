import { useState, useEffect } from 'react';
// Note: In a real project, you'd install this. For this environment, we load it via a script tag.
// We'll use a simple state management approach instead of relying on an external library for this example.

// --- Helper Components (Previously in separate files) ---

/**
 * Displays a list of chat messages.
 * @param {object} props - The component props.
 * @param {Array<object>} props.messages - The array of message objects.
 */
function ChatMessages({ messages }) {
  return (
    <div className="flex-grow overflow-y-auto p-4 space-y-4">
      {messages.map((msg, index) => (
        <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
          <div
            className={`max-w-lg px-4 py-2 rounded-xl shadow-md ${
              msg.role === 'user'
                ? 'bg-blue-500 text-white'
                : `bg-gray-700 text-gray-200 ${msg.loading ? 'animate-pulse' : ''}`
            }`}
          >
            {/* A simple way to render newlines */}
            {typeof msg.content === 'string' && msg.content.split('\n').map((line, i) => (
              <p key={i}>{line}</p>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Renders the chat input field and send button.
 * @param {object} props - The component props.
 * @param {string} props.newMessage - The current value of the input field.
 * @param {function} props.setNewMessage - Function to update the input field's value.
 * @param {function} props.submitNewMessage - Function to call when the message is submitted.
 * @param {boolean} props.isLoading - Whether the app is currently waiting for a response.
 */
function ChatInput({ newMessage, setNewMessage, submitNewMessage, isLoading }) {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitNewMessage();
    }
  };

  return (
    <div className="p-4 bg-gray-800 border-t border-gray-700">
      <div className="relative">
        <textarea
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          className="w-full p-3 pr-12 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          rows="1"
          disabled={isLoading}
        />
        <button
          onClick={submitNewMessage}
          disabled={isLoading || !newMessage.trim()}
          className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-blue-500 text-white rounded-full disabled:bg-gray-600 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="feather feather-send"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
        </button>
      </div>
    </div>
  );
}


// --- API Handler (Previously in a separate file) ---

// Explicitly define the backend server URL.
const API_BASE_URL = 'http://127.0.0.1:8000';

const api = {
  /**
   * Starts a new conversation.
   * @param {string} userQuery - The initial query from the user.
   * @returns {Promise<object>} - The response from the server.
   */
  startPipeline: async (userQuery) => {
    const response = await fetch(`${API_BASE_URL}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_query: userQuery }),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return response.json();
  },

  /**
   * Continues an existing conversation.
   * @param {string} userQuery - The follow-up query from the user.
   * @param {string} userId - The ID of the current user session.
   * @returns {Promise<object>} - The response from the server.
   */
  continuePipeline: async (userQuery, userId) => {
    const response = await fetch(`${API_BASE_URL}/continue`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_query: userQuery, user_id: userId }),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return response.json();
  },
};


// --- Main Chatbot Component ---

function App() {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [userId, setUserId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const getTimeBasedGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning!';
    if (hour < 18) return 'Good afternoon!';
    return 'Good evening!';
  };

  const submitNewMessage = async () => {
    const trimmedMessage = newMessage.trim();
    if (!trimmedMessage || isLoading) return;

    // Use a functional update to get the latest state
    setMessages(prevMessages => [
        ...prevMessages,
        { role: 'user', content: trimmedMessage },
        { role: 'assistant', content: 'Processing...', loading: true }
    ]);

    setNewMessage('');
    setIsLoading(true);

    try {
      let response;
      if (userId) {
        response = await api.continuePipeline(trimmedMessage, userId);
        console.log(`Follow-up Response from /continue:`, response);
      } else {
        response = await api.startPipeline(trimmedMessage);
        console.log(`Initial Response from /start:`, response);
        if (response.user_id) {
          setUserId(response.user_id);
        }
      }

      setMessages(prevMessages => {
          const newMessages = [...prevMessages];
          newMessages[newMessages.length - 1] = {
            role: 'assistant',
            content: response.message || 'Sorry, I could not get a response.',
            loading: false,
          };
          return newMessages;
      });

    } catch (err) {
      console.error("Error communicating with API:", err);
      setMessages(prevMessages => {
          const newMessages = [...prevMessages];
          newMessages[newMessages.length - 1] = {
            role: 'assistant',
            content: 'Something went wrong while processing your query.',
            loading: false,
          };
          return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className='relative grow flex flex-col h-full bg-gray-900 text-white font-sans'>
      {messages.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <p className="text-4xl font-semibold text-gray-500 text-center px-4">
            Hello there, {getTimeBasedGreeting()}
          </p>
        </div>
      )}
      <ChatMessages messages={messages} />
      <ChatInput
        newMessage={newMessage}
        isLoading={isLoading}
        setNewMessage={setNewMessage}
        submitNewMessage={submitNewMessage}
      />
    </div>
  );
}

export default App;
