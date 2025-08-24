import useAutosize from '@/hooks/useAutosize';
import sendIcon from '@/assets/images/send.svg';

function ChatInput({ newMessage, isLoading, setNewMessage, submitNewMessage }) {
  const textareaRef = useAutosize(newMessage);

  function handleKeyDown(e) {
    if (e.keyCode === 13 && !e.shiftKey && !isLoading) {
      e.preventDefault();
      submitNewMessage();
    }
  }

  return (
    <div className='sticky bottom-0 shrink-0 py-10'>
              <div className='pr-0.5 bg-gray-100 relative shrink-0 rounded-3xl overflow-hidden ring-gray-300 ring-1 focus-within:ring-2 transition-all'>
          <textarea
            className='block w-full max-h-[140px] py-2 px-4 pr-11 bg-gray-50 text-gray-800 rounded-3xl resize-none placeholder:text-gray-500 focus:outline-none'
            ref={textareaRef}
            rows='1'
            value={newMessage}
            onChange={e => setNewMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
          />
          <button
            className='absolute top-1/2 -translate-y-1/2 right-3 p-1 rounded-md hover:bg-gray-300'
            onClick={submitNewMessage}
            disabled={isLoading}
          >
            <img src={sendIcon} alt='send' />
          </button>
        </div>
      </div>
  );
}

export default ChatInput;
