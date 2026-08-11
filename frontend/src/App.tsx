
import { useState } from 'react';
import './App.css'
import { ChatInput } from "@/components/custom-ui/chat-input"


function App() 
{
    const [query, setQuery] = useState("")
    const [text, setText] = useState("")

    function onChange(e: any) 
    {
        e.preventDefault();
        setQuery(e.target.value);
    }

    function onClick(e: any) 
    {
        e.preventDefault();
        streamResponse(query, (chunk: string) => 
        {
            setText(prev => prev + chunk)
        })
    }

    return (
        <div className="flex flex-col p-4 gap-4 items-center h-screen bg-mist-300 App">
            <div className="max-w-xl flex-1 bg-gray-200">
                {text}
            </div>
            <ChatInput onChange={onChange} onClick={onClick} />
        </div>
    )
}


async function streamResponse(query: string, onChunk: (chunk: string) => void) 
{
    const baseUrl = import.meta.env.VITE_BACKEND_BASE_URL;
    const response = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
    });

    if (!response.body) throw new Error("No response body");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while(true)
    {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        onChunk(chunk);
    }
}

export default App
