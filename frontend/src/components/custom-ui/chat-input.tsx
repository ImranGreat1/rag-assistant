import { Plus, ArrowUpIcon } from "lucide-react";
import { 
    InputGroup, 
    InputGroupAddon, 
    InputGroupInput,
    InputGroupButton
} from "@/components/ui/input-group";


export function ChatInput({ onChange, onClick, value }: any) 
{
    return (
        <InputGroup className="fixed bottom-0 left-1/2 -translate-x-1/2 bg-stone-900 w-3xl rounded-2xl p-2 border-stone-700 text-stone-400 hover:border-stone-600 has-[[data-slot=input-group-control]:focus]:border-stone-600 has-[[data-slot=input-group-control]:focus-visible]:ring-0">
            <InputGroupInput onChange={onChange} value={value} placeholder="Search..."/>
            <InputGroupAddon align="block-end">
                <Plus className="text-stone-50" />
                <InputGroupButton className="ml-auto rounded-xl px-2 py-4 bg-stone-600" onClick={onClick}>
                    <ArrowUpIcon className="text-stone-50"/>
                </InputGroupButton>
            </InputGroupAddon>
        </InputGroup>
    )
}